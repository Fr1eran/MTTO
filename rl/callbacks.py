import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl.evaluation import (
    PolicyEvaluationResult,
    describe_best_update_reason,
    evaluate_policy_once,
    save_policy_evaluation_curve,
)


def _safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


@dataclass(frozen=True)
class BufferedScalarEvent:
    step: int
    scalars: dict[str, float]


class TensorboardCallback(BaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        tb_sample_interval_steps: int = 1,
        force_dump_interval_steps: int | None = None,
        batch_dump_records: int | None = None,
        async_dump: bool = True,
    ):
        super().__init__(verbose)
        self.min_tb_sample_interval_steps = max(1, int(tb_sample_interval_steps))
        self.force_dump_interval_steps = (
            None
            if force_dump_interval_steps is None or int(force_dump_interval_steps) <= 0
            else int(force_dump_interval_steps)
        )
        self.batch_dump_records = (
            None
            if batch_dump_records is None or int(batch_dump_records) <= 0
            else int(batch_dump_records)
        )
        self._last_sample_step: int = -self.min_tb_sample_interval_steps
        self._last_dump_step: int = 0
        self._pending_sample_records: int = 0
        self._pending_events: list[BufferedScalarEvent] = []
        self._episode_ids_by_env: list[int] = []
        self._tb_writer: Any = None
        self._async_dump = bool(async_dump)
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_futures: list[Future[None]] = []

    def _init_callback(self) -> None:
        self._tb_writer = self._resolve_tensorboard_writer()
        if self._tb_writer is not None and self._async_dump:
            self._start_async_writer()

    def _resolve_tensorboard_writer(self):
        output_formats = getattr(self.logger, "output_formats", None)
        if not isinstance(output_formats, (list, tuple)):
            return None
        for fmt in output_formats:
            writer = getattr(fmt, "writer", None)
            if (
                writer is not None
                and hasattr(writer, "add_scalar")
                and hasattr(writer, "flush")
            ):
                return writer
        return None

    def _start_async_writer(self) -> None:
        self._async_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tensorboard-callback-writer",
        )

    def _write_events_to_writer(
        self, writer: Any, events: tuple[BufferedScalarEvent, ...]
    ) -> None:
        for event in events:
            for tag, value in event.scalars.items():
                writer.add_scalar(tag, value, event.step)

        writer.flush()

    def _raise_completed_async_errors(self) -> None:
        if not self._async_futures:
            return

        pending_futures: list[Future[None]] = []
        for future in self._async_futures:
            if future.done():
                future.result()
            else:
                pending_futures.append(future)
        self._async_futures = pending_futures

    def _dispatch_events_to_writer(
        self, events: tuple[BufferedScalarEvent, ...]
    ) -> None:
        if not events:
            return

        if self._async_executor is None:
            self._write_events_to_writer(self._tb_writer, events)
            return

        self._raise_completed_async_errors()
        future = self._async_executor.submit(
            self._write_events_to_writer,
            self._tb_writer,
            events,
        )
        self._async_futures.append(future)

    def _finish_async_writer(self) -> None:
        if self._async_executor is None:
            return

        async_executor = self._async_executor
        try:
            for future in self._async_futures:
                future.result()
        finally:
            self._async_futures.clear()
            self._async_executor = None
            async_executor.shutdown(wait=True)

    def _sync_episode_tracker(self, num_envs: int) -> None:
        if num_envs < 0:
            return
        if len(self._episode_ids_by_env) != num_envs:
            self._episode_ids_by_env = [0] * num_envs

    def _advance_episode_tracker(self) -> None:
        dones_raw = self.locals.get("dones")
        if dones_raw is None:
            return

        try:
            dones = list(dones_raw)
        except TypeError:
            return

        self._sync_episode_tracker(len(dones))
        for env_idx, done in enumerate(dones):
            if bool(done):
                self._episode_ids_by_env[env_idx] += 1

    def _get_namespace_payloads_from_locals(
        self, namespace: str
    ) -> list[dict[str, float]]:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return []

        self._sync_episode_tracker(len(infos))

        payloads: list[dict[str, float]] = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            namespace_payload = info.get(namespace)
            if isinstance(namespace_payload, dict):
                payloads.append(namespace_payload)

        return payloads

    def _extract_all_namespace_payloads(self) -> dict[str, list[dict[str, float]]]:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return {}

        self._sync_episode_tracker(len(infos))

        buckets = {
            "rewards": [],
            "state": [],
            "constraint": [],
            "event": [],
            "basic": [],
        }
        for info in infos:
            if not isinstance(info, dict):
                continue
            for ns in buckets:
                payload = info.get(ns)
                if isinstance(payload, dict):
                    buckets[ns].append(payload)

        return {ns: pl for ns, pl in buckets.items() if pl}

    def _enrich_state_payloads(
        self, payloads: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        enriched_payloads: list[dict[str, float]] = []
        for env_idx, payload in enumerate(payloads):
            payload_copy = dict(payload)
            episode_id = (
                self._episode_ids_by_env[env_idx]
                if env_idx < len(self._episode_ids_by_env)
                else 0
            )
            payload_copy["episode_id"] = float(episode_id)
            enriched_payloads.append(payload_copy)

        return enriched_payloads

    def _collect_namespace_scalars(
        self, namespace: str, payloads: list[dict[str, float]]
    ) -> dict[str, float]:
        if not payloads:
            return {}
        if namespace == "state":
            payloads = self._enrich_state_payloads(payloads)

        aggregated_values: dict[str, list[float]] = {}
        for payload in payloads:
            for key, value in payload.items():
                try:
                    scalar_value = float(value)
                except (TypeError, ValueError):
                    continue
                aggregated_values.setdefault(key, []).append(scalar_value)

        namespace_scalars: dict[str, float] = {}
        for key, values in aggregated_values.items():
            if not values:
                continue
            namespace_scalars[f"{namespace}/{key}"] = sum(values) / len(values)

        return namespace_scalars

    def _build_sample_event(self, step: int) -> BufferedScalarEvent | None:
        all_payloads = self._extract_all_namespace_payloads()

        scalars: dict[str, float] = {}
        for namespace, payloads in all_payloads.items():
            namespace_scalars = self._collect_namespace_scalars(namespace, payloads)
            if namespace_scalars:
                scalars.update(namespace_scalars)

        if not scalars:
            return None

        return BufferedScalarEvent(step=step, scalars=scalars)

    def _record_all_namespaces_legacy(self) -> None:
        all_payloads = self._extract_all_namespace_payloads()
        for namespace, payloads in all_payloads.items():
            namespace_scalars = self._collect_namespace_scalars(namespace, payloads)
            for key, value in namespace_scalars.items():
                self.logger.record(key, value)

    def _record_namespace_legacy(self, namespace: str) -> None:
        payloads = self._get_namespace_payloads_from_locals(namespace)
        namespace_scalars = self._collect_namespace_scalars(namespace, payloads)
        for key, value in namespace_scalars.items():
            self.logger.record(key, value)

    def _flush_pending_events(self) -> None:
        if not self._pending_events:
            return

        writer = self._tb_writer
        if writer is None:
            for event in self._pending_events:
                for tag, value in event.scalars.items():
                    self.logger.record(tag, value)
                self.logger.dump(event.step)
            self._pending_events.clear()
            self._pending_sample_records = 0
            self._last_dump_step = int(self.num_timesteps)
            return

        events_to_flush = tuple(self._pending_events)
        self._dispatch_events_to_writer(events_to_flush)
        self._pending_events.clear()
        self._pending_sample_records = 0
        self._last_dump_step = int(self.num_timesteps)

    def _should_dump(self, current_step: int) -> bool:
        if (
            self.batch_dump_records is not None
            and self._pending_sample_records >= self.batch_dump_records
        ):
            return True

        if self.force_dump_interval_steps is not None:
            if (current_step - self._last_dump_step) >= self.force_dump_interval_steps:
                if self._tb_writer is None:
                    return True
                return bool(self._pending_events)

        return False

    def _on_step(self) -> bool:
        current_step = int(self.num_timesteps)

        should_sample = (
            current_step - self._last_sample_step
        ) >= self.min_tb_sample_interval_steps

        if should_sample:
            self._last_sample_step = current_step
            if self._tb_writer is None:
                self._record_all_namespaces_legacy()
                self._pending_sample_records += 1
            else:
                sample_event = self._build_sample_event(step=current_step)
                if sample_event is not None:
                    self._pending_events.append(sample_event)
                    self._pending_sample_records += 1

        if self._should_dump(current_step):
            if self._tb_writer is None:
                self.logger.dump(current_step)
                self._last_dump_step = current_step
                self._pending_sample_records = 0
            else:
                self._flush_pending_events()

        self._advance_episode_tracker()

        return True

    def _on_training_end(self) -> None:
        if self._pending_events:
            self._flush_pending_events()
            self._finish_async_writer()
            return

        if self._pending_sample_records > 0:
            current_step = int(self.num_timesteps)
            self.logger.dump(current_step)
            self._last_dump_step = current_step
            self._pending_sample_records = 0

        self._finish_async_writer()


class BestTrajectoryEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: Any,
        output_dir: str,
        artifact_metadata: dict[str, Any] | None = None,
        eval_trigger_mode: str = "steps",
        eval_trigger_interval: int = 10_000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if eval_trigger_mode not in ("steps", "episodes"):
            raise ValueError("trigger mode must be 'steps' or 'episodes'")
        if eval_trigger_interval <= 0:
            raise ValueError("trigger_interval must be positive")

        self.eval_env = eval_env
        self.output_dir = output_dir
        self.artifact_metadata = dict(artifact_metadata or {})
        self.eval_trigger_mode = eval_trigger_mode
        self.trigger_interval = int(eval_trigger_interval)
        self.deterministic = deterministic
        self.best_result: PolicyEvaluationResult | None = None
        self.best_trigger_interval: int | None = None
        self._episodes_seen = 0
        self._next_trigger_value = self.trigger_interval

    def _init_callback(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _count_completed_episodes(self) -> int:
        dones_raw = self.locals.get("dones")
        if dones_raw is None:
            return 0

        try:
            return sum(1 for done in dones_raw if bool(done))
        except TypeError:
            return int(bool(dones_raw))

    def _advance_trigger_threshold(self, current_value: int) -> None:
        while self._next_trigger_value <= current_value:
            self._next_trigger_value += self.trigger_interval

    def _save_best_artifacts(
        self,
        result: PolicyEvaluationResult,
        *,
        eval_trigger_interval: int,
        best_update_reason: str,
    ) -> None:
        model_path = os.path.join(self.output_dir, "best_model")
        vecnormalize_path = os.path.join(self.output_dir, "best_vecnormalize.pkl")
        trajectory_path = os.path.join(self.output_dir, "best_trajectory.npz")

        self.model.save(model_path)

        training_env = self.training_env
        save_training_env = getattr(training_env, "save", None)
        if not callable(save_training_env):
            raise TypeError(
                "Training environment does not support saving VecNormalize stats"
            )
        save_training_env(vecnormalize_path)

        extra_metrics = result.to_metrics(
            num_timesteps=int(self.num_timesteps),
            eval_trigger_mode=self.eval_trigger_mode,
            eval_trigger_interval=eval_trigger_interval,
        )
        extra_metrics.update(self.artifact_metadata)
        extra_metrics["trajectory_source"] = "best"
        extra_metrics["best_update_reason"] = best_update_reason
        save_policy_evaluation_curve(
            result,
            trajectory_path,
            extra_metrics=extra_metrics,
        )

    def _run_evaluation(self, *, eval_trigger_interval: int) -> None:
        result = evaluate_policy_once(
            self.model,
            self.eval_env,
            deterministic=self.deterministic,
        )

        self.logger.record("best_eval/last_success", float(result.success))
        self.logger.record("best_eval/last_total_reward", result.total_reward)
        self.logger.record("best_eval/last_stop_error_m", result.stop_error_m)
        self.logger.record("best_eval/last_time_error_s", result.time_error_s)
        self.logger.record("best_eval/last_total_energy_j", result.total_energy_j)

        best_update_reason = describe_best_update_reason(result, self.best_result)
        if best_update_reason is None:
            return

        self._save_best_artifacts(
            result,
            eval_trigger_interval=eval_trigger_interval,
            best_update_reason=best_update_reason,
        )
        self.best_result = result
        self.best_trigger_interval = eval_trigger_interval

        self.logger.record("best_eval/best_success", float(result.success))
        self.logger.record("best_eval/best_total_reward", result.total_reward)
        self.logger.record("best_eval/best_stop_error_m", result.stop_error_m)
        self.logger.record("best_eval/best_time_error_s", result.time_error_s)
        self.logger.record("best_eval/best_total_energy_j", result.total_energy_j)

        if self.verbose > 0:
            print(
                "New best trajectory saved: "
                f"mode={self.eval_trigger_mode}, "
                f"trigger_interval={eval_trigger_interval}, "
                f"success={result.success}, "
                f"total_reward={result.total_reward:.6f}"
            )

    def _on_step(self) -> bool:
        if self.eval_trigger_mode == "episodes":
            self._episodes_seen += self._count_completed_episodes()
            current_value = self._episodes_seen
        else:
            current_value = int(self.num_timesteps)

        if current_value < self._next_trigger_value:
            return True

        self._run_evaluation(eval_trigger_interval=current_value)
        self._advance_trigger_threshold(current_value)
        return True

    def _on_training_end(self) -> None:
        close_fn = getattr(self.eval_env, "close", None)
        if callable(close_fn):
            close_fn()


class EpisodeMetricsCollector(BaseCallback):
    def __init__(
        self,
        output_path: str,
        collect_interval_steps: int = 1024,
        record_trigger_mode: str = "steps",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if record_trigger_mode not in ("steps", "episodes"):
            raise ValueError(
                f"record_trigger_mode must be 'steps' or 'episodes', "
                f"got '{record_trigger_mode}'"
            )
        self._output_path = output_path
        self._record_trigger_mode = record_trigger_mode
        self._collect_interval = max(1, int(collect_interval_steps))
        self._last_collect_step: int = -self._collect_interval
        self._steps: list[int] = []
        self._rewards: list[float] = []
        self._lengths: list[float] = []
        self._episode_indices: list[int] = []

    def _on_step(self) -> bool:
        if self._record_trigger_mode == "episodes":
            return self._on_step_episodes_mode()
        return self._on_step_steps_mode()

    def _on_step_steps_mode(self) -> bool:
        current_step = int(self.num_timesteps)
        if current_step - self._last_collect_step < self._collect_interval:
            return True

        buf = getattr(self.model, "ep_info_buffer", None)
        if buf is None or len(buf) == 0 or len(buf[0]) == 0:
            return True

        reward_mean = _safe_mean([ep_info["r"] for ep_info in buf])
        len_mean = _safe_mean([ep_info["l"] for ep_info in buf])

        self._steps.append(current_step)
        self._rewards.append(reward_mean)
        self._lengths.append(len_mean)
        self._last_collect_step = current_step
        return True

    def _on_step_episodes_mode(self) -> bool:
        dones_raw = self.locals.get("dones")
        if dones_raw is None:
            return True

        try:
            dones = list(dones_raw)
        except TypeError:
            return True

        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return True

        for env_idx, done in enumerate(dones):
            if not bool(done):
                continue
            if env_idx >= len(infos):
                continue
            info = infos[env_idx]
            if not isinstance(info, dict):
                continue
            episode_info = info.get("episode")
            if not isinstance(episode_info, dict):
                continue
            r_val = episode_info.get("r")
            l_val = episode_info.get("l")
            if r_val is None or l_val is None:
                continue

            self._episode_indices.append(len(self._episode_indices))
            self._rewards.append(float(r_val))
            self._lengths.append(float(l_val))

        return True

    def _on_training_end(self) -> None:
        if self._record_trigger_mode == "episodes":
            if self._episode_indices:
                np.savez(
                    self._output_path,
                    index=np.asarray(self._episode_indices, dtype=np.float64),
                    ep_reward=np.asarray(self._rewards, dtype=np.float64),
                    ep_len=np.asarray(self._lengths, dtype=np.float64),
                )
        else:
            if self._steps:
                np.savez(
                    self._output_path,
                    index=np.asarray(self._steps, dtype=np.float64),
                    ep_reward=np.asarray(self._rewards, dtype=np.float64),
                    ep_len=np.asarray(self._lengths, dtype=np.float64),
                )
