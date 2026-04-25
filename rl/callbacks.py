import os
from dataclasses import dataclass
from typing import Any, Literal

from stable_baselines3.common.callbacks import BaseCallback

from rl.evaluation import (
    PolicyEvaluationResult,
    evaluate_policy_once,
    save_policy_evaluation_curve,
)


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

    def _collect_namespace_scalars(self, namespace: str) -> dict[str, float]:
        payloads = self._get_namespace_payloads_from_locals(namespace)
        if not payloads:
            return {}
        if namespace == "state":
            payloads = self._enrich_state_payloads(payloads)

        aggregated_values: dict[str, list[float]] = {}
        for payload in payloads:
            for key, value in payload.items():
                try:
                    scalar_value = float(value)
                except TypeError, ValueError:
                    continue
                aggregated_values.setdefault(key, []).append(scalar_value)

        namespace_scalars: dict[str, float] = {}
        for key, values in aggregated_values.items():
            if not values:
                continue
            namespace_scalars[f"{namespace}/{key}"] = sum(values) / len(values)

        return namespace_scalars

    def _build_sample_event(self, step: int) -> BufferedScalarEvent | None:
        scalars: dict[str, float] = {}
        for namespace in ("rewards", "state", "constraint", "event", "basic"):
            namespace_scalars = self._collect_namespace_scalars(namespace)
            if namespace_scalars:
                scalars.update(namespace_scalars)

        if not scalars:
            return None

        return BufferedScalarEvent(step=step, scalars=scalars)

    def _record_namespace_legacy(self, namespace: str) -> None:
        namespace_scalars = self._collect_namespace_scalars(namespace)
        for key, value in namespace_scalars.items():
            self.logger.record(key, value)

    def _get_tensorboard_writer(self):
        output_formats = getattr(self.logger, "output_formats", None)
        if not isinstance(output_formats, (list, tuple)):
            return None

        for output_format in output_formats:
            writer = getattr(output_format, "writer", None)
            if writer is None:
                continue
            if hasattr(writer, "add_scalar") and hasattr(writer, "flush"):
                return writer

        return None

    def _flush_pending_events(self) -> None:
        if not self._pending_events:
            return

        writer = self._get_tensorboard_writer()
        if writer is None:
            for event in self._pending_events:
                for tag, value in event.scalars.items():
                    self.logger.record(tag, value)
                self.logger.dump(event.step)
            self._pending_events.clear()
            self._pending_sample_records = 0
            self._last_dump_step = int(self.num_timesteps)
            return

        for event in self._pending_events:
            for tag, value in event.scalars.items():
                writer.add_scalar(tag, value, event.step)

        writer.flush()
        self._pending_events.clear()
        self._pending_sample_records = 0
        self._last_dump_step = int(self.num_timesteps)

    def _on_step(self) -> bool:
        current_step = int(self.num_timesteps)
        tensorboard_writer = self._get_tensorboard_writer()

        # 控制每次记录的最小步长间隔，避免高频写入造成 I/O 开销。
        should_sample = (
            current_step - self._last_sample_step
        ) >= self.min_tb_sample_interval_steps

        if should_sample:
            self._last_sample_step = current_step
            if tensorboard_writer is None:
                self._record_namespace_legacy(namespace="rewards")
                self._record_namespace_legacy(namespace="state")
                self._record_namespace_legacy(namespace="constraint")
                self._record_namespace_legacy(namespace="event")
                self._record_namespace_legacy(namespace="basic")
                self._pending_sample_records += 1
            else:
                sample_event = self._build_sample_event(step=current_step)
                if sample_event is not None:
                    self._pending_events.append(sample_event)
                    self._pending_sample_records += 1

        should_dump = False
        if (
            self.force_dump_interval_steps is not None
            and (tensorboard_writer is not None and self._pending_events)
            and (current_step - self._last_dump_step) >= self.force_dump_interval_steps
        ):
            should_dump = True

        if (
            self.force_dump_interval_steps is not None
            and tensorboard_writer is None
            and (current_step - self._last_dump_step) >= self.force_dump_interval_steps
        ):
            should_dump = True

        if (
            self.batch_dump_records is not None
            and self._pending_sample_records >= self.batch_dump_records
        ):
            should_dump = True

        if should_dump:
            if tensorboard_writer is None:
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
            return

        if self._pending_sample_records <= 0:
            return

        current_step = int(self.num_timesteps)
        self.logger.dump(current_step)
        self._last_dump_step = current_step
        self._pending_sample_records = 0


class BestTrajectoryEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: Any,
        output_dir: str,
        trigger_mode: Literal["steps", "episodes"] = "steps",
        trigger_interval: int = 10_000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if trigger_interval <= 0:
            raise ValueError("trigger_interval must be positive")

        self.eval_env = eval_env
        self.output_dir = output_dir
        self.trigger_mode = trigger_mode
        self.trigger_interval = int(trigger_interval)
        self.deterministic = deterministic
        self.best_result: PolicyEvaluationResult | None = None
        self.best_trigger_value: int | None = None
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

    def _is_new_best(self, result: PolicyEvaluationResult) -> bool:
        if self.best_result is None:
            return True
        return result.comparison_key() > self.best_result.comparison_key()

    def _save_best_artifacts(
        self,
        result: PolicyEvaluationResult,
        *,
        trigger_value: int,
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
            trigger_mode=self.trigger_mode,
            trigger_value=trigger_value,
        )
        save_policy_evaluation_curve(
            result,
            trajectory_path,
            extra_metrics=extra_metrics,
        )

    def _run_evaluation(self, *, trigger_value: int) -> None:
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

        if not self._is_new_best(result):
            return

        self._save_best_artifacts(result, trigger_value=trigger_value)
        self.best_result = result
        self.best_trigger_value = trigger_value

        self.logger.record("best_eval/best_success", float(result.success))
        self.logger.record("best_eval/best_total_reward", result.total_reward)
        self.logger.record("best_eval/best_stop_error_m", result.stop_error_m)
        self.logger.record("best_eval/best_time_error_s", result.time_error_s)
        self.logger.record("best_eval/best_total_energy_j", result.total_energy_j)

        if self.verbose > 0:
            print(
                "New best trajectory saved: "
                f"mode={self.trigger_mode}, trigger_value={trigger_value}, "
                f"success={result.success}, total_reward={result.total_reward:.6f}"
            )

    def _on_step(self) -> bool:
        if self.trigger_mode == "episodes":
            self._episodes_seen += self._count_completed_episodes()
            current_value = self._episodes_seen
        else:
            current_value = int(self.num_timesteps)

        if current_value < self._next_trigger_value:
            return True

        self._run_evaluation(trigger_value=current_value)
        self._advance_trigger_threshold(current_value)
        return True

    def _on_training_end(self) -> None:
        close_fn = getattr(self.eval_env, "close", None)
        if callable(close_fn):
            close_fn()
