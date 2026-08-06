import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th
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


class FixedReverseCurriculumCallback(BaseCallback):
    """Broadcast a deterministic reverse-curriculum reset distribution."""

    def __init__(
        self,
        *,
        remaining_distances_m: np.ndarray,
        whole_distance_m: float,
        total_timesteps: int,
        profile: Any,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        remaining = np.asarray(remaining_distances_m, dtype=np.float64)
        if remaining.ndim != 1 or remaining.size == 0:
            raise ValueError("remaining_distances_m must be a non-empty 1-D array")
        if not np.all(np.isfinite(remaining)) or np.any(remaining < 0.0):
            raise ValueError("remaining_distances_m must be finite and non-negative")
        if not np.isfinite(whole_distance_m) or whole_distance_m <= 0.0:
            raise ValueError("whole_distance_m must be finite and positive")
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if getattr(profile, "controller_kind", None) != "fixed_reverse":
            raise ValueError("profile must use the fixed_reverse controller")
        self._remaining_distances_m = remaining
        self._whole_distance_m = float(whole_distance_m)
        self._total_timesteps = int(total_timesteps)
        self._profile = profile
        self._version = 0
        self._rollout_index = 0
        self._start_index = int(np.argmax(remaining))
        self._current_weights = self._weights_for_progress(0.0)

    def initial_weights(self) -> np.ndarray:
        """Return the distribution required before SB3's first environment reset."""
        return self._current_weights.copy()

    def _on_rollout_start(self) -> None:
        if self._rollout_index > 0:
            progress = min(1.0, float(self.num_timesteps) / self._total_timesteps)
            weights = self._weights_for_progress(progress)
            if not np.array_equal(weights, self._current_weights):
                self._version += 1
                self.training_env.env_method(
                    "set_reference_initial_state_distribution",
                    weights,
                    version=self._version,
                )
                self._current_weights = weights
        self._rollout_index += 1

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # Do not alter workers here: only release the schedule data retained by
        # the callback after learning has finished.
        self._remaining_distances_m = np.empty(0, dtype=np.float64)

    def _weights_for_progress(self, progress: float) -> np.ndarray:
        start = float(self._profile.expansion_start_ratio)
        end = float(self._profile.expansion_end_ratio)
        start_only = float(self._profile.start_only_ratio)
        minimum = float(self._profile.min_remaining_distance_m)
        base_start_probability = float(self._profile.base_start_probability)
        initial_cap = min(
            float(self._profile.initial_max_remaining_distance_m),
            self._whole_distance_m,
        )

        if progress < start:
            cap = initial_cap
            start_probability = base_start_probability
        elif progress < end:
            ratio = (progress - start) / max(end - start, 1e-12)
            cap = initial_cap + ratio * (self._whole_distance_m - initial_cap)
            start_probability = base_start_probability
        elif progress < start_only:
            cap = self._whole_distance_m
            ratio = (progress - end) / max(start_only - end, 1e-12)
            start_probability = (
                base_start_probability + (1.0 - base_start_probability) * ratio
            )
        else:
            result = np.zeros_like(self._remaining_distances_m)
            result[self._start_index] = 1.0
            return result

        base_mask = (self._remaining_distances_m >= minimum) & (
            self._remaining_distances_m <= cap
        )
        if not np.any(base_mask):
            raise ValueError(
                "fixed reverse curriculum range contains no eligible nodes"
            )
        base_weights = base_mask.astype(np.float64)
        base_weights /= float(np.sum(base_weights))
        if not 0.0 <= start_probability <= 1.0:
            raise ValueError("base_start_probability must be within [0, 1]")
        start_weights = np.zeros_like(base_weights)
        start_weights[self._start_index] = 1.0
        return (
            1.0 - start_probability
        ) * base_weights + start_probability * start_weights


@dataclass(frozen=True)
class _SPDLContextSample:
    reference_index: int


class SPDLReferenceCurriculumCallback(BaseCallback):
    """Discrete SPDL driven by Critic values over the full reference-context pool."""

    _KL_TOLERANCE = 1e-10
    _TARGET_KL_TOLERANCE = 1e-8

    def __init__(
        self,
        *,
        remaining_distances_m: np.ndarray,
        reference_observations: np.ndarray,
        gamma: float,
        profile: Any,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        remaining = np.asarray(remaining_distances_m, dtype=np.float64)
        observations = np.asarray(reference_observations, dtype=np.float32)
        if remaining.ndim != 1 or remaining.size == 0:
            raise ValueError("remaining_distances_m must be a non-empty 1-D array")
        if not np.all(np.isfinite(remaining)) or np.any(remaining < 0.0):
            raise ValueError("remaining_distances_m must be finite and non-negative")
        if observations.ndim != 2 or observations.shape[0] != remaining.size:
            raise ValueError(
                "reference_observations must have one row per eligible reference node"
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("reference_observations must be finite")
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if getattr(profile, "controller_kind", None) != "spdl":
            raise ValueError("profile must use the spdl controller")

        self._remaining_distances_m = remaining
        self._reference_observations = observations
        self._gamma = float(gamma)
        self._profile = profile
        self._start_index = int(np.argmax(remaining))
        self._validate_profile()
        self._target_weights = self._build_target_weights()
        self._current_weights = self._build_initial_weights()
        self._version = 0
        self._rollout_index = 0
        self._context_update_count = 0
        self._context_samples: list[_SPDLContextSample] = []
        self._completed_returns: list[float] = []
        self._active_context_by_env: dict[int, tuple[int, int, float, float]] = {}

    def initial_weights(self) -> np.ndarray:
        """Return the all-support distribution injected before the first reset."""
        return self._current_weights.copy()

    @property
    def target_weights(self) -> np.ndarray:
        return self._target_weights.copy()

    def _on_rollout_start(self) -> None:
        if self._rollout_index > 0:
            self._maybe_update_distribution()
        self._rollout_index += 1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones_raw = self.locals.get("dones", [])
        if not isinstance(infos, (list, tuple)):
            return True
        try:
            dones = list(dones_raw)
        except TypeError:
            dones = [bool(dones_raw)]
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)

        for env_index, info in enumerate(infos):
            if not isinstance(info, dict) or env_index >= reward_values.size:
                continue
            sample = self._context_from_info(info)
            if sample is None:
                continue
            sample_id, reference_index = sample
            if not 0 <= reference_index < self._current_weights.size:
                continue
            active = self._active_context_by_env.get(env_index)
            if active is None or active[0] != sample_id:
                self._context_samples.append(_SPDLContextSample(reference_index))
                active = (sample_id, reference_index, 0.0, 1.0)

            _, active_index, total_return, discount = active
            total_return += discount * float(reward_values[env_index])
            discount *= self._gamma
            self._active_context_by_env[env_index] = (
                sample_id,
                active_index,
                total_return,
                discount,
            )
            if env_index < len(dones) and bool(dones[env_index]):
                self._completed_returns.append(total_return)
                self._active_context_by_env.pop(env_index, None)
        return True

    def _on_training_end(self) -> None:
        self._context_samples.clear()
        self._completed_returns.clear()
        self._active_context_by_env.clear()
        self._reference_observations = np.empty((0, 0), dtype=np.float32)

    def _maybe_update_distribution(self) -> None:
        min_context_samples = int(self._profile.spdl_min_context_samples)
        if len(self._context_samples) < min_context_samples:
            return

        warmup_updates = int(self._profile.spdl_alpha_warmup_updates)
        alpha = 0.0
        if self._context_update_count >= warmup_updates:
            min_completed = int(self._profile.spdl_min_completed_episodes)
            if len(self._completed_returns) < min_completed:
                return
            mean_return = float(np.mean(self._completed_returns))
            target_kl = self._kl_divergence(self._current_weights, self._target_weights)
            if target_kl <= self._TARGET_KL_TOLERANCE:
                self._clear_update_buffers()
                return
            alpha = (
                float(self._profile.spdl_zeta)
                * max(0.0, mean_return)
                / target_kl
            )

        coefficients = self._all_context_critic_values()
        candidate = self._solve_distribution(coefficients, alpha)
        self._context_update_count += 1
        self._clear_update_buffers()
        if np.allclose(candidate, self._current_weights, rtol=1e-10, atol=1e-12):
            return

        self._version += 1
        self.training_env.env_method(
            "set_reference_initial_state_distribution",
            candidate,
            version=self._version,
        )
        self._current_weights = candidate

    def _all_context_critic_values(self) -> np.ndarray:
        """Estimate $a_i$ for every eligible context with the current Critic."""
        indices = np.arange(self._current_weights.size, dtype=np.int64)
        return self._critic_values(indices)

    def _critic_values(self, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)
        observations = np.asarray(self._reference_observations[indices], dtype=np.float32)
        policy = self.model.policy
        observation_tensor, _ = policy.obs_to_tensor(observations)
        with th.no_grad():
            values = policy.predict_values(observation_tensor)
        return values.detach().cpu().numpy().reshape(-1).astype(np.float64)

    def _solve_distribution(self, coefficients: np.ndarray, alpha: float) -> np.ndarray:
        bound = float(self._profile.spdl_relative_entropy_bound)
        if alpha > 0.0:
            unconstrained = self._weights_for_dual(coefficients, alpha, 0.0)
            if self._kl_divergence(unconstrained, self._current_weights) <= bound:
                return unconstrained

        lower = 0.0
        upper = 1.0
        while (
            self._kl_divergence(
                self._weights_for_dual(coefficients, alpha, upper),
                self._current_weights,
            )
            > bound
        ):
            upper *= 2.0
            if upper > 1e12:
                raise RuntimeError("could not satisfy the SPDL relative-entropy bound")

        for _ in range(80):
            middle = (lower + upper) / 2.0
            candidate = self._weights_for_dual(coefficients, alpha, middle)
            if self._kl_divergence(candidate, self._current_weights) > bound:
                lower = middle
            else:
                upper = middle
        return self._weights_for_dual(coefficients, alpha, upper)

    def _weights_for_dual(
        self, coefficients: np.ndarray, alpha: float, dual: float
    ) -> np.ndarray:
        denominator = alpha + dual
        if denominator <= 0.0:
            raise ValueError("SPDL dual denominator must be positive")
        log_weights = (
            coefficients / denominator
            + alpha / denominator * np.log(self._target_weights)
            + dual / denominator * np.log(self._current_weights)
        )
        log_weights -= float(np.max(log_weights))
        weights = np.maximum(np.exp(log_weights), np.finfo(np.float64).tiny)
        return weights / float(np.sum(weights))

    def _build_initial_weights(self) -> np.ndarray:
        easy_mask = (
            self._remaining_distances_m >= float(self._profile.min_remaining_distance_m)
        ) & (
            self._remaining_distances_m
            <= float(self._profile.initial_max_remaining_distance_m)
        )
        if not np.any(easy_mask):
            raise ValueError("SPDL initial easy range contains no eligible nodes")
        easy = easy_mask.astype(np.float64)
        easy /= float(np.sum(easy))
        start = np.zeros_like(easy)
        start[self._start_index] = 1.0
        uniform = np.full_like(easy, 1.0 / easy.size)
        weights = (
            float(self._profile.spdl_initial_easy_mass) * easy
            + float(self._profile.spdl_initial_start_mass) * start
            + float(self._profile.spdl_initial_uniform_mass) * uniform
        )
        return weights / float(np.sum(weights))

    def _build_target_weights(self) -> np.ndarray:
        uniform_mass = float(self._profile.spdl_target_uniform_mass)
        target = np.full(
            self._current_size(), uniform_mass / self._current_size(), dtype=np.float64
        )
        target[self._start_index] += 1.0 - uniform_mass
        return target / float(np.sum(target))

    def _current_size(self) -> int:
        return int(self._remaining_distances_m.size)

    def _validate_profile(self) -> None:
        masses = np.asarray(
            [
                self._profile.spdl_initial_easy_mass,
                self._profile.spdl_initial_start_mass,
                self._profile.spdl_initial_uniform_mass,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("SPDL initial distribution masses must be non-negative")
        if not np.isclose(float(np.sum(masses)), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("SPDL initial distribution masses must sum to one")
        target_uniform_mass = float(self._profile.spdl_target_uniform_mass)
        if not 0.0 < target_uniform_mass < 1.0:
            raise ValueError("SPDL target uniform mass must be within (0, 1)")
        if int(self._profile.spdl_alpha_warmup_updates) < 0:
            raise ValueError("SPDL alpha warm-up updates must be non-negative")
        if not 0.0 < float(self._profile.spdl_relative_entropy_bound):
            raise ValueError("SPDL relative-entropy bound must be positive")
        if not 0.0 <= float(self._profile.spdl_zeta):
            raise ValueError("SPDL zeta must be non-negative")
        if int(self._profile.spdl_min_context_samples) <= 0:
            raise ValueError("SPDL minimum context sample count must be positive")
        if int(self._profile.spdl_min_completed_episodes) <= 0:
            raise ValueError("SPDL minimum completed episode count must be positive")

    @staticmethod
    def _context_from_info(info: dict[str, Any]) -> tuple[int, int] | None:
        try:
            return (
                int(info["reference_context_sample_id"]),
                int(info["reference_context_index"]),
            )
        except KeyError, TypeError, ValueError:
            return None

    def _clear_update_buffers(self) -> None:
        self._context_samples.clear()
        self._completed_returns.clear()

    @staticmethod
    def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        smallest = np.finfo(np.float64).tiny
        safe_left = np.maximum(left, smallest)
        safe_right = np.maximum(right, smallest)
        return float(np.sum(safe_left * (np.log(safe_left) - np.log(safe_right))))


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
            "outcome": [],
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

    def _enrich_basic_payloads(
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
        if namespace == "basic":
            payloads = self._enrich_basic_payloads(payloads)

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


class BestTrajectoryRecorder:
    def __init__(
        self,
        *,
        output_dir: str,
        artifact_metadata: dict[str, Any] | None = None,
    ):
        self.output_dir = output_dir
        self.artifact_metadata = dict(artifact_metadata or {})
        self.best_result: PolicyEvaluationResult | None = None
        self.best_trigger_interval: int | None = None

    def init_callback(self, callback: BaseCallback) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_best_artifacts(
        self,
        callback: BaseCallback,
        result: PolicyEvaluationResult,
        *,
        eval_trigger_interval: int,
        best_update_reason: str,
    ) -> None:
        model_path = os.path.join(self.output_dir, "best_model")
        trajectory_path = os.path.join(self.output_dir, "best_trajectory.npz")

        callback.model.save(model_path)

        extra_metrics = result.to_metrics(
            num_timesteps=int(callback.num_timesteps),
            eval_trigger_mode=getattr(callback, "eval_trigger_mode", None),
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

    def on_evaluation(
        self,
        callback: BaseCallback,
        result: PolicyEvaluationResult,
        *,
        eval_trigger_interval: int,
    ) -> None:
        callback.logger.record("best_eval/last_success", float(result.success))
        callback.logger.record(
            "best_eval/last_precise_arrival", float(result.precise_arrival)
        )
        callback.logger.record(
            "best_eval/last_punctual_arrival", float(result.punctual_arrival)
        )
        callback.logger.record("best_eval/last_total_reward", result.total_reward)
        callback.logger.record("best_eval/last_stop_error_m", result.stop_error_m)
        callback.logger.record("best_eval/last_time_error_s", result.time_error_s)
        callback.logger.record("best_eval/last_total_energy_j", result.total_energy_j)

        best_update_reason = describe_best_update_reason(result, self.best_result)
        if best_update_reason is None:
            return

        self._save_best_artifacts(
            callback,
            result,
            eval_trigger_interval=eval_trigger_interval,
            best_update_reason=best_update_reason,
        )
        self.best_result = result
        self.best_trigger_interval = eval_trigger_interval

        callback.logger.record("best_eval/best_success", float(result.success))
        callback.logger.record(
            "best_eval/best_precise_arrival", float(result.precise_arrival)
        )
        callback.logger.record(
            "best_eval/best_punctual_arrival", float(result.punctual_arrival)
        )
        callback.logger.record("best_eval/best_total_reward", result.total_reward)
        callback.logger.record("best_eval/best_stop_error_m", result.stop_error_m)
        callback.logger.record("best_eval/best_time_error_s", result.time_error_s)
        callback.logger.record("best_eval/best_total_energy_j", result.total_energy_j)

        if callback.verbose > 0:
            print(
                "New best trajectory saved: "
                f"mode={getattr(callback, 'eval_trigger_mode', None)}, "
                f"trigger_interval={eval_trigger_interval}, "
                f"success={result.success}, "
                f"total_reward={result.total_reward:.6f}"
            )


class SafetyViolationPositionRecorder(BaseCallback):
    SPEED_LOW_VIOLATION_CODE = 2
    SPEED_HIGH_VIOLATION_CODE = 3

    def __init__(
        self,
        *,
        output_path: str,
        position_bin_size_m: float = 5000.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_path = str(output_path)
        self.position_bin_size_m = max(1.0, float(position_bin_size_m))
        self._sample_exposure_by_bin: dict[int, int] = {}
        self._sample_violation_by_bin: dict[int, int] = {}
        self._episode_exposure_by_bin: dict[int, int] = {}
        self._episode_violation_by_bin: dict[int, int] = {}
        self._safety_truncation_by_bin: dict[int, int] = {}
        self._episode_bins_by_env: list[set[int]] = []
        self._episode_violation_bins_by_env: list[set[int]] = []

    def _init_callback(self) -> None:
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _sync_episode_trackers(self, num_envs: int) -> None:
        if num_envs < 0:
            return
        while len(self._episode_bins_by_env) < num_envs:
            self._episode_bins_by_env.append(set())
            self._episode_violation_bins_by_env.append(set())
        if len(self._episode_bins_by_env) > num_envs:
            self._episode_bins_by_env = self._episode_bins_by_env[:num_envs]
            self._episode_violation_bins_by_env = self._episode_violation_bins_by_env[
                :num_envs
            ]

    def _bin_index_for_position(self, position: float) -> int:
        return int(np.floor(float(position) / self.position_bin_size_m))

    @staticmethod
    def _increment(counter: dict[int, int], bin_index: int, value: int = 1) -> None:
        counter[bin_index] = counter.get(bin_index, 0) + int(value)

    @classmethod
    def _is_safety_violation(cls, constraint: dict[str, Any]) -> bool:
        violation_code = int(round(float(constraint.get("violation_code", 0.0))))
        if violation_code in {
            cls.SPEED_LOW_VIOLATION_CODE,
            cls.SPEED_HIGH_VIOLATION_CODE,
        }:
            return True

        margin_to_vmax = constraint.get("margin_to_vmax_mps")
        margin_to_vmin = constraint.get("margin_to_vmin_mps")
        try:
            margin_high = float(margin_to_vmax)
            margin_low = float(margin_to_vmin)
        except TypeError, ValueError:
            return False
        return bool(margin_high < 0.0 or margin_low < 0.0)

    def _extract_position(self, info: dict[str, Any]) -> float | None:
        basic = info.get("basic")
        if isinstance(basic, dict) and "position" in basic:
            return float(basic["position"])
        # Compatibility with diagnostics emitted before basic_info absorbed
        # the state namespace.
        state = info.get("state")
        if isinstance(state, dict) and "position" in state:
            return float(state["position"])
        if "position" in info:
            return float(info["position"])
        return None

    def _flush_episode(self, env_idx: int) -> None:
        if env_idx >= len(self._episode_bins_by_env):
            return
        for bin_index in self._episode_bins_by_env[env_idx]:
            self._increment(self._episode_exposure_by_bin, bin_index)
        for bin_index in self._episode_violation_bins_by_env[env_idx]:
            self._increment(self._episode_violation_by_bin, bin_index)
        self._episode_bins_by_env[env_idx] = set()
        self._episode_violation_bins_by_env[env_idx] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return True

        self._sync_episode_trackers(len(infos))
        dones_raw = self.locals.get("dones", [])
        try:
            dones = list(dones_raw)
        except TypeError:
            dones = [bool(dones_raw)]

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            constraint = info.get("constraint")
            if not isinstance(constraint, dict):
                continue
            try:
                position = self._extract_position(info)
            except TypeError, ValueError:
                continue
            if position is None:
                continue

            bin_index = self._bin_index_for_position(position)
            is_violation = self._is_safety_violation(constraint)
            outcome = info.get("outcome")
            is_truncated = (
                bool(outcome.get("truncated", False))
                if isinstance(outcome, dict)
                else bool(constraint.get("is_truncated", False))
            )

            self._increment(self._sample_exposure_by_bin, bin_index)
            self._episode_bins_by_env[env_idx].add(bin_index)
            if is_violation:
                self._increment(self._sample_violation_by_bin, bin_index)
                self._episode_violation_bins_by_env[env_idx].add(bin_index)
                if is_truncated:
                    self._increment(self._safety_truncation_by_bin, bin_index)

            if env_idx < len(dones) and bool(dones[env_idx]):
                self._flush_episode(env_idx)

        return True

    def _on_training_end(self) -> None:
        for env_idx in range(len(self._episode_bins_by_env)):
            self._flush_episode(env_idx)

        all_bins = sorted(
            set(self._sample_exposure_by_bin)
            | set(self._sample_violation_by_bin)
            | set(self._episode_exposure_by_bin)
            | set(self._episode_violation_by_bin)
            | set(self._safety_truncation_by_bin)
        )
        if not all_bins:
            return

        bin_start = np.asarray(
            [bin_index * self.position_bin_size_m for bin_index in all_bins],
            dtype=np.float64,
        )
        bin_end = bin_start + self.position_bin_size_m
        sample_exposure = np.asarray(
            [self._sample_exposure_by_bin.get(bin_index, 0) for bin_index in all_bins],
            dtype=np.float64,
        )
        sample_violation = np.asarray(
            [self._sample_violation_by_bin.get(bin_index, 0) for bin_index in all_bins],
            dtype=np.float64,
        )
        episode_exposure = np.asarray(
            [self._episode_exposure_by_bin.get(bin_index, 0) for bin_index in all_bins],
            dtype=np.float64,
        )
        episode_violation = np.asarray(
            [
                self._episode_violation_by_bin.get(bin_index, 0)
                for bin_index in all_bins
            ],
            dtype=np.float64,
        )
        safety_truncation = np.asarray(
            [
                self._safety_truncation_by_bin.get(bin_index, 0)
                for bin_index in all_bins
            ],
            dtype=np.float64,
        )

        np.savez(
            self.output_path,
            bin_start_m=bin_start,
            bin_end_m=bin_end,
            sample_exposure_count=sample_exposure,
            sample_violation_count=sample_violation,
            sample_violation_rate=np.divide(
                sample_violation,
                sample_exposure,
                out=np.zeros_like(sample_violation, dtype=np.float64),
                where=sample_exposure > 0.0,
            ),
            episode_exposure_count=episode_exposure,
            episode_violation_count=episode_violation,
            episode_violation_rate=np.divide(
                episode_violation,
                episode_exposure,
                out=np.zeros_like(episode_violation, dtype=np.float64),
                where=episode_exposure > 0.0,
            ),
            safety_truncation_count=safety_truncation,
            position_bin_size_m=np.asarray(
                [self.position_bin_size_m], dtype=np.float64
            ),
        )


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: Any,
        output_dir: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
        recorders: list[Any] | tuple[Any, ...] | None = None,
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
        self.eval_trigger_mode = eval_trigger_mode
        self.trigger_interval = int(eval_trigger_interval)
        self.deterministic = deterministic
        if recorders is None:
            if output_dir is None:
                raise ValueError(
                    "output_dir is required when recorders are not provided"
                )
            self.recorders = [
                BestTrajectoryRecorder(
                    output_dir=output_dir,
                    artifact_metadata=artifact_metadata,
                )
            ]
        else:
            self.recorders = list(recorders)
        self._episodes_seen = 0
        self._next_trigger_value = self.trigger_interval

    @property
    def best_recorder(self) -> BestTrajectoryRecorder | None:
        for recorder in self.recorders:
            if isinstance(recorder, BestTrajectoryRecorder):
                return recorder
        return None

    @property
    def best_result(self) -> PolicyEvaluationResult | None:
        recorder = self.best_recorder
        return None if recorder is None else recorder.best_result

    @property
    def best_trigger_interval(self) -> int | None:
        recorder = self.best_recorder
        return None if recorder is None else recorder.best_trigger_interval

    def _init_callback(self) -> None:
        for recorder in self.recorders:
            init_fn = getattr(recorder, "init_callback", None)
            if callable(init_fn):
                init_fn(self)

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

    def _run_evaluation(self, *, eval_trigger_interval: int) -> None:
        result = evaluate_policy_once(
            self.model,
            self.eval_env,
            deterministic=self.deterministic,
        )

        for recorder in self.recorders:
            on_evaluation = getattr(recorder, "on_evaluation", None)
            if callable(on_evaluation):
                on_evaluation(
                    self,
                    result,
                    eval_trigger_interval=eval_trigger_interval,
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
        for recorder in self.recorders:
            on_training_end = getattr(recorder, "on_training_end", None)
            if callable(on_training_end):
                on_training_end(self)

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
