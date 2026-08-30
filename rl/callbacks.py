from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, override

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback

from contracts.evaluation import EvaluationHistory
from rl.evaluation import (
    PolicyEvaluationResult,
    describe_best_update_reason,
    evaluate_policy_once,
    save_policy_evaluation_curve,
)
from rl.operational_state import ViolationCode
from rl.reward_diagnostics import (
    REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    REWARD_NAMES,
    REWARD_SIGNAL_COUNT,
)
from utils.io_utils import save_evaluation_history

DEFAULT_EVALUATION_INTERVAL_ROLLOUTS = 12


@dataclass(frozen=True, slots=True)
class PolicyEvaluationEvent:
    result: PolicyEvaluationResult
    training_step: int
    rollout_index: int
    completed_training_episodes: int = 0


class EvaluationResultHandler(Protocol):
    def initialize(self, host: ScheduledPolicyEvaluationCallback) -> None: ...

    def handle_evaluation(
        self,
        host: ScheduledPolicyEvaluationCallback,
        event: PolicyEvaluationEvent,
    ) -> None: ...

    def finalize(self, host: ScheduledPolicyEvaluationCallback) -> None: ...


class RewardDiagnosticsArtifactCallback(BaseCallback):
    """Drain worker reward moments per rollout and persist one final artifact."""

    def __init__(self, *, output_path: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.output_path = str(output_path)
        self._rollout_end_steps: list[int] = []
        self._rollout_counts: list[int] = []
        self._rollout_sums: list[NDArray[np.float64]] = []
        self._rollout_abs_sums: list[NDArray[np.float64]] = []
        self._rollout_nonzero_counts: list[NDArray[np.int64]] = []
        self._rollout_cross_products: list[NDArray[np.float64]] = []
        self._completed_episode_count = 0
        self._episode_chunks: dict[str, list[np.ndarray]] = {
            "end_step": [],
            "worker_rank": [],
            "index": [],
            "length": [],
            "terminated": [],
            "truncated": [],
            "complete": [],
            "violation_code": [],
            "reward_sums": [],
        }

    @property
    def completed_episode_count(self) -> int:
        """Return the number of completed episodes drained from all workers."""
        return self._completed_episode_count

    @override
    def _init_callback(self) -> None:
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @override
    def _on_step(self) -> bool:
        return True

    @staticmethod
    def _array(
        payload: dict[str, object],
        key: str,
        *,
        dtype: np.dtype[Any] | type[Any],
        shape: tuple[int | None, ...],
    ) -> np.ndarray:
        if key not in payload:
            raise ValueError(f"reward diagnostics payload is missing '{key}'")
        value = np.asarray(payload[key], dtype=dtype)
        if value.ndim != len(shape) or any(
            expected is not None and value.shape[index] != expected
            for index, expected in enumerate(shape)
        ):
            raise ValueError(f"reward diagnostics payload has invalid '{key}' shape")
        return value

    def _drain_worker_batches(self, *, finalize: bool) -> None:
        payloads = self.training_env.env_method(
            "drain_reward_diagnostics", finalize=finalize
        )
        rollout_count = 0
        reward_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
        reward_abs_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
        reward_nonzero_count = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.int64)
        reward_cross_product = np.zeros(
            (REWARD_SIGNAL_COUNT, REWARD_SIGNAL_COUNT), dtype=np.float64
        )
        num_envs = int(self.training_env.num_envs)

        for payload_raw in payloads:
            if not isinstance(payload_raw, dict):
                raise TypeError("reward diagnostics payload must be a dictionary")
            payload = payload_raw
            count = self._array(payload, "transition_count", dtype=np.int64, shape=(1,))
            worker_count = int(count[0])
            if worker_count < 0:
                raise ValueError(
                    "reward diagnostics transition count must be nonnegative"
                )
            rollout_count += worker_count
            reward_sum += self._array(
                payload,
                "reward_sum",
                dtype=np.float64,
                shape=(REWARD_SIGNAL_COUNT,),
            )
            reward_abs_sum += self._array(
                payload,
                "reward_abs_sum",
                dtype=np.float64,
                shape=(REWARD_SIGNAL_COUNT,),
            )
            reward_nonzero_count += self._array(
                payload,
                "reward_nonzero_count",
                dtype=np.int64,
                shape=(REWARD_SIGNAL_COUNT,),
            )
            reward_cross_product += self._array(
                payload,
                "reward_cross_product",
                dtype=np.float64,
                shape=(REWARD_SIGNAL_COUNT, REWARD_SIGNAL_COUNT),
            )

            end_worker_step = self._array(
                payload, "episode_end_worker_step", dtype=np.int64, shape=(None,)
            )
            episode_count = end_worker_step.size
            episode_arrays = {
                "end_step": end_worker_step * num_envs,
                "worker_rank": self._array(
                    payload,
                    "episode_worker_rank",
                    dtype=np.int16,
                    shape=(episode_count,),
                ),
                "index": self._array(
                    payload,
                    "episode_index",
                    dtype=np.int64,
                    shape=(episode_count,),
                ),
                "length": self._array(
                    payload,
                    "episode_length",
                    dtype=np.int32,
                    shape=(episode_count,),
                ),
                "terminated": self._array(
                    payload,
                    "episode_terminated",
                    dtype=np.bool_,
                    shape=(episode_count,),
                ),
                "truncated": self._array(
                    payload,
                    "episode_truncated",
                    dtype=np.bool_,
                    shape=(episode_count,),
                ),
                "complete": self._array(
                    payload,
                    "episode_complete",
                    dtype=np.bool_,
                    shape=(episode_count,),
                ),
                "violation_code": self._array(
                    payload,
                    "episode_violation_code",
                    dtype=np.int8,
                    shape=(episode_count,),
                ),
                "reward_sums": self._array(
                    payload,
                    "episode_reward_sums",
                    dtype=np.float64,
                    shape=(episode_count, REWARD_SIGNAL_COUNT),
                ),
            }
            if episode_count:
                self._completed_episode_count += int(
                    np.count_nonzero(episode_arrays["complete"])
                )
                for key, value in episode_arrays.items():
                    self._episode_chunks[key].append(value)

        if rollout_count:
            self._rollout_end_steps.append(int(self.num_timesteps))
            self._rollout_counts.append(rollout_count)
            self._rollout_sums.append(reward_sum)
            self._rollout_abs_sums.append(reward_abs_sum)
            self._rollout_nonzero_counts.append(reward_nonzero_count)
            self._rollout_cross_products.append(reward_cross_product)

    @override
    def _on_rollout_end(self) -> None:
        self._drain_worker_batches(finalize=False)

    @staticmethod
    def _stack_or_empty(
        values: list[np.ndarray], *, shape: tuple[int, ...], dtype: np.dtype[Any]
    ) -> np.ndarray:
        if values:
            return np.stack(values).astype(dtype, copy=False)
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def _concat_or_empty(
        values: list[np.ndarray], *, shape: tuple[int, ...], dtype: np.dtype[Any]
    ) -> np.ndarray:
        if values:
            return np.concatenate(values, axis=0).astype(dtype, copy=False)
        return np.empty(shape, dtype=dtype)

    @override
    def _on_training_end(self) -> None:
        self._drain_worker_batches(finalize=True)
        rollout_count = len(self._rollout_counts)
        fields = {
            "schema_version": np.asarray(
                [REWARD_DIAGNOSTICS_SCHEMA_VERSION], dtype=np.int16
            ),
            "reward_names": np.asarray(REWARD_NAMES),
            "rollout_end_step": np.asarray(self._rollout_end_steps, dtype=np.int64),
            "rollout_transition_count": np.asarray(
                self._rollout_counts, dtype=np.int64
            ),
            "rollout_reward_sum": self._stack_or_empty(
                self._rollout_sums,
                shape=(0, REWARD_SIGNAL_COUNT),
                dtype=np.dtype(np.float64),
            ),
            "rollout_reward_abs_sum": self._stack_or_empty(
                self._rollout_abs_sums,
                shape=(0, REWARD_SIGNAL_COUNT),
                dtype=np.dtype(np.float64),
            ),
            "rollout_reward_nonzero_count": self._stack_or_empty(
                self._rollout_nonzero_counts,
                shape=(0, REWARD_SIGNAL_COUNT),
                dtype=np.dtype(np.int64),
            ),
            "rollout_reward_cross_product": self._stack_or_empty(
                self._rollout_cross_products,
                shape=(0, REWARD_SIGNAL_COUNT, REWARD_SIGNAL_COUNT),
                dtype=np.dtype(np.float64),
            ),
            "episode_end_step": self._concat_or_empty(
                self._episode_chunks["end_step"],
                shape=(0,),
                dtype=np.dtype(np.int64),
            ),
            "episode_worker_rank": self._concat_or_empty(
                self._episode_chunks["worker_rank"],
                shape=(0,),
                dtype=np.dtype(np.int16),
            ),
            "episode_index": self._concat_or_empty(
                self._episode_chunks["index"],
                shape=(0,),
                dtype=np.dtype(np.int64),
            ),
            "episode_length": self._concat_or_empty(
                self._episode_chunks["length"],
                shape=(0,),
                dtype=np.dtype(np.int32),
            ),
            "episode_terminated": self._concat_or_empty(
                self._episode_chunks["terminated"],
                shape=(0,),
                dtype=np.dtype(np.bool_),
            ),
            "episode_truncated": self._concat_or_empty(
                self._episode_chunks["truncated"],
                shape=(0,),
                dtype=np.dtype(np.bool_),
            ),
            "episode_complete": self._concat_or_empty(
                self._episode_chunks["complete"],
                shape=(0,),
                dtype=np.dtype(np.bool_),
            ),
            "episode_violation_code": self._concat_or_empty(
                self._episode_chunks["violation_code"],
                shape=(0,),
                dtype=np.dtype(np.int8),
            ),
            "episode_reward_sums": self._concat_or_empty(
                self._episode_chunks["reward_sums"],
                shape=(0, REWARD_SIGNAL_COUNT),
                dtype=np.dtype(np.float64),
            ),
        }
        if fields["rollout_end_step"].shape != (rollout_count,):
            raise ValueError("reward diagnostics rollout arrays are inconsistent")
        episode_order = np.lexsort(
            (
                fields["episode_index"],
                fields["episode_worker_rank"],
                fields["episode_end_step"],
            )
        )
        for key in (
            "episode_end_step",
            "episode_worker_rank",
            "episode_index",
            "episode_length",
            "episode_terminated",
            "episode_truncated",
            "episode_complete",
            "episode_violation_code",
            "episode_reward_sums",
        ):
            fields[key] = fields[key][episode_order]
        temporary_path = f"{self.output_path}.tmp.npz"
        np.savez(temporary_path, **fields)
        os.replace(temporary_path, self.output_path)


class SafetyTruncationPositionHistogramCallback(BaseCallback):
    """Collect worker-buffered safety truncations and persist position bins."""

    def __init__(
        self,
        *,
        output_path: str,
        position_bin_size_m: float = 5000.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_path = str(output_path)
        self.position_bin_size_m = float(position_bin_size_m)
        if not np.isfinite(self.position_bin_size_m) or self.position_bin_size_m <= 0:
            raise ValueError("position_bin_size_m must be finite and positive")
        self._position_chunks: list[NDArray[np.float32]] = []
        self._violation_code_chunks: list[NDArray[np.int8]] = []

    @override
    def _init_callback(self) -> None:
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @override
    def _on_step(self) -> bool:
        return True

    def _drain_worker_batches(self) -> None:
        payloads = self.training_env.env_method("drain_safety_truncations")
        valid_codes = np.asarray(
            [int(ViolationCode.SPEED_LOW), int(ViolationCode.SPEED_HIGH)],
            dtype=np.int8,
        )
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("safety truncation payload must be a dictionary")
            if "position_m" not in payload or "violation_code" not in payload:
                raise ValueError("safety truncation payload is missing required fields")
            positions = np.asarray(payload["position_m"], dtype=np.float32).reshape(-1)
            codes = np.asarray(payload["violation_code"], dtype=np.int8).reshape(-1)
            if positions.shape != codes.shape:
                raise ValueError(
                    "safety truncation payload arrays must have equal shape"
                )
            if not np.all(np.isfinite(positions)):
                raise ValueError("safety truncation positions must be finite")
            if codes.size and not np.all(np.isin(codes, valid_codes)):
                raise ValueError("safety truncation payload contains an invalid code")
            if positions.size:
                self._position_chunks.append(positions)
                self._violation_code_chunks.append(codes)

    @override
    def _on_rollout_end(self) -> None:
        self._drain_worker_batches()

    @override
    def _on_training_end(self) -> None:
        self._drain_worker_batches()
        if self._position_chunks:
            positions = np.concatenate(self._position_chunks).astype(np.float64)
            codes = np.concatenate(self._violation_code_chunks)
            absolute_bins, inverse = np.unique(
                np.floor(positions / self.position_bin_size_m).astype(np.int64),
                return_inverse=True,
            )
            bin_count = absolute_bins.size
            total = np.bincount(inverse, minlength=bin_count).astype(np.int64)
            low = np.bincount(
                inverse[codes == int(ViolationCode.SPEED_LOW)], minlength=bin_count
            ).astype(np.int64)
            high = np.bincount(
                inverse[codes == int(ViolationCode.SPEED_HIGH)], minlength=bin_count
            ).astype(np.int64)
            starts = absolute_bins.astype(np.float64) * self.position_bin_size_m
            ends = starts + self.position_bin_size_m
            shares = total.astype(np.float64) / float(total.sum())
        else:
            starts = ends = shares = np.empty(0, dtype=np.float64)
            total = low = high = np.empty(0, dtype=np.int64)
        np.savez(
            self.output_path,
            bin_start_m=starts,
            bin_end_m=ends,
            safety_truncation_count=total,
            low_safety_truncation_count=low,
            high_safety_truncation_count=high,
            global_safety_truncation_share=shares,
            position_bin_size_m=np.asarray(
                [self.position_bin_size_m], dtype=np.float64
            ),
        )


class BestEvaluationArtifactHandler:
    def __init__(
        self,
        *,
        output_dir: str,
        artifact_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.output_dir = str(output_dir)
        self.artifact_metadata = dict(artifact_metadata or {})
        self.best_result: PolicyEvaluationResult | None = None

    def initialize(self, host: ScheduledPolicyEvaluationCallback) -> None:
        del host
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _log_result(
        host: ScheduledPolicyEvaluationCallback,
        result: PolicyEvaluationResult,
        *,
        prefix: str,
    ) -> None:
        values = {
            "success": float(result.success),
            "precise_arrival": float(result.precise_arrival),
            "punctual_arrival": float(result.punctual_arrival),
            "total_reward": result.total_reward,
            "stop_error_m": result.stop_error_m,
            "time_error_s": result.time_error_s,
            "abs_time_error_s": abs(result.time_error_s),
            "total_energy_j": result.total_energy_j,
            "comfort_tav": result.comfort_tav,
            "comfort_er_pct": result.comfort_er_pct,
            "comfort_rms": result.comfort_rms,
        }
        for name, value in values.items():
            host.logger.record(f"best_eval/{prefix}_{name}", value)

    def handle_evaluation(
        self,
        host: ScheduledPolicyEvaluationCallback,
        event: PolicyEvaluationEvent,
    ) -> None:
        self._log_result(host, event.result, prefix="last")
        reason = describe_best_update_reason(event.result, self.best_result)
        if reason is None:
            return
        tracked = event.result
        host.model.save(os.path.join(self.output_dir, "policy_best"))
        metrics = tracked.to_metrics(
            num_timesteps=event.training_step,
            evaluation_rollout_index=event.rollout_index,
        )
        metrics.update(self.artifact_metadata)
        metrics["trajectory_source"] = "best"
        metrics["best_update_reason"] = reason
        save_policy_evaluation_curve(
            tracked,
            os.path.join(self.output_dir, "best_trajectory.npz"),
            extra_metrics=metrics,
            metrics_path=os.path.join(self.output_dir, "metrics_best.json"),
        )
        self.best_result = tracked
        self._log_result(host, tracked, prefix="best")

    def finalize(self, host: ScheduledPolicyEvaluationCallback) -> None:
        del host


class EvaluationHistoryArtifactHandler:
    """Persist fixed-start metrics from every scheduled policy evaluation."""

    def __init__(self, *, output_path: str) -> None:
        self.output_path = str(output_path)
        self._events: list[PolicyEvaluationEvent] = []

    def initialize(self, host: ScheduledPolicyEvaluationCallback) -> None:
        del host
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def handle_evaluation(
        self,
        host: ScheduledPolicyEvaluationCallback,
        event: PolicyEvaluationEvent,
    ) -> None:
        del host
        self._events.append(event)

    def finalize(self, host: ScheduledPolicyEvaluationCallback) -> None:
        del host
        positions = [
            np.asarray(event.result.safety_violation_positions_m, dtype=np.float64)
            for event in self._events
        ]
        offsets = np.zeros(len(positions) + 1, dtype=np.int64)
        for index, values in enumerate(positions, start=1):
            offsets[index] = offsets[index - 1] + values.size
        flattened = (
            np.concatenate(positions)
            if offsets[-1] > 0
            else np.empty(0, dtype=np.float64)
        )
        history = EvaluationHistory(
            training_steps=np.asarray(
                [event.training_step for event in self._events], dtype=np.int64
            ),
            rollout_indices=np.asarray(
                [event.rollout_index for event in self._events], dtype=np.int64
            ),
            total_reward=np.asarray(
                [event.result.total_reward for event in self._events], dtype=np.float64
            ),
            episode_steps=np.asarray(
                [event.result.episode_steps for event in self._events], dtype=np.int64
            ),
            success=np.asarray(
                [event.result.success for event in self._events], dtype=np.bool_
            ),
            stop_error_m=np.asarray(
                [event.result.stop_error_m for event in self._events], dtype=np.float64
            ),
            time_error_s=np.asarray(
                [event.result.time_error_s for event in self._events], dtype=np.float64
            ),
            total_energy_j=np.asarray(
                [event.result.total_energy_j for event in self._events],
                dtype=np.float64,
            ),
            comfort_tav=np.asarray(
                [event.result.comfort_tav for event in self._events], dtype=np.float64
            ),
            completed_training_episodes=np.asarray(
                [event.completed_training_episodes for event in self._events],
                dtype=np.int64,
            ),
            safety_violation_positions_m=flattened,
            safety_violation_position_offsets=offsets,
        )
        _ = save_evaluation_history(history, self.output_path)


class ScheduledPolicyEvaluationCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: Any,
        handlers: Sequence[EvaluationResultHandler],
        evaluation_interval_rollouts: int = DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        deterministic: bool = True,
        get_completed_training_episodes: Callable[[], int] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if evaluation_interval_rollouts <= 0:
            raise ValueError("evaluation_interval_rollouts must be positive")
        self.eval_env = eval_env
        self.handlers = list(handlers)
        self.evaluation_interval_rollouts = int(evaluation_interval_rollouts)
        self.deterministic = bool(deterministic)
        self.get_completed_training_episodes = get_completed_training_episodes
        self._rollouts_completed = 0

    @override
    def _init_callback(self) -> None:
        for handler in self.handlers:
            handler.initialize(self)

    @override
    def _on_step(self) -> bool:
        return True

    def run_evaluation(self) -> PolicyEvaluationResult:
        return evaluate_policy_once(
            self.model, self.eval_env, deterministic=self.deterministic
        )

    @override
    def _on_rollout_end(self) -> None:
        self._rollouts_completed += 1
        if self._rollouts_completed % self.evaluation_interval_rollouts != 0:
            return
        result = self.run_evaluation()
        event = PolicyEvaluationEvent(
            result=result,
            training_step=int(self.num_timesteps),
            rollout_index=self._rollouts_completed,
            completed_training_episodes=(
                int(self.get_completed_training_episodes())
                if self.get_completed_training_episodes is not None
                else 0
            ),
        )
        for handler in self.handlers:
            handler.handle_evaluation(self, event)

    @override
    def _on_training_end(self) -> None:
        for handler in self.handlers:
            handler.finalize(self)
        close = getattr(self.eval_env, "close", None)
        if callable(close):
            close()


class EpisodeProgressBarCallback(ProgressBarCallback):
    """
    Display a progress bar based on training episodes instead of timesteps.
    """

    def __init__(self, total_episodes: int) -> None:
        super().__init__()
        self.total_episodes = total_episodes

    def _on_training_start(self) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            from tqdm import tqdm
        self.pbar = tqdm(total=self.total_episodes, desc="Training Episodes")

    def _on_step(self) -> bool:
        if "dones" in self.locals:
            new_episodes = int(np.sum(self.locals["dones"]))
            if new_episodes > 0:
                self.pbar.update(new_episodes)
        return True
