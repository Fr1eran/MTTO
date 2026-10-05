"""PPO construction, learning rate scheduling, and in-memory training callbacks."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, override

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mtto.evaluation.quality import QualityReport, assess, best_update_reason
from mtto.rl.diagnostics import (
    REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    REWARD_NAMES,
    REWARD_SIGNAL_COUNT,
    BestPolicy,
    EvaluationHistory,
    RewardDiagnostics,
    SafetyTruncationHistogram,
)
from mtto.rl.evaluate import RLRun, calculate_route_completion_ratio, run_policy
from mtto.rl.state import TerminationReason

DEFAULT_BATCH_SIZE: Final[int] = 512
DEFAULT_N_EPOCHS: Final[int] = 8
# Initial policy std exp(-1) = 0.37 m/s^2: the unit-std start spends the first
# rollouts in a high-energy plateau and converges two to three times later.
DEFAULT_LOG_STD_INIT: Final[float] = -1.0
DEFAULT_DEVICE: Final[str] = "cpu"
DEFAULT_EVALUATION_INTERVAL_ROLLOUTS: Final[int] = 12
# The networks are small: one torch thread trains about 2.4x faster than the
# default eight, whose synchronisation outweighs the parallel work.
TORCH_NUM_THREADS: Final[int] = 1

LEARNING_RATE_SCHEDULE_ID: Final[str] = "cosine_completed_episodes_v1"
STEP_LEARNING_RATE_SCHEDULE_ID: Final[str] = "cosine_environment_steps_v1"
INITIAL_LEARNING_RATE: Final[float] = 3e-4
FINAL_LEARNING_RATE: Final[float] = 1e-5


@dataclass(slots=True)
class CompletedEpisodeProgress:
    """Mutable global episode progress shared by stopping and schedules."""

    target_episodes: int
    completed_episodes: int = 0

    def __post_init__(self) -> None:
        if self.target_episodes < 1:
            raise ValueError("target_episodes must be >= 1")

    @property
    def fraction(self) -> float:
        return min(1.0, max(0.0, self.completed_episodes / self.target_episodes))


class StopTrainingOnCompletedEpisodes(BaseCallback):
    """Stop at a global completed-episode target and update shared progress."""

    def __init__(self, progress: CompletedEpisodeProgress, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.progress = progress

    @property
    def n_episodes(self) -> int:
        """Compatibility view used by diagnostics and test doubles."""
        return self.progress.completed_episodes

    @n_episodes.setter
    def n_episodes(self, value: int) -> None:
        self.progress.completed_episodes = int(value)

    def _on_step(self) -> bool:
        assert "dones" in self.locals, "dones is required for episode accounting"
        self.progress.completed_episodes += int(np.sum(self.locals["dones"]))
        return self.progress.completed_episodes < self.progress.target_episodes


def completed_episode_cosine_annealing_schedule(
    progress: CompletedEpisodeProgress,
    initial_value: float = INITIAL_LEARNING_RATE,
    final_value: float = FINAL_LEARNING_RATE,
) -> Callable[[float], float]:
    def func(_sb3_progress_remaining: float) -> float:
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress.fraction))
        return final_value + (initial_value - final_value) * cosine_decay

    return func


def environment_step_cosine_annealing_schedule(
    initial_value: float = INITIAL_LEARNING_RATE,
    final_value: float = FINAL_LEARNING_RATE,
) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        progress = min(1.0, max(0.0, 1.0 - float(progress_remaining)))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_value + (initial_value - final_value) * cosine_decay

    return func


def learning_rate_schedule_parameters(
    budget_mode: Literal["completed_episodes", "environment_steps"] = (
        "completed_episodes"
    ),
) -> dict[str, float | str]:
    return {
        "id": (
            STEP_LEARNING_RATE_SCHEDULE_ID
            if budget_mode == "environment_steps"
            else LEARNING_RATE_SCHEDULE_ID
        ),
        "progress_unit": (
            "global_environment_transitions"
            if budget_mode == "environment_steps"
            else "global_completed_training_episodes"
        ),
        "initial_value": INITIAL_LEARNING_RATE,
        "final_value": FINAL_LEARNING_RATE,
    }


def build_ppo(
    venv: Any,
    *,
    device: str = DEFAULT_DEVICE,
    n_steps: int,
    gamma: float,
    learning_rate: float | Callable[[float], float],
    tensorboard_log: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PPO:
    return PPO(
        "MlpPolicy",
        venv,
        device=device,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=DEFAULT_N_EPOCHS,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            log_std_init=DEFAULT_LOG_STD_INIT,
        ),
    )


class RewardDiagnosticsCallback(BaseCallback):
    """Drain worker reward moments per rollout and construct in-memory
    RewardDiagnostics.
    """

    def __init__(self, *, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
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
            "termination_reason": [],
            "complete": [],
            "reward_sums": [],
        }
        self._diagnostics: RewardDiagnostics | None = None

    @property
    def completed_episode_count(self) -> int:
        """Return the number of completed episodes drained from all workers."""
        return self._completed_episode_count

    @property
    def diagnostics(self) -> RewardDiagnostics:
        if self._diagnostics is None:
            self._diagnostics = self._build_diagnostics(finalize=True)
        return self._diagnostics

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
                "termination_reason": self._array(
                    payload,
                    "episode_termination_reason",
                    dtype=np.int8,
                    shape=(episode_count,),
                ),
                "complete": self._array(
                    payload,
                    "episode_complete",
                    dtype=np.bool_,
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

    def _build_diagnostics(self, *, finalize: bool) -> RewardDiagnostics:
        self._drain_worker_batches(finalize=finalize)
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
            "episode_termination_reason": self._concat_or_empty(
                self._episode_chunks["termination_reason"],
                shape=(0,),
                dtype=np.dtype(np.int8),
            ),
            "episode_complete": self._concat_or_empty(
                self._episode_chunks["complete"],
                shape=(0,),
                dtype=np.dtype(np.bool_),
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
            "episode_termination_reason",
            "episode_complete",
            "episode_reward_sums",
        ):
            fields[key] = fields[key][episode_order]
        return RewardDiagnostics(**fields)

    @override
    def _on_training_end(self) -> None:
        self._diagnostics = self._build_diagnostics(finalize=True)


class SafetyTruncationHistogramCallback(BaseCallback):
    """Collect worker-buffered safety truncations in memory and construct
    SafetyTruncationHistogram.
    """

    def __init__(
        self,
        *,
        position_bin_size_m: float = 5000.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.position_bin_size_m = float(position_bin_size_m)
        if not np.isfinite(self.position_bin_size_m) or self.position_bin_size_m <= 0:
            raise ValueError("position_bin_size_m must be finite and positive")
        self._position_chunks: list[NDArray[np.float32]] = []
        self._termination_reason_chunks: list[NDArray[np.int8]] = []
        self._histogram: SafetyTruncationHistogram | None = None

    @property
    def histogram(self) -> SafetyTruncationHistogram:
        if self._histogram is None:
            self._histogram = self._build_histogram()
        return self._histogram

    @override
    def _on_step(self) -> bool:
        return True

    def _drain_worker_batches(self) -> None:
        payloads = self.training_env.env_method("drain_safety_truncations")
        valid_codes = np.asarray(
            [
                int(TerminationReason.UNDER_LOWER_LIMIT),
                int(TerminationReason.OVER_UPPER_LIMIT),
                int(TerminationReason.OVER_SRTSP),
            ],
            dtype=np.int8,
        )
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("safety truncation payload must be a dictionary")
            if "position_m" not in payload or "termination_reason" not in payload:
                raise ValueError("safety truncation payload is missing required fields")
            positions = np.asarray(payload["position_m"], dtype=np.float32).reshape(-1)
            codes = np.asarray(payload["termination_reason"], dtype=np.int8).reshape(-1)
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
                self._termination_reason_chunks.append(codes)

    @override
    def _on_rollout_end(self) -> None:
        self._drain_worker_batches()

    def _build_histogram(self) -> SafetyTruncationHistogram:
        self._drain_worker_batches()
        if self._position_chunks:
            positions = np.concatenate(self._position_chunks).astype(np.float64)
            codes = np.concatenate(self._termination_reason_chunks)
            absolute_bins, inverse = np.unique(
                np.floor(positions / self.position_bin_size_m).astype(np.int64),
                return_inverse=True,
            )
            bin_count = absolute_bins.size
            total = np.bincount(inverse, minlength=bin_count).astype(np.int64)
            low = np.bincount(
                inverse[codes == int(TerminationReason.UNDER_LOWER_LIMIT)],
                minlength=bin_count,
            ).astype(np.int64)
            high = np.bincount(
                inverse[
                    np.isin(
                        codes,
                        [
                            int(TerminationReason.OVER_UPPER_LIMIT),
                            int(TerminationReason.OVER_SRTSP),
                        ],
                    )
                ],
                minlength=bin_count,
            ).astype(np.int64)
            starts = absolute_bins.astype(np.float64) * self.position_bin_size_m
            ends = starts + self.position_bin_size_m
            shares = total.astype(np.float64) / float(total.sum())
        else:
            starts = ends = shares = np.empty(0, dtype=np.float64)
            total = low = high = np.empty(0, dtype=np.int64)
        return SafetyTruncationHistogram(
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

    @override
    def _on_training_end(self) -> None:
        self._histogram = self._build_histogram()


@dataclass(frozen=True, slots=True, eq=False)
class _EvaluationRecord:
    training_step: int
    rollout_index: int
    completed_episodes: int
    scheduled_completed_episodes: int
    run: RLRun
    quality: QualityReport


class ScheduledPolicyEvaluationCallback(BaseCallback):
    """Evaluate deterministic policy periodically in memory and track best policy."""

    def __init__(
        self,
        *,
        eval_env: Any,
        evaluation_interval_rollouts: int | None = None,
        evaluation_interval_episodes: int | None = None,
        deterministic: bool = True,
        get_completed_training_episodes: Callable[[], int] | None = None,
        evaluate_at_boundaries: bool = False,
        max_rollouts_exclusive: int | None = None,
        max_completed_episodes_exclusive: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if (
            evaluation_interval_rollouts is None
            and evaluation_interval_episodes is None
        ):
            evaluation_interval_rollouts = DEFAULT_EVALUATION_INTERVAL_ROLLOUTS
        elif (
            evaluation_interval_rollouts is not None
            and evaluation_interval_episodes is not None
        ):
            raise ValueError(
                "exactly one of evaluation_interval_rollouts and "
                "evaluation_interval_episodes must be configured"
            )
        if (
            evaluation_interval_rollouts is not None
            and evaluation_interval_rollouts <= 0
        ):
            raise ValueError("evaluation_interval_rollouts must be positive")
        if (
            evaluation_interval_episodes is not None
            and evaluation_interval_episodes <= 0
        ):
            raise ValueError("evaluation_interval_episodes must be positive")
        if (
            evaluation_interval_episodes is not None
            and get_completed_training_episodes is None
        ):
            raise ValueError(
                "episode evaluation scheduling requires completed-episode progress"
            )
        self.eval_env = eval_env
        self.evaluation_interval_rollouts = (
            None
            if evaluation_interval_rollouts is None
            else int(evaluation_interval_rollouts)
        )
        self.evaluation_interval_episodes = (
            None
            if evaluation_interval_episodes is None
            else int(evaluation_interval_episodes)
        )
        self.deterministic = bool(deterministic)
        self.get_completed_training_episodes = get_completed_training_episodes
        self.evaluate_at_boundaries = evaluate_at_boundaries
        self.max_rollouts_exclusive = max_rollouts_exclusive
        self.max_completed_episodes_exclusive = max_completed_episodes_exclusive
        self._rollouts_completed = 0
        self._last_scheduled_completed_episodes = 0
        self._records: list[_EvaluationRecord] = []
        self._best: BestPolicy | None = None
        self._history: EvaluationHistory | None = None

    @property
    def best(self) -> BestPolicy | None:
        return self._best

    @property
    def history(self) -> EvaluationHistory:
        if self._history is None:
            self._history = self._build_history()
        return self._history

    @override
    def _on_step(self) -> bool:
        return True

    @override
    def _on_training_start(self) -> None:
        if self.evaluate_at_boundaries:
            self._emit_evaluation()

    @override
    def _on_rollout_start(self) -> None:
        interval = self.evaluation_interval_episodes
        if interval is None:
            return
        assert self.get_completed_training_episodes is not None
        completed = int(self.get_completed_training_episodes())
        scheduled = (completed // interval) * interval
        if scheduled <= self._last_scheduled_completed_episodes:
            return
        if (
            self.max_completed_episodes_exclusive is not None
            and scheduled >= self.max_completed_episodes_exclusive
        ):
            return
        self._last_scheduled_completed_episodes = scheduled
        self._emit_evaluation(scheduled_completed_episodes=scheduled)

    @override
    def _on_rollout_end(self) -> None:
        self._rollouts_completed += 1
        if (
            self.max_rollouts_exclusive is not None
            and self._rollouts_completed >= self.max_rollouts_exclusive
        ):
            return
        if self.evaluation_interval_rollouts is None:
            return
        if self._rollouts_completed % self.evaluation_interval_rollouts != 0:
            return
        self._emit_evaluation()

    def _log_result(
        self,
        *,
        run: RLRun,
        quality: QualityReport,
        prefix: str,
    ) -> None:
        time_error_s = float(quality.metrics.arrival_time_error_s)
        values = {
            "success": float(quality.completed),
            "precise_arrival": float(quality.precise_stop),
            "punctual_arrival": float(quality.punctual),
            "total_reward": float(run.total_reward),
            "stop_error_m": float(quality.metrics.stop_error_m),
            "time_error_s": time_error_s,
            "abs_time_error_s": abs(time_error_s),
            "total_energy_j": float(quality.metrics.total_energy_kj * 1000.0),
            "comfort_tav": float(quality.metrics.comfort_tav_mps2),
            "comfort_er_pct": float(quality.metrics.comfort_exceedance_pct),
            "comfort_rms": float(quality.metrics.comfort_rms_mps2),
        }
        for name, value in values.items():
            self.logger.record(f"best_eval/{prefix}_{name}", value)

    def _emit_evaluation(self, *, scheduled_completed_episodes: int = 0) -> None:
        run = run_policy(self.model, self.eval_env, deterministic=self.deterministic)
        quality = assess(run.profile, self.eval_env.scenario, self.eval_env.task)
        training_step = int(self.num_timesteps)
        rollout_index = self._rollouts_completed
        completed_episodes = (
            int(self.get_completed_training_episodes())
            if self.get_completed_training_episodes is not None
            else 0
        )
        self._records.append(
            _EvaluationRecord(
                training_step=training_step,
                rollout_index=rollout_index,
                completed_episodes=completed_episodes,
                scheduled_completed_episodes=scheduled_completed_episodes,
                run=run,
                quality=quality,
            )
        )
        self._log_result(run=run, quality=quality, prefix="last")
        reason = best_update_reason(
            quality,
            self._best.quality if self._best is not None else None,
        )
        if reason is not None:
            parameters = copy.deepcopy(self.model.get_parameters())
            self._best = BestPolicy(
                parameters=parameters,
                run=run,
                quality=quality,
                update_reason=reason,
                training_step=training_step,
                rollout_index=rollout_index,
            )
            self._log_result(run=run, quality=quality, prefix="best")

    def _build_history(self) -> EvaluationHistory:
        if self._records:
            training_steps = np.asarray(
                [r.training_step for r in self._records], dtype=np.int64
            )
            rollout_indices = np.asarray(
                [r.rollout_index for r in self._records], dtype=np.int64
            )
            completed_episodes = np.asarray(
                [r.completed_episodes for r in self._records], dtype=np.int64
            )
            scheduled_completed_episodes = np.asarray(
                [r.scheduled_completed_episodes for r in self._records], dtype=np.int64
            )
            total_reward = np.asarray(
                [r.run.total_reward for r in self._records], dtype=np.float64
            )
            episode_steps = np.asarray(
                [r.run.steps for r in self._records], dtype=np.int64
            )
            success = np.asarray(
                [r.quality.completed for r in self._records], dtype=np.bool_
            )
            safe = np.asarray([r.quality.safe for r in self._records], dtype=np.bool_)
            feasible = np.asarray(
                [r.quality.feasible for r in self._records], dtype=np.bool_
            )
            stop_error_m = np.asarray(
                [r.quality.metrics.stop_error_m for r in self._records],
                dtype=np.float64,
            )
            time_error_s = np.asarray(
                [r.quality.metrics.arrival_time_error_s for r in self._records],
                dtype=np.float64,
            )
            total_energy_j = np.asarray(
                [r.quality.metrics.total_energy_kj * 1000.0 for r in self._records],
                dtype=np.float64,
            )
            comfort_tav = np.asarray(
                [r.quality.metrics.comfort_tav_mps2 for r in self._records],
                dtype=np.float64,
            )
            task = self.eval_env.task
            route_completion_ratio = np.asarray(
                [
                    calculate_route_completion_ratio(
                        start_position_m=task.start_position_m,
                        target_position_m=task.target_position_m,
                        final_position_m=float(r.run.profile.position_m[-1]),
                    )
                    for r in self._records
                ],
                dtype=np.float64,
            )
            positions = [
                np.asarray(
                    [v.position_m for v in r.quality.audit.violations],
                    dtype=np.float64,
                )
                for r in self._records
            ]
            offsets = np.zeros(len(positions) + 1, dtype=np.int64)
            for index, values in enumerate(positions, start=1):
                offsets[index] = offsets[index - 1] + values.size
            flattened = (
                np.concatenate(positions)
                if offsets[-1] > 0
                else np.empty(0, dtype=np.float64)
            )
        else:
            training_steps = np.empty(0, dtype=np.int64)
            rollout_indices = np.empty(0, dtype=np.int64)
            total_reward = np.empty(0, dtype=np.float64)
            episode_steps = np.empty(0, dtype=np.int64)
            success = np.empty(0, dtype=np.bool_)
            safe = np.empty(0, dtype=np.bool_)
            feasible = np.empty(0, dtype=np.bool_)
            stop_error_m = np.empty(0, dtype=np.float64)
            time_error_s = np.empty(0, dtype=np.float64)
            total_energy_j = np.empty(0, dtype=np.float64)
            comfort_tav = np.empty(0, dtype=np.float64)
            completed_episodes = np.empty(0, dtype=np.int64)
            scheduled_completed_episodes = np.empty(0, dtype=np.int64)
            route_completion_ratio = np.empty(0, dtype=np.float64)
            flattened = np.empty(0, dtype=np.float64)
            offsets = np.asarray([0], dtype=np.int64)

        return EvaluationHistory(
            training_steps=training_steps,
            rollout_indices=rollout_indices,
            total_reward=total_reward,
            episode_steps=episode_steps,
            success=success,
            safe=safe,
            feasible=feasible,
            stop_error_m=stop_error_m,
            time_error_s=time_error_s,
            total_energy_j=total_energy_j,
            comfort_tav=comfort_tav,
            completed_training_episodes=completed_episodes,
            scheduled_completed_training_episodes=scheduled_completed_episodes,
            route_completion_ratio=route_completion_ratio,
            safety_violation_positions_m=flattened,
            safety_violation_position_offsets=offsets,
        )

    @override
    def _on_training_end(self) -> None:
        if self.evaluate_at_boundaries:
            self._emit_evaluation()
        self._history = self._build_history()
        close = getattr(self.eval_env, "close", None)
        if callable(close):
            close()
