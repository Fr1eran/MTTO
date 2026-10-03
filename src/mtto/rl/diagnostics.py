"""In-memory training diagnostics, telemetry accumulators, and evaluation history."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypedDict

import numpy as np
from numpy.typing import NDArray

from mtto.rl.rewards import RewardBreakdown
from mtto.rl.state import TerminationReason

if TYPE_CHECKING:
    from mtto.evaluation.quality import QualityReport
    from mtto.rl.evaluate import RLRun

REWARD_DIAGNOSTICS_SCHEMA_VERSION: Final[int] = 8
_SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset(
    {REWARD_DIAGNOSTICS_SCHEMA_VERSION}
)
REWARD_NAMES: Final[tuple[str, ...]] = (
    "safety",
    "energy",
    "comfort",
    "terminal_stopping",
    "terminal_punctuality",
    "progress",
    "truncation",
    "punctuality_shaping",
    "total",
)
REWARD_SIGNAL_COUNT: Final[int] = len(REWARD_NAMES)
TOTAL_REWARD_INDEX: Final[int] = REWARD_NAMES.index("total")

EVALUATION_HISTORY_ARTIFACT_TYPE: Final[str] = "rl_evaluation_history"
EVALUATION_HISTORY_SCHEMA_VERSION: Final[int] = 4


class RewardDiagnosticsBatch(TypedDict):
    """Compact statistics drained from one worker at a rollout boundary."""

    transition_count: NDArray[np.int64]
    reward_sum: NDArray[np.float64]
    reward_abs_sum: NDArray[np.float64]
    reward_nonzero_count: NDArray[np.int64]
    reward_cross_product: NDArray[np.float64]
    episode_end_worker_step: NDArray[np.int64]
    episode_worker_rank: NDArray[np.int16]
    episode_index: NDArray[np.int64]
    episode_length: NDArray[np.int32]
    episode_termination_reason: NDArray[np.int8]
    episode_complete: NDArray[np.bool_]
    episode_reward_sums: NDArray[np.float64]


class RewardDiagnosticsAccumulator:
    """Accumulate transition moments and raw episode returns inside one worker."""

    def __init__(self, *, worker_rank: int, rollout_capacity: int) -> None:
        self.worker_rank = int(worker_rank)
        if not np.iinfo(np.int16).min <= self.worker_rank <= np.iinfo(np.int16).max:
            raise ValueError("worker_rank does not fit the artifact's int16 schema")
        capacity = int(rollout_capacity)
        if capacity <= 0:
            raise ValueError("rollout_capacity must be positive")
        self._transition_rewards = np.empty(
            (capacity, REWARD_SIGNAL_COUNT), dtype=np.float32
        )
        self._transition_count = 0
        self._worker_transition_step = 0
        self._episode_index = 0
        self._episode_length = 0
        self._episode_reward_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
        self._episodes: list[
            tuple[int, int, int, int, int, bool, NDArray[np.float64]]
        ] = []

    def _ensure_capacity(self) -> None:
        if self._transition_count < self._transition_rewards.shape[0]:
            return
        grown = np.empty(
            (self._transition_rewards.shape[0] * 2, REWARD_SIGNAL_COUNT),
            dtype=np.float32,
        )
        grown[: self._transition_count] = self._transition_rewards
        self._transition_rewards = grown

    def record(
        self,
        reward: RewardBreakdown,
        *,
        termination_reason: TerminationReason | None,
    ) -> None:
        self._ensure_capacity()
        vector = self._transition_rewards[self._transition_count]
        for index, name in enumerate(REWARD_NAMES):
            vector[index] = getattr(reward, name)
        self._transition_count += 1
        self._worker_transition_step += 1
        self._episode_length += 1
        self._episode_reward_sum += vector
        if termination_reason is not None:
            self._finish_episode(
                termination_reason=termination_reason,
            )

    def _finish_episode(
        self,
        *,
        termination_reason: TerminationReason | None,
    ) -> None:
        if self._episode_length <= 0:
            return
        code = int(termination_reason) if termination_reason is not None else 0
        complete = code != 0
        self._episodes.append(
            (
                self._worker_transition_step,
                self.worker_rank,
                self._episode_index,
                self._episode_length,
                code,
                complete,
                self._episode_reward_sum.copy(),
            )
        )
        self._episode_index += 1
        self._episode_length = 0
        self._episode_reward_sum.fill(0.0)

    def drain(self, *, finalize: bool = False) -> RewardDiagnosticsBatch:
        if finalize:
            self._finish_episode(
                termination_reason=None,
            )

        count = self._transition_count
        if count:
            matrix = self._transition_rewards[:count].astype(np.float64)
            reward_sum = matrix.sum(axis=0)
            reward_abs_sum = np.abs(matrix).sum(axis=0)
            reward_nonzero_count = np.count_nonzero(matrix, axis=0).astype(np.int64)
            reward_cross_product = matrix.T @ matrix
        else:
            reward_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
            reward_abs_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
            reward_nonzero_count = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.int64)
            reward_cross_product = np.zeros(
                (REWARD_SIGNAL_COUNT, REWARD_SIGNAL_COUNT), dtype=np.float64
            )
        self._transition_count = 0

        episode_count = len(self._episodes)
        if episode_count:
            end_steps = np.fromiter(
                (item[0] for item in self._episodes), dtype=np.int64
            )
            worker_ranks = np.fromiter(
                (item[1] for item in self._episodes), dtype=np.int16
            )
            episode_indices = np.fromiter(
                (item[2] for item in self._episodes), dtype=np.int64
            )
            lengths = np.fromiter((item[3] for item in self._episodes), dtype=np.int32)
        else:
            end_steps = np.empty(0, dtype=np.int64)
            worker_ranks = np.empty(0, dtype=np.int16)
            episode_indices = np.empty(0, dtype=np.int64)
            lengths = np.empty(0, dtype=np.int32)

        if episode_count:
            reason_values = np.asarray(
                [item[4] for item in self._episodes], dtype=np.int8
            )
            complete_values = np.asarray(
                [item[5] for item in self._episodes], dtype=np.bool_
            )
            episode_rewards = np.stack([item[6] for item in self._episodes]).astype(
                np.float64, copy=False
            )
        else:
            reason_values = np.empty(0, dtype=np.int8)
            complete_values = np.empty(0, dtype=np.bool_)
            episode_rewards = np.empty((0, REWARD_SIGNAL_COUNT), dtype=np.float64)
        self._episodes.clear()

        return {
            "transition_count": np.asarray([count], dtype=np.int64),
            "reward_sum": reward_sum,
            "reward_abs_sum": reward_abs_sum,
            "reward_nonzero_count": reward_nonzero_count,
            "reward_cross_product": reward_cross_product,
            "episode_end_worker_step": end_steps,
            "episode_worker_rank": worker_ranks,
            "episode_index": episode_indices,
            "episode_length": lengths,
            "episode_termination_reason": reason_values,
            "episode_complete": complete_values,
            "episode_reward_sums": episode_rewards,
        }


class SafetyTruncationBatch(TypedDict):
    """Compact payload transferred from one environment at rollout boundaries."""

    position_m: NDArray[np.float32]
    termination_reason: NDArray[np.int8]


class SafetyTruncationBuffer:
    """Collect only speed-bound truncations without per-step VecEnv telemetry."""

    def __init__(self) -> None:
        self._positions_m: list[float] = []
        self._termination_reasons: list[int] = []

    def record(
        self,
        *,
        position_m: float,
        termination_reason: TerminationReason | None,
    ) -> None:
        if termination_reason not in {
            TerminationReason.UNDER_LOWER_LIMIT,
            TerminationReason.OVER_UPPER_LIMIT,
            TerminationReason.OVER_SRTSP,
        }:
            return
        position = float(position_m)
        if not np.isfinite(position):
            raise ValueError("safety truncation position must be finite")
        self._positions_m.append(position)
        self._termination_reasons.append(int(termination_reason))

    def drain(self) -> SafetyTruncationBatch:
        payload: SafetyTruncationBatch = {
            "position_m": np.asarray(self._positions_m, dtype=np.float32),
            "termination_reason": np.asarray(self._termination_reasons, dtype=np.int8),
        }
        self._positions_m.clear()
        self._termination_reasons.clear()
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationHistory:
    """Typed periodic evaluation history persisted by the training callback."""

    training_steps: NDArray[np.int64]
    rollout_indices: NDArray[np.int64]
    total_reward: NDArray[np.float64]
    episode_steps: NDArray[np.int64]
    success: NDArray[np.bool_]
    safe: NDArray[np.bool_]
    feasible: NDArray[np.bool_]
    stop_error_m: NDArray[np.float64]
    time_error_s: NDArray[np.float64]
    total_energy_j: NDArray[np.float64]
    comfort_tav: NDArray[np.float64]
    completed_training_episodes: NDArray[np.int64]
    scheduled_completed_training_episodes: NDArray[np.int64]
    route_completion_ratio: NDArray[np.float64]
    safety_violation_positions_m: NDArray[np.float64]
    safety_violation_position_offsets: NDArray[np.int64]

    def __post_init__(self) -> None:
        array_specs: tuple[tuple[str, np.dtype], ...] = (
            ("training_steps", np.dtype(np.int64)),
            ("rollout_indices", np.dtype(np.int64)),
            ("total_reward", np.dtype(np.float64)),
            ("episode_steps", np.dtype(np.int64)),
            ("success", np.dtype(np.bool_)),
            ("safe", np.dtype(np.bool_)),
            ("feasible", np.dtype(np.bool_)),
            ("stop_error_m", np.dtype(np.float64)),
            ("time_error_s", np.dtype(np.float64)),
            ("total_energy_j", np.dtype(np.float64)),
            ("comfort_tav", np.dtype(np.float64)),
            ("completed_training_episodes", np.dtype(np.int64)),
            ("scheduled_completed_training_episodes", np.dtype(np.int64)),
            ("route_completion_ratio", np.dtype(np.float64)),
            ("safety_violation_positions_m", np.dtype(np.float64)),
            ("safety_violation_position_offsets", np.dtype(np.int64)),
        )
        for name, dtype in array_specs:
            array = np.asarray(getattr(self, name), dtype=dtype)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            object.__setattr__(self, name, array.copy())

        series = (
            self.training_steps,
            self.rollout_indices,
            self.total_reward,
            self.episode_steps,
            self.success,
            self.safe,
            self.feasible,
            self.stop_error_m,
            self.time_error_s,
            self.total_energy_j,
            self.comfort_tav,
            self.completed_training_episodes,
            self.scheduled_completed_training_episodes,
            self.route_completion_ratio,
        )
        lengths = {item.size for item in series}
        if len(lengths) != 1:
            raise ValueError("evaluation history series must have equal lengths")
        if np.any(self.scheduled_completed_training_episodes < 0):
            raise ValueError(
                "scheduled_completed_training_episodes must be non-negative"
            )
        if np.any(self.completed_training_episodes < 0):
            raise ValueError("completed_training_episodes must be non-negative")
        if not np.all(np.isfinite(self.route_completion_ratio)) or np.any(
            (self.route_completion_ratio < 0.0) | (self.route_completion_ratio > 1.0)
        ):
            raise ValueError("route_completion_ratio must be finite and within [0, 1]")
        offset_count = np.asarray(self.safety_violation_position_offsets).size
        if offset_count != self.training_steps.size + 1:
            raise ValueError(
                "safety_violation_position_offsets must have one more item "
                "than evaluation history rows"
            )
        offsets = np.asarray(self.safety_violation_position_offsets)
        if offsets.size and (offsets[0] != 0 or np.any(np.diff(offsets) < 0)):
            raise ValueError("safety violation offsets must be monotonic from zero")
        if offsets.size and offsets[-1] != self.safety_violation_positions_m.size:
            raise ValueError(
                "safety violation offsets do not match flattened positions"
            )

    @property
    def abs_time_error_s(self) -> NDArray[np.float64]:
        return np.abs(np.asarray(self.time_error_s, dtype=np.float64))

    def to_npz_mapping(self) -> dict[str, NDArray[np.generic] | NDArray[np.str_]]:
        return {
            "artifact_type": np.asarray([EVALUATION_HISTORY_ARTIFACT_TYPE]),
            "schema_version": np.asarray(
                [EVALUATION_HISTORY_SCHEMA_VERSION], dtype=np.int16
            ),
            "training_steps": np.asarray(self.training_steps, dtype=np.int64),
            "rollout_indices": np.asarray(self.rollout_indices, dtype=np.int64),
            "total_reward": np.asarray(self.total_reward, dtype=np.float64),
            "episode_steps": np.asarray(self.episode_steps, dtype=np.int64),
            "success": np.asarray(self.success, dtype=np.bool_),
            "safe": np.asarray(self.safe, dtype=np.bool_),
            "feasible": np.asarray(self.feasible, dtype=np.bool_),
            "stop_error_m": np.asarray(self.stop_error_m, dtype=np.float64),
            "time_error_s": np.asarray(self.time_error_s, dtype=np.float64),
            "total_energy_j": np.asarray(self.total_energy_j, dtype=np.float64),
            "comfort_tav": np.asarray(self.comfort_tav, dtype=np.float64),
            "completed_training_episodes": np.asarray(
                self.completed_training_episodes, dtype=np.int64
            ),
            "scheduled_completed_training_episodes": np.asarray(
                self.scheduled_completed_training_episodes, dtype=np.int64
            ),
            "route_completion_ratio": np.asarray(
                self.route_completion_ratio, dtype=np.float64
            ),
            "safety_violation_positions_m": np.asarray(
                self.safety_violation_positions_m, dtype=np.float64
            ),
            "safety_violation_position_offsets": np.asarray(
                self.safety_violation_position_offsets, dtype=np.int64
            ),
        }

    @classmethod
    def from_npz_mapping(cls, data: Mapping[str, object]) -> EvaluationHistory:
        required = {
            "artifact_type",
            "schema_version",
            "training_steps",
            "rollout_indices",
            "total_reward",
            "episode_steps",
            "success",
            "safe",
            "feasible",
            "stop_error_m",
            "time_error_s",
            "total_energy_j",
            "comfort_tav",
            "completed_training_episodes",
            "scheduled_completed_training_episodes",
            "route_completion_ratio",
            "safety_violation_positions_m",
            "safety_violation_position_offsets",
        }
        unknown = sorted(set(data) - required)
        if unknown:
            raise ValueError(
                "evaluation history contains unknown arrays: " + ", ".join(unknown)
            )
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(
                "evaluation history is missing required arrays: " + ", ".join(missing)
            )
        artifact_type = np.asarray(data["artifact_type"]).reshape(-1)
        if (
            artifact_type.size != 1
            or str(artifact_type[0]) != EVALUATION_HISTORY_ARTIFACT_TYPE
        ):
            raise ValueError("Unsupported evaluation history artifact_type")
        schema_version = np.asarray(data["schema_version"]).reshape(-1)
        if (
            schema_version.size != 1
            or int(schema_version[0]) != EVALUATION_HISTORY_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported evaluation history schema_version")
        return cls(
            training_steps=np.asarray(data["training_steps"], dtype=np.int64),
            rollout_indices=np.asarray(data["rollout_indices"], dtype=np.int64),
            total_reward=np.asarray(data["total_reward"], dtype=np.float64),
            episode_steps=np.asarray(data["episode_steps"], dtype=np.int64),
            success=np.asarray(data["success"], dtype=np.bool_),
            safe=np.asarray(data["safe"], dtype=np.bool_),
            feasible=np.asarray(data["feasible"], dtype=np.bool_),
            stop_error_m=np.asarray(data["stop_error_m"], dtype=np.float64),
            time_error_s=np.asarray(data["time_error_s"], dtype=np.float64),
            total_energy_j=np.asarray(data["total_energy_j"], dtype=np.float64),
            comfort_tav=np.asarray(data["comfort_tav"], dtype=np.float64),
            completed_training_episodes=np.asarray(
                data["completed_training_episodes"], dtype=np.int64
            ),
            scheduled_completed_training_episodes=np.asarray(
                data["scheduled_completed_training_episodes"], dtype=np.int64
            ),
            route_completion_ratio=np.asarray(
                data["route_completion_ratio"], dtype=np.float64
            ),
            safety_violation_positions_m=np.asarray(
                data["safety_violation_positions_m"], dtype=np.float64
            ),
            safety_violation_position_offsets=np.asarray(
                data["safety_violation_position_offsets"], dtype=np.int64
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class RewardDiagnostics:
    schema_version: NDArray[np.int16]
    reward_names: NDArray[np.str_]
    rollout_end_step: NDArray[np.int64]
    rollout_transition_count: NDArray[np.int64]
    rollout_reward_sum: NDArray[np.float64]
    rollout_reward_abs_sum: NDArray[np.float64]
    rollout_reward_nonzero_count: NDArray[np.int64]
    rollout_reward_cross_product: NDArray[np.float64]
    episode_end_step: NDArray[np.int64]
    episode_worker_rank: NDArray[np.int16]
    episode_index: NDArray[np.int64]
    episode_length: NDArray[np.int32]
    episode_termination_reason: NDArray[np.int8]
    episode_complete: NDArray[np.bool_]
    episode_reward_sums: NDArray[np.float64]

    def to_arrays(self) -> dict[str, NDArray]:
        return {
            "schema_version": self.schema_version,
            "reward_names": self.reward_names,
            "rollout_end_step": self.rollout_end_step,
            "rollout_transition_count": self.rollout_transition_count,
            "rollout_reward_sum": self.rollout_reward_sum,
            "rollout_reward_abs_sum": self.rollout_reward_abs_sum,
            "rollout_reward_nonzero_count": self.rollout_reward_nonzero_count,
            "rollout_reward_cross_product": self.rollout_reward_cross_product,
            "episode_end_step": self.episode_end_step,
            "episode_worker_rank": self.episode_worker_rank,
            "episode_index": self.episode_index,
            "episode_length": self.episode_length,
            "episode_termination_reason": self.episode_termination_reason,
            "episode_complete": self.episode_complete,
            "episode_reward_sums": self.episode_reward_sums,
        }


@dataclass(frozen=True, slots=True, eq=False)
class SafetyTruncationHistogram:
    bin_start_m: NDArray[np.float64]
    bin_end_m: NDArray[np.float64]
    safety_truncation_count: NDArray[np.int64]
    low_safety_truncation_count: NDArray[np.int64]
    high_safety_truncation_count: NDArray[np.int64]
    global_safety_truncation_share: NDArray[np.float64]
    position_bin_size_m: NDArray[np.float64]

    def to_arrays(self) -> dict[str, NDArray]:
        return {
            "bin_start_m": self.bin_start_m,
            "bin_end_m": self.bin_end_m,
            "safety_truncation_count": self.safety_truncation_count,
            "low_safety_truncation_count": self.low_safety_truncation_count,
            "high_safety_truncation_count": self.high_safety_truncation_count,
            "global_safety_truncation_share": self.global_safety_truncation_share,
            "position_bin_size_m": self.position_bin_size_m,
        }


@dataclass(frozen=True, slots=True, eq=False)
class TrainingDiagnostics:
    reward: RewardDiagnostics
    safety: SafetyTruncationHistogram


@dataclass(frozen=True, slots=True, eq=False)
class BestPolicy:
    parameters: dict[str, Any]
    run: RLRun
    quality: QualityReport
    update_reason: str
    training_step: int
    rollout_index: int
