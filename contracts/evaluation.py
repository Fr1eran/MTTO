"""Canonical RL evaluation metrics and trajectory contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .common import (
    ContractError,
    JSONMapping,
    JSONValue,
    from_dict,
    to_dict,
)

EVALUATION_METRICS_ARTIFACT_TYPE = "rl_evaluation_metrics"
EVALUATION_METRICS_SCHEMA_VERSION = 2
EVALUATION_HISTORY_ARTIFACT_TYPE = "rl_evaluation_history"
EVALUATION_HISTORY_SCHEMA_VERSION = 3
SAFETY_MARGIN_EPS_MPS = 1e-6


def is_successful_evaluation(*, terminated: bool, truncated: bool) -> bool:
    """Return whether an evaluation ended in normal task completion."""
    return bool(terminated and not truncated)


def is_precise_evaluation(
    *, success: bool, stop_error_m: float, stop_error_limit_m: float
) -> bool:
    """Return the canonical stopping-accuracy status (inclusive limit)."""
    if stop_error_limit_m < 0.0:
        raise ValueError("stop_error_limit_m must be non-negative")
    return bool(success and abs(float(stop_error_m)) <= stop_error_limit_m)


def is_punctual_evaluation(
    *, precise_arrival: bool, time_error_s: float, time_error_limit_s: float
) -> bool:
    """Return the canonical punctuality status (exclusive limit)."""
    if time_error_limit_s < 0.0:
        raise ValueError("time_error_limit_s must be non-negative")
    return bool(precise_arrival and abs(float(time_error_s)) < time_error_limit_s)


def is_safe_evaluation(
    *,
    min_safety_margin_mps: float,
    safety_violation_count: int,
    safety_margin_eps_mps: float = SAFETY_MARGIN_EPS_MPS,
) -> bool:
    """Return the canonical safety status for one evaluated trajectory."""
    if safety_violation_count < 0:
        raise ValueError("safety_violation_count must be non-negative")
    if safety_margin_eps_mps < 0.0:
        raise ValueError("safety_margin_eps_mps must be non-negative")
    return bool(
        min_safety_margin_mps >= -safety_margin_eps_mps and safety_violation_count == 0
    )


def is_feasible_evaluation(
    *,
    success: bool,
    precise_arrival: bool,
    punctual_arrival: bool,
    safe: bool,
) -> bool:
    """Return the canonical strict-feasibility status."""
    return bool(success and precise_arrival and punctual_arrival and safe)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics(Mapping[str, object]):
    """Scalar outcome of one policy evaluation."""

    success: bool
    precise_arrival: bool
    punctual_arrival: bool
    total_reward: float
    total_time_s: float
    target_time_s: float
    time_error_s: float
    start_position_m: float
    target_position_m: float
    final_position_m: float
    final_speed_mps: float
    stop_error_m: float
    total_energy_j: float
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float
    terminated: bool
    truncated: bool
    episode_steps: int = field(metadata={"minimum": 0})
    min_safety_margin_mps: float
    mean_safety_margin_mps: float
    safety_violation_count: int = field(metadata={"minimum": 0})
    safe: bool
    feasible: bool
    strict_stop_error_limit_m: float
    strict_time_error_limit_s: float
    selection_comparison_key: tuple[float, ...] = field(metadata={"non_empty": True})
    selection_rule: str | None = None
    num_timesteps: int | None = field(default=None, metadata={"minimum": 0})
    evaluation_rollout_index: int | None = field(default=None, metadata={"minimum": 0})
    created_at: str | None = None
    extensions: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episode_steps < 0:
            raise ValueError("episode_steps must be non-negative")
        if self.safety_violation_count < 0:
            raise ValueError("safety_violation_count must be non-negative")
        expected_success = is_successful_evaluation(
            terminated=self.terminated,
            truncated=self.truncated,
        )
        if self.success != expected_success:
            raise ValueError("success does not match the canonical completion rule")
        expected_precise = is_precise_evaluation(
            success=self.success,
            stop_error_m=self.stop_error_m,
            stop_error_limit_m=self.strict_stop_error_limit_m,
        )
        if self.precise_arrival != expected_precise:
            raise ValueError(
                "precise_arrival does not match the canonical stopping rule"
            )
        expected_punctual = is_punctual_evaluation(
            precise_arrival=self.precise_arrival,
            time_error_s=self.time_error_s,
            time_error_limit_s=self.strict_time_error_limit_s,
        )
        if self.punctual_arrival != expected_punctual:
            raise ValueError(
                "punctual_arrival does not match the canonical punctuality rule"
            )
        expected_safe = is_safe_evaluation(
            min_safety_margin_mps=self.min_safety_margin_mps,
            safety_violation_count=self.safety_violation_count,
        )
        if self.safe != expected_safe:
            raise ValueError("safe does not match the canonical safety rule")
        expected_feasible = is_feasible_evaluation(
            success=self.success,
            precise_arrival=self.precise_arrival,
            punctual_arrival=self.punctual_arrival,
            safe=self.safe,
        )
        if self.feasible != expected_feasible:
            raise ValueError("feasible does not match the canonical constraint rule")
        if not self.selection_comparison_key:
            raise ValueError("selection_comparison_key must not be empty")
        if self.num_timesteps is not None and self.num_timesteps < 0:
            raise ValueError("num_timesteps must be non-negative")
        if (
            self.evaluation_rollout_index is not None
            and self.evaluation_rollout_index < 0
        ):
            raise ValueError("evaluation_rollout_index must be non-negative")
        for value in self.selection_comparison_key:
            if not np.isfinite(value):
                raise ValueError("selection_comparison_key must contain finite numbers")

    @property
    def total_energy_kj(self) -> float:
        """Presentation-only conversion; J is the canonical stored unit."""
        return self.total_energy_j / 1000.0

    def __getitem__(self, key: str) -> object:
        return self.to_display_mapping()[key]

    def __iter__(self):
        return iter(self.to_display_mapping())

    def __len__(self) -> int:
        return len(self.to_display_mapping())

    def to_mapping(self) -> JSONMapping:
        return to_dict(
            self,
            headers={
                "artifact_type": EVALUATION_METRICS_ARTIFACT_TYPE,
                "schema_version": EVALUATION_METRICS_SCHEMA_VERSION,
            },
            compact=True,
            late_optional=True,
        )

    def extension(self, name: str, default: object = None) -> object:
        """Read an explicitly namespaced optional artifact field."""
        return self.extensions.get(name, default)

    def to_display_mapping(self) -> dict[str, object]:
        """Flatten extensions for presentation-only consumers."""
        payload = dict(self.to_mapping())
        payload["total_energy_kj"] = self.total_energy_kj
        extensions = payload.pop("extensions", {})
        if isinstance(extensions, Mapping):
            payload.update(extensions)
        return payload

    @classmethod
    def from_mapping(cls, payload: object) -> EvaluationMetrics:
        return from_dict(
            cls,
            payload,
            context="evaluation_metrics",
            headers={
                "artifact_type": EVALUATION_METRICS_ARTIFACT_TYPE,
                "schema_version": EVALUATION_METRICS_SCHEMA_VERSION,
            },
        )


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    """Array payload accompanying evaluation metrics."""

    position_m: NDArray[np.float32]
    speed_mps: NDArray[np.float32]
    safety_violation_positions_m: NDArray[np.float32]

    def __post_init__(self) -> None:
        for name in (
            "position_m",
            "speed_mps",
            "safety_violation_positions_m",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float32)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            object.__setattr__(self, name, array.copy())
        if self.position_m.size != self.speed_mps.size:
            raise ValueError("position_m and speed_mps must have equal length")

    def to_npz_mapping(self) -> dict[str, NDArray[np.float32]]:
        return {
            "pos_m": self.position_m,
            "speed_mps": self.speed_mps,
            "safety_violation_positions_m": self.safety_violation_positions_m,
        }


@dataclass(frozen=True, slots=True)
class EvaluationHistory:
    """Typed periodic evaluation history persisted by the training callback."""

    training_steps: NDArray[np.int64]
    rollout_indices: NDArray[np.int64]
    total_reward: NDArray[np.float64]
    episode_steps: NDArray[np.int64]
    success: NDArray[np.bool_]
    safe: NDArray[np.bool_]
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
            raise ContractError(
                "evaluation history contains unknown arrays: " + ", ".join(unknown)
            )
        missing = sorted(required - set(data))
        if missing:
            raise ContractError(
                "evaluation history is missing required arrays: " + ", ".join(missing)
            )
        artifact_type = np.asarray(data["artifact_type"]).reshape(-1)
        if (
            artifact_type.size != 1
            or str(artifact_type[0]) != EVALUATION_HISTORY_ARTIFACT_TYPE
        ):
            raise ContractError("Unsupported evaluation history artifact_type")
        schema_version = np.asarray(data["schema_version"]).reshape(-1)
        if (
            schema_version.size != 1
            or int(schema_version[0]) != EVALUATION_HISTORY_SCHEMA_VERSION
        ):
            raise ContractError("Unsupported evaluation history schema_version")
        return cls(
            training_steps=np.asarray(data["training_steps"], dtype=np.int64),
            rollout_indices=np.asarray(data["rollout_indices"], dtype=np.int64),
            total_reward=np.asarray(data["total_reward"], dtype=np.float64),
            episode_steps=np.asarray(data["episode_steps"], dtype=np.int64),
            success=np.asarray(data["success"], dtype=np.bool_),
            safe=np.asarray(data["safe"], dtype=np.bool_),
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


@dataclass(frozen=True, slots=True)
class EvaluationArtifact:
    """Complete evaluation result independent of its filesystem layout."""

    metrics: EvaluationMetrics
    trajectory: TrajectoryData

    def __post_init__(self) -> None:
        violation_count = int(self.trajectory.safety_violation_positions_m.size)
        if self.metrics.safety_violation_count != violation_count:
            raise ValueError(
                "metrics safety_violation_count does not match trajectory violations"
            )
