"""Load and sample validated discrete states from DP reference trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dp.experiment_utils import DP_CURVE_FILENAME, load_dp_curve_artifact
from model.ocs import TrainService
from utils.trajectory import OptimizedCurveArtifact, compute_segment_accelerations

__all__ = ["DPTrajectorySampler", "DPTrajectoryState"]


@dataclass(frozen=True)
class DPTrajectoryState:
    """A physically meaningful state taken from one discrete DP trajectory node."""

    reference_index: int
    position_m: float
    speed_mps: float
    acceleration_mps2: float
    operation_time_s: float


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


class DPTrajectorySampler:
    """Validated sampler over the discrete nodes of a DP trajectory artifact."""

    _METRIC_EXPECTATIONS = (
        ("target_time_s", "schedule_time"),
        ("start_position_m", "start_position"),
        ("start_speed_mps", "start_speed"),
        ("target_position_m", "target_position"),
    )

    def __init__(
        self,
        *,
        artifact: OptimizedCurveArtifact,
        train_service: TrainService,
        target_speed: float,
        max_step_distance_m: float,
        match_tolerance: float,
        pos_arr: NDArray[np.float64],
        speed_arr: NDArray[np.float64],
        cum_time_arr: NDArray[np.float64],
        metrics: dict[str, object],
    ) -> None:
        self._artifact = artifact
        self._train_service = train_service
        self._target_speed = float(target_speed)
        self._max_step_distance_m = float(max_step_distance_m)
        self._match_tolerance = float(match_tolerance)
        self._pos_arr = pos_arr
        self._speed_arr = speed_arr
        self._cum_time_arr = cum_time_arr
        self._metrics = dict(metrics)

        segment_acc = compute_segment_accelerations(pos_arr, speed_arr)
        self._acc_arr = np.empty(pos_arr.size, dtype=np.float64)
        # The first reference node has no incoming transition.  Keep it
        # consistent with MTTOEnv.reset(), whose initial acceleration is zero.
        self._acc_arr[0] = 0.0
        self._acc_arr[1:] = segment_acc
        self._remaining_distance_arr = np.abs(
            float(train_service.target_position) - pos_arr
        )

    @classmethod
    def from_curve_dir(
        cls,
        *,
        curve_dir: str | Path,
        train_service: TrainService,
        max_step_distance_m: float,
        target_speed: float = 0.0,
        match_tolerance: float = 1e-3,
    ) -> DPTrajectorySampler:
        """Load the newest DP artifact under ``curve_dir`` matching this task."""
        cls._validate_match_tolerance(match_tolerance)
        cls._validate_max_step_distance(max_step_distance_m)
        search_root = Path(curve_dir)
        if not search_root.is_dir():
            raise FileNotFoundError(
                f"DP trajectory directory does not exist: {search_root}"
            )

        candidates: list[OptimizedCurveArtifact] = []
        for curve_path in search_root.rglob(DP_CURVE_FILENAME):
            if not curve_path.is_file():
                continue
            metrics_path = curve_path.with_name(f"{curve_path.stem}_metrics.json")
            metrics = cls._read_metrics(metrics_path)
            if metrics is not None and cls._metrics_match_task(
                metrics,
                train_service=train_service,
                target_speed=target_speed,
                match_tolerance=match_tolerance,
            ) and cls._metrics_match_discretization(
                metrics,
                max_step_distance_m=max_step_distance_m,
            ):
                candidates.append(
                    OptimizedCurveArtifact(
                        npz_path=str(curve_path),
                        metrics_path=str(metrics_path),
                    )
                )

        if not candidates:
            raise FileNotFoundError(
                "Could not find a DP trajectory artifact matching the requested task."
            )

        artifact = max(
            candidates,
            key=lambda item: (Path(item.npz_path).stat().st_mtime, item.npz_path),
        )
        return cls.from_artifact(
            artifact=artifact,
            train_service=train_service,
            target_speed=target_speed,
            max_step_distance_m=max_step_distance_m,
            match_tolerance=match_tolerance,
        )

    @classmethod
    def from_artifact(
        cls,
        *,
        artifact: OptimizedCurveArtifact,
        train_service: TrainService,
        max_step_distance_m: float,
        target_speed: float = 0.0,
        match_tolerance: float = 1e-3,
    ) -> DPTrajectorySampler:
        """Load one explicitly selected DP trajectory artifact."""
        cls._validate_match_tolerance(match_tolerance)
        cls._validate_max_step_distance(max_step_distance_m)
        pos_arr, speed_arr, cum_time_arr, metrics = load_dp_curve_artifact(artifact)
        if not cls._metrics_match_task(
            metrics,
            train_service=train_service,
            target_speed=target_speed,
            match_tolerance=match_tolerance,
        ):
            raise ValueError("DP trajectory artifact metadata does not match the task.")
        cls._validate_discretization_metadata(
            metrics,
            max_step_distance_m=max_step_distance_m,
        )

        pos, speed, cum_time = cls._validate_arrays(
            pos_arr=pos_arr,
            speed_arr=speed_arr,
            cum_time_arr=cum_time_arr,
            train_service=train_service,
            match_tolerance=match_tolerance,
        )
        return cls(
            artifact=artifact,
            train_service=train_service,
            target_speed=target_speed,
            max_step_distance_m=max_step_distance_m,
            match_tolerance=match_tolerance,
            pos_arr=pos,
            speed_arr=speed,
            cum_time_arr=cum_time,
            metrics=metrics,
        )

    @property
    def artifact(self) -> OptimizedCurveArtifact:
        return self._artifact

    @property
    def node_count(self) -> int:
        return int(self._pos_arr.size)

    @property
    def positions_m(self) -> NDArray[np.float64]:
        return self._pos_arr.copy()

    @property
    def speeds_mps(self) -> NDArray[np.float64]:
        return self._speed_arr.copy()

    @property
    def cumulative_times_s(self) -> NDArray[np.float64]:
        return self._cum_time_arr.copy()

    @property
    def metrics(self) -> dict[str, object]:
        return dict(self._metrics)

    @property
    def max_step_distance_m(self) -> float:
        return self._max_step_distance_m

    def state_at(self, index: int) -> DPTrajectoryState:
        """Return the reference state at a single, non-negative node index."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")
        index_value = int(index)
        if not 0 <= index_value < self.node_count:
            raise IndexError(
                f"index {index_value} is outside [0, {self.node_count - 1}]"
            )
        return DPTrajectoryState(
            reference_index=index_value,
            position_m=float(self._pos_arr[index_value]),
            speed_mps=float(self._speed_arr[index_value]),
            acceleration_mps2=float(self._acc_arr[index_value]),
            operation_time_s=float(self._cum_time_arr[index_value]),
        )

    def sample(
        self,
        rng: np.random.Generator,
        *,
        index_range: tuple[int, int] | None = None,
        remaining_distance_range_m: tuple[float, float] | None = None,
    ) -> DPTrajectoryState:
        """Uniformly sample a discrete node within one inclusive selection range."""
        if index_range is not None and remaining_distance_range_m is not None:
            raise ValueError(
                "Specify at most one of index_range and remaining_distance_range_m."
            )
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        if index_range is not None:
            lower, upper = self._validate_index_range(index_range)
            candidate_indices = np.arange(lower, upper + 1, dtype=np.int64)
        elif remaining_distance_range_m is not None:
            lower, upper = self._validate_remaining_distance_range(
                remaining_distance_range_m
            )
            candidate_indices = np.flatnonzero(
                (self._remaining_distance_arr >= lower)
                & (self._remaining_distance_arr <= upper)
            )
        else:
            candidate_indices = np.arange(self.node_count, dtype=np.int64)

        if candidate_indices.size == 0:
            raise ValueError("The requested sampling range contains no DP nodes.")
        selected = int(rng.choice(candidate_indices))
        return self.state_at(selected)

    @staticmethod
    def _validate_match_tolerance(match_tolerance: float) -> None:
        if not np.isfinite(match_tolerance) or match_tolerance < 0.0:
            raise ValueError("match_tolerance must be a finite non-negative value")

    @staticmethod
    def _validate_max_step_distance(max_step_distance_m: float) -> None:
        if not np.isfinite(max_step_distance_m) or max_step_distance_m <= 0.0:
            raise ValueError("max_step_distance_m must be a finite positive value")

    @staticmethod
    def _read_metrics(metrics_path: Path) -> dict[str, object] | None:
        if not metrics_path.is_file():
            return None
        try:
            with metrics_path.open(encoding="utf-8") as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        return metrics if isinstance(metrics, dict) else None

    @classmethod
    def _metrics_match_task(
        cls,
        metrics: dict[str, object],
        *,
        train_service: TrainService,
        target_speed: float,
        match_tolerance: float,
    ) -> bool:
        expected_values = {
            metric_key: float(getattr(train_service, service_attr))
            for metric_key, service_attr in cls._METRIC_EXPECTATIONS
        }
        expected_values["target_speed_mps"] = float(target_speed)
        return all(
            (actual := _metric_as_float(metrics.get(key))) is not None
            and abs(actual - expected) <= match_tolerance
            for key, expected in expected_values.items()
        )

    @staticmethod
    def _metrics_match_discretization(
        metrics: dict[str, object],
        *,
        max_step_distance_m: float,
    ) -> bool:
        saved_step_distance = _metric_as_float(metrics.get("max_step_distance_m"))
        return (
            metrics.get("stage_division") == "uniform"
            and saved_step_distance is not None
            and abs(saved_step_distance - float(max_step_distance_m)) <= 1e-9
        )

    @classmethod
    def _validate_discretization_metadata(
        cls,
        metrics: dict[str, object],
        *,
        max_step_distance_m: float,
    ) -> None:
        if metrics.get("stage_division") != "uniform":
            raise ValueError(
                "DP trajectory stage_division must be 'uniform' for sampling."
            )
        saved_step_distance = _metric_as_float(metrics.get("max_step_distance_m"))
        if saved_step_distance is None:
            raise ValueError("DP trajectory max_step_distance_m metadata is missing.")
        if not cls._metrics_match_discretization(
            metrics,
            max_step_distance_m=max_step_distance_m,
        ):
            raise ValueError(
                "DP trajectory max_step_distance_m does not match the environment."
            )

    @staticmethod
    def _validate_arrays(
        *,
        pos_arr: Any,
        speed_arr: Any,
        cum_time_arr: Any,
        train_service: TrainService,
        match_tolerance: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        pos = np.asarray(pos_arr, dtype=np.float64)
        speed = np.asarray(speed_arr, dtype=np.float64)
        cum_time = np.asarray(cum_time_arr, dtype=np.float64)
        if pos.ndim != 1 or speed.ndim != 1 or cum_time.ndim != 1:
            raise ValueError("DP trajectory arrays must be one-dimensional")
        if not (pos.size == speed.size == cum_time.size):
            raise ValueError("DP trajectory arrays must have equal length")
        if pos.size < 2:
            raise ValueError("DP trajectory must contain at least two nodes")
        if not (
            np.all(np.isfinite(pos))
            and np.all(np.isfinite(speed))
            and np.all(np.isfinite(cum_time))
        ):
            raise ValueError("DP trajectory arrays must contain only finite values")
        if np.any(speed < 0.0):
            raise ValueError("DP trajectory speeds must be non-negative")
        if not np.isclose(pos[0], train_service.start_position, atol=match_tolerance):
            raise ValueError("DP trajectory start position does not match the task")
        if not np.isclose(pos[-1], train_service.target_position, atol=match_tolerance):
            raise ValueError("DP trajectory target position does not match the task")

        direction = np.sign(
            train_service.target_position - train_service.start_position
        )
        if direction == 0.0:
            raise ValueError("Train service start and target positions must differ")
        if np.any(direction * np.diff(pos) <= 0.0):
            raise ValueError("DP trajectory positions must be strictly monotonic")
        if np.any(np.diff(cum_time) <= 0.0):
            raise ValueError(
                "DP trajectory cumulative times must be strictly increasing"
            )
        return pos, speed, cum_time

    def _validate_index_range(self, index_range: tuple[int, int]) -> tuple[int, int]:
        if len(index_range) != 2:
            raise ValueError("index_range must contain exactly two indices")
        lower, upper = index_range
        if not isinstance(lower, (int, np.integer)) or not isinstance(
            upper, (int, np.integer)
        ):
            raise TypeError("index_range values must be integers")
        lower_value, upper_value = int(lower), int(upper)
        if lower_value > upper_value:
            raise ValueError("index_range lower bound must not exceed upper bound")
        if lower_value < 0 or upper_value >= self.node_count:
            raise ValueError(
                f"index_range must stay within [0, {self.node_count - 1}]"
            )
        return lower_value, upper_value

    @staticmethod
    def _validate_remaining_distance_range(
        remaining_distance_range_m: tuple[float, float],
    ) -> tuple[float, float]:
        if len(remaining_distance_range_m) != 2:
            raise ValueError(
                "remaining_distance_range_m must contain exactly two values"
            )
        lower, upper = map(float, remaining_distance_range_m)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0.0:
            raise ValueError(
                "remaining_distance_range_m must contain finite non-negative values"
            )
        if lower > upper:
            raise ValueError(
                "remaining_distance_range_m lower bound must not exceed upper bound"
            )
        return lower, upper
