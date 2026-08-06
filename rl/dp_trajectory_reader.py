"""DP artifact adapter for the source-agnostic reference trajectory API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dp.experiment_utils import DP_CURVE_FILENAME, load_dp_curve_artifact
from model.ocs import TrainService
from rl.reference_trajectory_sampler import ReferenceTrajectory
from utils.trajectory import OptimizedCurveArtifact

__all__ = ["DPTrajectoryReader"]


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


class DPTrajectoryReader:
    """Load a task-matching DP curve as a :class:`ReferenceTrajectory`.

    The class deliberately no longer samples DP nodes or constructs RL states.
    Use :class:`rl.reference_trajectory_sampler.ReferenceTrajectorySampler`
    for RL-grid resampling and environment-state reconstruction.
    """

    _METRIC_EXPECTATIONS: tuple[tuple[str, str], ...] = (
        ("target_time_s", "schedule_time"),
        ("start_position_m", "start_position"),
        ("start_speed_mps", "start_speed"),
        ("target_position_m", "target_position"),
    )

    @classmethod
    def from_curve_dir(
        cls,
        *,
        curve_dir: str | Path,
        train_service: TrainService,
        target_speed: float = 0.0,
        match_tolerance: float = 1e-3,
    ) -> ReferenceTrajectory:
        """Load the newest task-matching DP trajectory from ``curve_dir``."""
        artifact = cls.resolve_matching_artifact(
            curve_dir=curve_dir,
            train_service=train_service,
            target_speed=target_speed,
            match_tolerance=match_tolerance,
        )
        return cls.from_artifact(
            artifact=artifact,
            train_service=train_service,
            target_speed=target_speed,
            match_tolerance=match_tolerance,
        )

    @classmethod
    def resolve_matching_artifact(
        cls,
        *,
        curve_dir: str | Path,
        train_service: TrainService,
        target_speed: float = 0.0,
        match_tolerance: float = 1e-3,
    ) -> OptimizedCurveArtifact:
        """Return the newest DP artifact whose metadata matches the task."""
        cls._validate_match_tolerance(match_tolerance)
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
        return artifact

    @classmethod
    def from_artifact(
        cls,
        *,
        artifact: OptimizedCurveArtifact,
        train_service: TrainService,
        target_speed: float = 0.0,
        match_tolerance: float = 1e-3,
    ) -> ReferenceTrajectory:
        """Load one DP artifact without imposing an RL discretization match."""
        cls._validate_match_tolerance(match_tolerance)
        position, speed, cumulative_time, metrics = load_dp_curve_artifact(artifact)
        if not cls._metrics_match_task(
            metrics,
            train_service=train_service,
            target_speed=target_speed,
            match_tolerance=match_tolerance,
        ):
            raise ValueError("DP trajectory artifact metadata does not match the task.")
        return ReferenceTrajectory(
            position_m=np.asarray(position, dtype=np.float64),
            speed_mps=np.asarray(speed, dtype=np.float64),
            cumulative_time_s=np.asarray(cumulative_time, dtype=np.float64),
            metadata=metrics,
        )

    @staticmethod
    def _validate_match_tolerance(match_tolerance: float) -> None:
        if not np.isfinite(match_tolerance) or match_tolerance < 0.0:
            raise ValueError("match_tolerance must be a finite non-negative value")

    @staticmethod
    def _read_metrics(metrics_path: Path) -> dict[str, object] | None:
        if not metrics_path.is_file():
            return None
        try:
            with metrics_path.open(encoding="utf-8") as file:
                metrics = json.load(file)
        except OSError, json.JSONDecodeError:
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
