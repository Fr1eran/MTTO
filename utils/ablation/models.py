"""Typed data models shared by the ablation command line tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

RunStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class MetricStats:
    """Point-wise statistics for one metric."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: NDArray[np.int64]


@dataclass(frozen=True)
class CurveAggregate:
    """A curve aggregate whose x-axis is explicit rather than implicit."""

    variant_id: str
    x_name: str
    x: NDArray[np.float64]
    metrics: Mapping[str, MetricStats]
    valid_run_count: int
    label: str | None = None
    color: str | None = None
    axes: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def means(self) -> dict[str, NDArray[np.float64]]:
        return {key: value.mean for key, value in self.metrics.items()}

    @property
    def stds(self) -> dict[str, NDArray[np.float64]]:
        return {key: value.std for key, value in self.metrics.items()}

    def axis_for(self, metric: str) -> NDArray[np.float64]:
        """Return the explicit x-axis associated with a metric."""
        return self.axes.get(metric, self.x)


@dataclass(frozen=True)
class FinalMetricAggregate:
    """Final metrics grouped by one ablation variant."""

    variant_id: str
    metrics: Mapping[str, MetricStats]
    valid_run_count: int
    success_rate: float
    label: str | None = None
    color: str | None = None

    @property
    def means(self) -> dict[str, float]:
        return {
            key: float(np.asarray(value.mean).reshape(-1)[0])
            for key, value in self.metrics.items()
        }

    @property
    def stds(self) -> dict[str, float]:
        return {
            key: float(np.asarray(value.std).reshape(-1)[0])
            for key, value in self.metrics.items()
        }


@dataclass(frozen=True)
class ArtifactLayout:
    """Canonical artifact names for one RL run.

    ``legacy_paths`` is retained only for the explicit, opt-in migration
    helper.  Production training and aggregation never depend on it.
    """

    run_root: Path
    policy_final: Path
    metadata: Path
    episodes: Path
    evaluations: Path
    trajectory_final: Path
    trajectory_best: Path | None
    metrics_final: Path
    metrics_best: Path | None
    safety_diagnostics: Path
    legacy_paths: Mapping[str, Path | None] = field(default_factory=dict)

    @classmethod
    def from_training_spec(cls, spec: Any) -> ArtifactLayout:
        """Build canonical paths from the existing RL TrainingRunSpec."""
        output_dir = Path(spec.output_dir)
        final_dir = Path(spec.final_output_dir)
        best_dir_raw = getattr(spec, "best_eval_output_dir", None)
        best_enabled = bool(getattr(spec, "enable_best_evaluation_artifacts", False))
        best_dir = Path(best_dir_raw) if best_enabled and best_dir_raw else None
        evaluation_history_raw = getattr(spec, "evaluation_history_path", None)
        return cls(
            run_root=output_dir,
            policy_final=final_dir / "policy_final.zip",
            metadata=output_dir / "metadata.json",
            episodes=final_dir / "episodes.npz",
            evaluations=final_dir / "evaluations.npz",
            trajectory_final=final_dir / "final_trajectory.npz",
            trajectory_best=(
                best_dir / "best_trajectory.npz" if best_dir is not None else None
            ),
            metrics_final=final_dir / "metrics_final.json",
            metrics_best=(
                best_dir / "metrics_best.json" if best_dir is not None else None
            ),
            safety_diagnostics=final_dir / "safety_diagnostics.npz",
            legacy_paths={
                "policy_final": _path_or_none(
                    getattr(spec, "final_model_save_path", None)
                ),
                "metadata": _path_or_none(getattr(spec, "run_metadata_path", None)),
                "episodes": _path_or_none(
                    getattr(spec, "reward_diagnostics_path", None)
                ),
                "evaluations": _path_or_none(evaluation_history_raw),
                "metrics_final": final_dir / "final_trajectory_metrics.json",
                "metrics_best": (
                    best_dir / "best_trajectory_metrics.json"
                    if best_dir is not None
                    else None
                ),
                "safety_diagnostics": (
                    final_dir / "safety_truncation_position_histogram.npz"
                ),
            },
        )


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if str(path) else None
