"""Canonical artifact names and safe normalization from RL outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .models import ArtifactLayout


def artifact_paths(layout: ArtifactLayout) -> dict[str, str]:
    """Serialize canonical artifact paths for a manifest entry."""
    paths = {
        "policy_final": str(layout.policy_final.resolve()),
        "metadata": str(layout.metadata.resolve()),
        "episodes": str(layout.episodes.resolve()),
        "evaluations": str(layout.evaluations.resolve()),
        "trajectory_final": str(layout.trajectory_final.resolve()),
        "metrics_final": str(layout.metrics_final.resolve()),
        "safety_diagnostics": str(layout.safety_diagnostics.resolve()),
    }
    if layout.metrics_best is not None:
        paths["metrics_best"] = str(layout.metrics_best.resolve())
    if layout.trajectory_best is not None:
        paths["trajectory_best"] = str(layout.trajectory_best.resolve())
    return paths


def materialize_canonical_artifacts(layout: ArtifactLayout) -> None:
    """Explicitly migrate legacy files; never called by normal workflows."""
    for key, source in layout.legacy_paths.items():
        target = getattr(layout, key)
        if target is None or source is None or target == source:
            continue
        if target.is_file() or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def load_npz_arrays(
    path: str | Path,
    required: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Load selected NPZ arrays with pickle disabled and schema checking."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"NPZ artifact not found: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing {missing} in {artifact_path}")
        return {key: np.asarray(data[key]).copy() for key in required}


def canonical_artifacts_complete(
    layout: ArtifactLayout,
    *,
    require_evaluations: bool = True,
) -> bool:
    required = [
        layout.policy_final,
        layout.episodes,
        layout.trajectory_final,
        layout.metrics_final,
    ]
    if require_evaluations:
        required.append(layout.evaluations)
    return all(path.is_file() for path in required)
