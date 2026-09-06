"""Canonical artifact names and safe normalization from RL outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from contracts.ablation import AblationRunRecord
from contracts.training import RunMetadata, TrainingBudget

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
        layout.metadata,
        layout.episodes,
        layout.trajectory_final,
        layout.metrics_final,
    ]
    if require_evaluations:
        required.append(layout.evaluations)
    return all(path.is_file() for path in required)


def training_budget_complete(
    budget: TrainingBudget | None,
    *,
    expected_effective_episodes: int | None = None,
) -> bool:
    """Return whether a completed-episode budget reached its effective target."""
    if budget is None or budget.mode != "completed_episodes":
        return False
    effective = budget.effective_training_episodes
    actual = budget.actual_completed_episodes
    if effective is None or actual is None or effective <= 0:
        return False
    if expected_effective_episodes is not None and effective != int(
        expected_effective_episodes
    ):
        return False
    return budget.target_reached is True and actual >= effective


def _load_metadata_budget(path: Path) -> TrainingBudget | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            metadata = RunMetadata.from_mapping(json.load(file_obj))
    except (OSError, TypeError, ValueError):
        return None
    return metadata.training_budget


def canonical_training_run_complete(
    layout: ArtifactLayout,
    *,
    expected_effective_episodes: int | None = None,
    require_evaluations: bool = True,
) -> bool:
    """Check canonical artifacts and the persisted completed-episode budget."""
    return canonical_artifacts_complete(
        layout, require_evaluations=require_evaluations
    ) and training_budget_complete(
        _load_metadata_budget(layout.metadata),
        expected_effective_episodes=expected_effective_episodes,
    )


def manifest_run_complete(
    run: AblationRunRecord,
    *,
    require_evaluations: bool = True,
) -> bool:
    """Validate a completed manifest record against its referenced artifacts."""
    if run.status != "completed" or not training_budget_complete(run.training_budget):
        return False
    required = (
        "policy_final",
        "metadata",
        "episodes",
        "trajectory_final",
        "metrics_final",
    )
    if require_evaluations:
        required += ("evaluations",)
    try:
        if not all(Path(run.artifacts.path_for(name)).is_file() for name in required):
            return False
        metadata_budget = _load_metadata_budget(
            Path(run.artifacts.path_for("metadata"))
        )
    except (KeyError, OSError):
        return False
    expected = run.training_budget.effective_training_episodes
    return training_budget_complete(
        metadata_budget, expected_effective_episodes=expected
    )
