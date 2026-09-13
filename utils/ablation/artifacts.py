"""Canonical artifact names and safe normalization from RL outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contracts.ablation import AblationRunRecord
from contracts.training import RunMetadata, TrainingBudget

from .models import ArtifactLayout


def artifact_paths(
    layout: ArtifactLayout, *, relative_to: str | Path | None = None
) -> dict[str, str]:
    """Serialize canonical artifact paths for a manifest entry."""

    def encode(path: Path) -> str:
        resolved = path.resolve()
        if relative_to is None:
            return str(resolved)
        return str(resolved.relative_to(Path(relative_to).resolve()))

    paths = {
        "policy_final": encode(layout.policy_final),
        "metadata": encode(layout.metadata),
        "episodes": encode(layout.episodes),
        "evaluations": encode(layout.evaluations),
        "trajectory_final": encode(layout.trajectory_final),
        "metrics_final": encode(layout.metrics_final),
        "safety_diagnostics": encode(layout.safety_diagnostics),
    }
    if layout.policy_best is not None:
        paths["policy_best"] = encode(layout.policy_best)
    if layout.metadata_best is not None:
        paths["metadata_best"] = encode(layout.metadata_best)
    if layout.metrics_best is not None:
        paths["metrics_best"] = encode(layout.metrics_best)
    if layout.trajectory_best is not None:
        paths["trajectory_best"] = encode(layout.trajectory_best)
    return paths


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
    if layout.policy_best is not None:
        required.extend(
            (
                layout.policy_best,
                layout.metadata_best,
                layout.trajectory_best,
                layout.metrics_best,
            )
        )
    return all(path.is_file() for path in required)


def training_budget_complete(
    budget: TrainingBudget | None,
    *,
    expected_effective_episodes: int | None = None,
    expected_training_timesteps: int | None = None,
    expected_training_rollouts: int | None = None,
) -> bool:
    """Return whether an episode or environment-step budget reached its target."""
    if budget is None or budget.target_reached is not True:
        return False
    if budget.mode == "environment_steps":
        actual_steps = budget.actual_training_timesteps
        actual_rollouts = budget.actual_training_rollouts
        target_steps = budget.derived_total_timesteps
        target_rollouts = budget.training_rollouts
        if actual_steps is None or actual_steps < target_steps or target_steps <= 0:
            return False
        if target_rollouts is None or actual_rollouts is None:
            return False
        if actual_rollouts < target_rollouts:
            return False
        if expected_training_timesteps is not None and target_steps != int(
            expected_training_timesteps
        ):
            return False
        if expected_training_rollouts is not None and target_rollouts != int(
            expected_training_rollouts
        ):
            return False
        return True
    if budget.mode != "completed_episodes":
        return False
    effective = budget.effective_training_episodes
    actual = budget.actual_completed_episodes
    if effective is None or actual is None or effective <= 0:
        return False
    if expected_effective_episodes is not None and effective != int(
        expected_effective_episodes
    ):
        return False
    return actual >= effective


def _load_metadata_budget(path: Path) -> TrainingBudget | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            metadata = RunMetadata.from_mapping(json.load(file_obj))
    except OSError, TypeError, ValueError:
        return None
    return metadata.training_budget


def canonical_training_run_complete(
    layout: ArtifactLayout,
    *,
    expected_effective_episodes: int | None = None,
    expected_training_timesteps: int | None = None,
    expected_training_rollouts: int | None = None,
    require_evaluations: bool = True,
) -> bool:
    """Check canonical artifacts and the persisted completed-episode budget."""
    return canonical_artifacts_complete(
        layout, require_evaluations=require_evaluations
    ) and training_budget_complete(
        _load_metadata_budget(layout.metadata),
        expected_effective_episodes=expected_effective_episodes,
        expected_training_timesteps=expected_training_timesteps,
        expected_training_rollouts=expected_training_rollouts,
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
    best_names = (
        "policy_best",
        "metadata_best",
        "trajectory_best",
        "metrics_best",
    )
    if any(getattr(run.artifacts, name) is not None for name in best_names):
        required += best_names
    try:
        if not all(Path(run.artifacts.path_for(name)).is_file() for name in required):
            return False
        metadata_budget = _load_metadata_budget(
            Path(run.artifacts.path_for("metadata"))
        )
    except KeyError, OSError:
        return False
    expected = run.training_budget.effective_training_episodes
    return training_budget_complete(
        metadata_budget,
        expected_effective_episodes=expected,
        expected_training_timesteps=run.training_budget.derived_total_timesteps,
        expected_training_rollouts=run.training_budget.training_rollouts,
    )
