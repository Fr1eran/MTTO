"""Stable data contracts shared by domain and persistence boundaries."""

from .ablation import (
    ABLATION_MANIFEST_ARTIFACT_TYPE,
    ABLATION_MANIFEST_SCHEMA_VERSION,
    AblationManifest,
    AblationRunRecord,
    ArtifactRefs,
    ManifestStatusUpdate,
)
from .common import ContractError, JSONMapping, JSONValue
from .environment import EpisodeInfo, EpisodeOutcome
from .evaluation import (
    EVALUATION_HISTORY_ARTIFACT_TYPE,
    EVALUATION_HISTORY_SCHEMA_VERSION,
    EVALUATION_METRICS_ARTIFACT_TYPE,
    EVALUATION_METRICS_SCHEMA_VERSION,
    SAFETY_MARGIN_EPS_MPS,
    EvaluationArtifact,
    EvaluationHistory,
    EvaluationMetrics,
    TrajectoryData,
    is_feasible_evaluation,
    is_precise_evaluation,
    is_punctual_evaluation,
    is_safe_evaluation,
    is_successful_evaluation,
)
from .training import (
    CurriculumMetadata,
    RewardConfigSnapshot,
    RunMetadata,
    TrainingBudget,
)

__all__ = [
    "ContractError",
    "CurriculumMetadata",
    "ABLATION_MANIFEST_ARTIFACT_TYPE",
    "ABLATION_MANIFEST_SCHEMA_VERSION",
    "AblationManifest",
    "AblationRunRecord",
    "ArtifactRefs",
    "EVALUATION_HISTORY_ARTIFACT_TYPE",
    "EVALUATION_HISTORY_SCHEMA_VERSION",
    "EVALUATION_METRICS_ARTIFACT_TYPE",
    "EVALUATION_METRICS_SCHEMA_VERSION",
    "SAFETY_MARGIN_EPS_MPS",
    "EpisodeInfo",
    "EpisodeOutcome",
    "EvaluationArtifact",
    "EvaluationHistory",
    "EvaluationMetrics",
    "JSONMapping",
    "JSONValue",
    "ManifestStatusUpdate",
    "RewardConfigSnapshot",
    "RunMetadata",
    "TrainingBudget",
    "TrajectoryData",
    "is_feasible_evaluation",
    "is_precise_evaluation",
    "is_punctual_evaluation",
    "is_safe_evaluation",
    "is_successful_evaluation",
]
