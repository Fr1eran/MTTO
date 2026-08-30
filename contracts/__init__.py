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
    EvaluationArtifact,
    EvaluationHistory,
    EvaluationMetrics,
    TrajectoryData,
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
]
