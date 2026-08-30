"""Typed training metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal

from .common import JSONMapping, MappingView, from_dict, to_dict

CurriculumProfileName = Literal["none", "dspdl", "dspdl_completion"]


@dataclass(frozen=True, slots=True)
class RewardConfigSnapshot(MappingView):
    energy_reward_scale: float
    comfort_reward_scale: float
    enable_potential_safety: bool
    survival_reward_scale: float

    def to_mapping(self) -> JSONMapping:
        return to_dict(self)


@dataclass(frozen=True, slots=True)
class CurriculumMetadata(MappingView):
    profile_name: CurriculumProfileName
    enabled: bool
    value_source: str | None
    dspdl_config: JSONMapping | None
    reference_curve_dir: str | None
    reference_curve_artifact_path: str | None
    reference_curve_metrics_path: str | None
    rl_step_distance_m: float | None
    context_count: int | None = field(metadata={"minimum": 0})
    initial_curriculum_version: int | None = field(metadata={"minimum": 0})
    completion_critic: JSONMapping | None

    def to_mapping(self) -> JSONMapping:
        return to_dict(self)


@dataclass(frozen=True, slots=True)
class TrainingBudget(MappingView):
    mode: Literal["completed_episodes"]
    training_episodes: int | None = field(metadata={"minimum": 0})
    effective_training_episodes: int | None = field(metadata={"minimum": 0})
    max_episode_steps: int | None = field(metadata={"minimum": 0})
    derived_total_timesteps: int = field(metadata={"minimum": 0})
    actual_completed_episodes: int | None = field(default=None, metadata={"minimum": 0})
    actual_training_timesteps: int | None = field(default=None, metadata={"minimum": 0})
    target_reached: bool | None = None
    stop_reason: str | None = None

    def to_mapping(self) -> JSONMapping:
        return to_dict(self)

    @classmethod
    def from_mapping(
        cls, payload: object, *, context: str = "training_budget"
    ) -> TrainingBudget:
        return from_dict(cls, payload, context=context)


@dataclass(frozen=True, slots=True)
class RunMetadata(MappingView):
    """Complete, versioned metadata snapshot for one training run."""

    reward_preset_name: str = field(metadata={"non_empty": True})
    reward_preset_label: str
    reward_preset_description: str
    potential_shaping_components: tuple[str, ...]
    reward_config: RewardConfigSnapshot
    curriculum: CurriculumMetadata
    schedule_time_s: float
    step_distance: float
    reward_discount: float
    experiment_token: str = field(metadata={"non_empty": True})
    training_budget: TrainingBudget | None = None
    experiment_tag: str | None = None
    run_mode: str | None = None
    enable_tb: bool | None = None
    enable_monitor: bool | None = None
    enable_auto_analysis: bool | None = None
    enable_best_evaluation_artifacts: bool | None = None
    enable_safety_truncation_histogram: bool | None = None
    safety_truncation_bin_size_m: float | None = None
    evaluation_interval_rollouts: int | None = None
    evaluation_deterministic: bool | None = None
    evaluation_history_path: str | None = None
    num_envs: int | None = None
    n_steps_per_env: int | None = None
    rollout_steps_per_update: int | None = None
    output_dir: str | None = None
    final_output_dir: str | None = None
    reward_diagnostics_path: str | None = None
    best_eval_output_dir: str | None = None
    tensorboard_log_dir: str | None = None
    tb_log_name: str | None = None
    reward_diagnostics_schema_version: int | None = None
    extensions: JSONMapping = field(default_factory=dict)

    ARTIFACT_TYPE: ClassVar[str] = "rl_training_metadata"
    SCHEMA_VERSION: ClassVar[int] = 1

    def to_mapping(self) -> JSONMapping:
        return to_dict(
            self,
            headers={
                "artifact_type": self.ARTIFACT_TYPE,
                "schema_version": self.SCHEMA_VERSION,
            },
            compact=True,
            late_optional=True,
        )

    def with_updates(self, **changes: object) -> RunMetadata:
        return replace(self, **changes)

    @classmethod
    def from_mapping(cls, payload: object) -> RunMetadata:
        return from_dict(
            cls,
            payload,
            context="run_metadata",
            headers={
                "artifact_type": cls.ARTIFACT_TYPE,
                "schema_version": cls.SCHEMA_VERSION,
            },
        )
