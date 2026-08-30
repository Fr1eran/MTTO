"""Versioned contracts for ablation matrices and their run artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import ClassVar, Literal

from .common import (
    ContractError,
    JSONMapping,
    MappingView,
    as_json_value,
    from_dict,
    require_object,
    to_dict,
)
from .training import TrainingBudget

AblationStatus = Literal["pending", "running", "completed", "failed"]
ABLATION_MANIFEST_ARTIFACT_TYPE = "ablation_manifest"
ABLATION_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArtifactRefs(MappingView):
    """Named paths referenced by one ablation run."""

    policy_final: str | None = field(default=None, metadata={"non_empty": True})
    metadata: str | None = field(default=None, metadata={"non_empty": True})
    episodes: str | None = field(default=None, metadata={"non_empty": True})
    evaluations: str | None = field(default=None, metadata={"non_empty": True})
    trajectory_final: str | None = field(default=None, metadata={"non_empty": True})
    trajectory_best: str | None = field(default=None, metadata={"non_empty": True})
    metrics_final: str | None = field(default=None, metadata={"non_empty": True})
    metrics_best: str | None = field(default=None, metadata={"non_empty": True})
    safety_diagnostics: str | None = field(default=None, metadata={"non_empty": True})
    extensions: JSONMapping = field(default_factory=dict)

    def to_mapping(self) -> JSONMapping:
        return to_dict(
            self,
            compact=True,
            omit_empty=frozenset({"extensions"}),
        )

    @classmethod
    def from_mapping(
        cls, payload: object, *, context: str = "artifacts"
    ) -> ArtifactRefs:
        return from_dict(cls, payload, context=context)

    def path_for(self, name: str) -> str:
        names = {item.name for item in fields(self)} - {"extensions"}
        if name not in names:
            raise KeyError(f"Unknown artifact reference {name!r}")
        value = getattr(self, name)
        if value is None:
            raise KeyError(f"Artifact reference {name!r} is not present")
        return value


@dataclass(frozen=True, slots=True)
class ManifestStatusUpdate(MappingView):
    status: AblationStatus
    error_message: str | None = None
    training_budget: TrainingBudget | None = None

    def to_mapping(self) -> JSONMapping:
        return to_dict(self, compact=True, late_optional=True)

    @classmethod
    def from_mapping(cls, payload: object, *, context: str) -> ManifestStatusUpdate:
        return from_dict(cls, payload, context=context, ignore_unknown=True)


@dataclass(frozen=True, slots=True)
class AblationRunRecord(MappingView):
    """One persisted matrix run, independent of its training implementation."""

    run_id: str = field(metadata={"non_empty": True})
    variant_id: str = field(metadata={"non_empty": True})
    variant: JSONMapping
    repeat_index: int = field(metadata={"minimum": 0})
    seed: int
    artifacts: ArtifactRefs
    status: AblationStatus
    experiment_tag: str | None = None
    error_message: str | None = None
    training_budget: TrainingBudget | None = None
    extensions: JSONMapping = field(default_factory=dict)

    def to_mapping(self) -> JSONMapping:
        payload = to_dict(self, compact=True, late_optional=True)
        payload["artifacts"] = self.artifacts.to_mapping()
        return payload

    @classmethod
    def from_mapping(
        cls, payload: object, *, context: str = "manifest.run"
    ) -> AblationRunRecord:
        data = dict(require_object(payload, context=context))
        variant = dict(
            require_object(data.get("variant"), context=f"{context}.variant")
        )
        if "step_distance" in data and "step_distance" not in variant:
            variant["step_distance"] = as_json_value(
                data["step_distance"], field=f"{context}.step_distance"
            )
        data.pop("step_distance", None)
        data["variant"] = variant
        return from_dict(cls, data, context=context)

    def with_status(self, update: ManifestStatusUpdate) -> AblationRunRecord:
        return replace(
            self,
            status=update.status,
            error_message=update.error_message,
            training_budget=update.training_budget,
        )


@dataclass(frozen=True, slots=True)
class AblationManifest(MappingView):
    """Typed matrix manifest used by all ablation workflows."""

    matrix_id: str = field(metadata={"non_empty": True})
    matrix_config: JSONMapping
    training_signature: JSONMapping
    runs: tuple[AblationRunRecord, ...]
    output_root: str | None = None
    extensions: JSONMapping = field(default_factory=dict)

    ARTIFACT_TYPE: ClassVar[str] = ABLATION_MANIFEST_ARTIFACT_TYPE
    SCHEMA_VERSION: ClassVar[int] = ABLATION_MANIFEST_SCHEMA_VERSION

    def to_mapping(self) -> JSONMapping:
        payload = to_dict(
            self,
            headers={
                "artifact_type": self.ARTIFACT_TYPE,
                "schema_version": self.SCHEMA_VERSION,
            },
            compact=True,
            late_optional=True,
        )
        payload["runs"] = [run.to_mapping() for run in self.runs]
        return payload

    def with_output_root(self, output_root: str) -> AblationManifest:
        return replace(self, output_root=output_root)

    def with_statuses(
        self, statuses: dict[str, ManifestStatusUpdate]
    ) -> AblationManifest:
        return replace(
            self,
            runs=tuple(
                run.with_status(statuses[run.run_id]) if run.run_id in statuses else run
                for run in self.runs
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: object,
        *,
        require_header: bool = True,
        context: str = "ablation_manifest",
    ) -> AblationManifest:
        manifest = from_dict(
            cls,
            payload,
            context=context,
            headers=(
                {
                    "artifact_type": cls.ARTIFACT_TYPE,
                    "schema_version": cls.SCHEMA_VERSION,
                }
                if require_header
                else None
            ),
        )
        run_ids = [run.run_id for run in manifest.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ContractError(f"{context}.runs contains duplicate run_id values")
        return manifest


def build_manifest(
    *,
    matrix_id: str,
    matrix_config: JSONMapping,
    training_signature: JSONMapping,
    runs: tuple[AblationRunRecord, ...],
    output_root: str | None = None,
) -> AblationManifest:
    return AblationManifest(
        matrix_id=matrix_id,
        matrix_config=matrix_config,
        training_signature=training_signature,
        runs=runs,
        output_root=output_root,
    )


def status_map(
    manifest: AblationManifest,
) -> dict[str, ManifestStatusUpdate]:
    return {
        run.run_id: ManifestStatusUpdate(
            status=run.status,
            error_message=run.error_message,
            training_budget=run.training_budget,
        )
        for run in manifest.runs
    }
