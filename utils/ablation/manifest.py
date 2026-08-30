"""Versioned, atomic matrix manifests for ablation experiments."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from contracts.ablation import (
    ABLATION_MANIFEST_SCHEMA_VERSION,
    AblationManifest,
    AblationRunRecord,
    ManifestStatusUpdate,
)
from contracts.common import ContractError, JSONMapping, JSONValue


class ManifestSchemaError(ContractError):
    """Raised when a matrix manifest is missing or violates the new schema."""


class ManifestStore:
    """Read and atomically write one manifest per ablation matrix."""

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        *,
        matrix_id: str,
        filename: str = "manifest.json",
        schema_version: int = 1,
    ) -> None:
        self.output_root = Path(output_root)
        self.matrix_id = matrix_id
        self.filename = filename
        self.schema_version = schema_version

    @property
    def path(self) -> Path:
        return self.output_root / self.filename

    def exists(self) -> bool:
        return self.path.is_file()

    def archive_existing(self) -> Path:
        """Move the current manifest aside without overwriting a backup."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.path}")

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive_path = self.path.with_name(f"{self.path.name}.bak.{timestamp}")
        suffix = 1
        while archive_path.exists():
            archive_path = self.path.with_name(
                f"{self.path.name}.bak.{timestamp}.{suffix}"
            )
            suffix += 1
        os.replace(self.path, archive_path)
        return archive_path

    def load(self) -> AblationManifest:
        if not self.path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.path}")
        try:
            with self.path.open(encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        except json.JSONDecodeError as exc:
            raise ManifestSchemaError(
                f"Manifest is not valid JSON: {self.path}"
            ) from exc
        try:
            manifest = AblationManifest.from_mapping(payload)
        except ContractError as exc:
            raise ManifestSchemaError(f"Invalid manifest {self.path}: {exc}") from exc
        self.validate_shape(manifest)
        return manifest

    def validate_shape(self, payload: AblationManifest | Mapping[str, object]) -> None:
        manifest = (
            payload
            if isinstance(payload, AblationManifest)
            else AblationManifest.from_mapping(payload)
        )
        if self.schema_version != ABLATION_MANIFEST_SCHEMA_VERSION:
            raise ManifestSchemaError(
                f"Unsupported configured manifest schema: {self.schema_version}"
            )
        if manifest.SCHEMA_VERSION != self.schema_version:
            raise ManifestSchemaError(
                f"Unsupported {self.matrix_id} manifest schema: "
                f"{manifest.SCHEMA_VERSION!r}; expected {self.schema_version}"
            )
        if manifest.matrix_id != self.matrix_id:
            raise ManifestSchemaError(
                f"Manifest matrix_id does not match {self.matrix_id!r}"
            )
        if not manifest.matrix_config:
            raise ManifestSchemaError("Manifest matrix_config must not be empty")
        if not isinstance(manifest.training_signature, Mapping):
            raise ManifestSchemaError("Manifest training_signature must be an object")
        if not manifest.runs:
            raise ManifestSchemaError("Manifest runs must not be empty")

    def validate_compatibility(
        self,
        payload: AblationManifest,
        *,
        matrix_config: Mapping[str, JSONValue],
        training_signature: Mapping[str, JSONValue],
    ) -> None:
        self.validate_shape(payload)
        if payload.matrix_config != dict(matrix_config):
            raise ValueError(
                f"Existing {self.matrix_id} manifest uses a different matrix"
            )
        if payload.training_signature != dict(training_signature):
            raise ValueError(
                f"Existing {self.matrix_id} manifest uses different training settings"
            )

    def save_atomic(self, payload: AblationManifest | Mapping[str, object]) -> None:
        if not isinstance(payload, AblationManifest):
            try:
                payload = AblationManifest.from_mapping(payload)
            except ContractError as exc:
                raise ManifestSchemaError(f"Invalid manifest payload: {exc}") from exc
        self.validate_shape(payload)
        self.output_root.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.filename}.",
            suffix=".tmp",
            dir=self.output_root,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_obj:
                json.dump(payload.to_mapping(), file_obj, ensure_ascii=False, indent=2)
                file_obj.write("\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def build_manifest_payload(
    *,
    matrix_id: str,
    matrix_config: Mapping[str, JSONValue],
    training_signature: Mapping[str, JSONValue],
    runs: Sequence[AblationRunRecord | Mapping[str, object]],
    schema_version: int = ABLATION_MANIFEST_SCHEMA_VERSION,
) -> AblationManifest:
    if schema_version != ABLATION_MANIFEST_SCHEMA_VERSION:
        raise ManifestSchemaError(
            f"Unsupported manifest schema_version: {schema_version}; "
            f"expected {ABLATION_MANIFEST_SCHEMA_VERSION}"
        )
    records = tuple(
        run
        if isinstance(run, AblationRunRecord)
        else AblationRunRecord.from_mapping(
            _normalize_builder_run(run), context=f"manifest.runs[{index}]"
        )
        for index, run in enumerate(runs)
    )
    return AblationManifest(
        matrix_id=matrix_id,
        matrix_config=dict(matrix_config),
        training_signature=dict(training_signature),
        runs=records,
    )


def status_map(
    payload: AblationManifest,
) -> dict[str, ManifestStatusUpdate]:
    """Return typed status updates keyed by stable run id."""
    return {
        run.run_id: ManifestStatusUpdate(
            status=run.status,
            error_message=run.error_message,
            training_budget=run.training_budget,
        )
        for run in payload.runs
    }


def manifest_runs(
    payload: AblationManifest | Mapping[str, object],
) -> tuple[AblationRunRecord, ...]:
    """Normalize a typed manifest or a synthetic in-memory fixture at a boundary."""
    if isinstance(payload, AblationManifest):
        return payload.runs
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ManifestSchemaError("Manifest runs must be a sequence")
    return tuple(
        item
        if isinstance(item, AblationRunRecord)
        else AblationRunRecord.from_mapping(
            _normalize_builder_run(item), context=f"manifest.runs[{index}]"
        )
        for index, item in enumerate(raw_runs)
    )


def manifest_matrix_config(
    payload: AblationManifest | Mapping[str, object],
) -> JSONMapping:
    if isinstance(payload, AblationManifest):
        return payload.matrix_config
    raw_config = payload.get("matrix_config", {})
    if not isinstance(raw_config, Mapping):
        raise ManifestSchemaError("Manifest matrix_config must be an object")
    return dict(raw_config)


def _normalize_builder_run(run: object) -> object:
    """Keep the builder tolerant of the runner's generic synthetic result key.

    Persisted manifests remain strict: only the explicit ``extensions`` field
    can carry non-standard metadata.  This conversion exists solely for the
    construction helper used by small generic runner tests.
    """
    if not isinstance(run, Mapping):
        return run
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping) or "result" not in artifacts:
        return run
    normalized = dict(run)
    normalized_artifacts = dict(artifacts)
    result = normalized_artifacts.pop("result")
    extensions = normalized_artifacts.get("extensions", {})
    if not isinstance(extensions, Mapping):
        return run
    normalized_artifacts["extensions"] = {
        **dict(extensions),
        "result": result,
    }
    normalized["artifacts"] = normalized_artifacts
    return normalized
