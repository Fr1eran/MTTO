"""Artifact storage definitions, strict validation, and I/O routines."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from mtto.domain.safeguard import ViolationKind
from mtto.domain.scenario import ScheduleChange, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import (
    AuditViolation,
    QualityMetrics,
    QualityReport,
    SafetyAudit,
    SpsEvent,
    SpsEventKind,
    ViolationCategory,
)
from mtto.rl.diagnostics import (
    REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    REWARD_NAMES,
    TOTAL_REWARD_INDEX,
    EvaluationHistory,
    RewardDiagnostics,
    SafetyTruncationHistogram,
    TrainingDiagnostics,
)
from mtto.rl.state import TerminationReason

__all__ = [
    "RUN_JSON",
    "PROFILE_NPZ",
    "QUALITY_JSON",
    "RESULT_JSON",
    "POLICY_ZIP",
    "DIAGNOSTICS_NPZ",
    "EVALUATIONS_NPZ",
    "BEST_DIR",
    "REQUIRED_FILES",
    "EVALUATION_POLICY_RESULT_FILE",
    "BEST_REQUIRED_FILES",
    "required_files_for",
    "RunKind",
    "RunRecord",
    "TrainingOutcome",
    "RLResult",
    "TrainingDiagnostics",
    "RunPayload",
    "CompletedRun",
    "ArtifactError",
    "canonical_json",
    "file_sha256",
    "task_to_json",
    "task_from_json",
    "write_profile",
    "read_profile",
    "write_quality",
    "read_quality",
    "write_result",
    "read_result",
    "write_diagnostics",
    "read_diagnostics",
    "write_analysis_report",
    "write_evaluations",
    "read_evaluations",
    "write_run_record",
    "read_run_record",
    "validate_payload",
    "write_run",
    "read_completed_run",
]

RUN_JSON = "run.json"
PROFILE_NPZ = "profile.npz"
QUALITY_JSON = "quality.json"
RESULT_JSON = "result.json"
POLICY_ZIP = "policy.zip"
DIAGNOSTICS_NPZ = "diagnostics.npz"
EVALUATIONS_NPZ = "evaluations.npz"
BEST_DIR = "best"

_PROFILE_KEYS = frozenset(
    {
        "position_m",
        "speed_mps",
        "time_s",
        "segment_acceleration_mps2",
        "propulsion_energy_kj",
        "levitation_energy_kj",
    }
)

_REWARD_KEYS = frozenset(f.name for f in dataclasses.fields(RewardDiagnostics))
_SAFETY_KEYS = frozenset(
    f"safety_{f.name}" for f in dataclasses.fields(SafetyTruncationHistogram)
)
_DIAGNOSTICS_KEYS = _REWARD_KEYS | _SAFETY_KEYS

_EVALUATIONS_KEYS = frozenset(f.name for f in dataclasses.fields(EvaluationHistory))


_TASK_KEYS = frozenset(
    {
        "start_position_m",
        "target_position_m",
        "schedule_time_s",
        "max_acc_change",
        "max_stop_error_m",
        "max_arr_time_error_s",
        "schedule_change",
    }
)

_SCHEDULE_CHANGE_KEYS = frozenset(
    {
        "trigger_position_m",
        "new_schedule_time_s",
    }
)

_QUALITY_KEYS = frozenset(
    {
        "metrics",
        "audit",
        "completed",
        "precise_stop",
        "safe",
        "punctual",
        "feasible",
    }
)

_METRICS_KEYS = frozenset(
    {
        "propulsion_energy_kj",
        "levitation_energy_kj",
        "run_time_s",
        "stop_error_m",
        "comfort_tav_mps2",
        "comfort_rms_mps2",
        "comfort_exceedance_pct",
        "arrival_time_error_s",
    }
)

_AUDIT_KEYS = frozenset(
    {
        "target_stopping_point",
        "request_pending",
        "lower_limit_mps",
        "upper_limit_mps",
        "violations",
        "events",
        "min_margin_mps",
    }
)

_VIOLATION_KEYS = frozenset(
    {
        "node_index",
        "position_m",
        "speed_mps",
        "kind",
        "margin_mps",
        "category",
    }
)

_EVENT_KEYS = frozenset(
    {
        "node_index",
        "kind",
        "time_s",
        "position_m",
        "target_stopping_point",
    }
)

_RESULT_KEYS = frozenset(
    {
        "termination_reason",
        "truncated",
        "total_reward",
        "steps",
        "final_position_m",
        "final_speed_mps",
        "final_time_s",
        "deterministic",
        "policy_run_id",
        "policy_sha256",
        "training",
    }
)

_TRAINING_OUTCOME_KEYS = frozenset(
    {
        "actual_training_timesteps",
        "actual_training_rollouts",
        "actual_completed_episodes",
        "target_reached",
        "stop_reason",
    }
)

_RUN_RECORD_KEYS = frozenset(
    {
        "run_id",
        "kind",
        "config",
        "scenario_hash",
        "task",
        "policy_io_version",
        "mtto_version",
        "created_at",
    }
)


class ArtifactError(ValueError):
    """Raised when an artifact fails strict validation or I/O constraints."""


class RunKind(StrEnum):
    RL_TRAIN = "rl_train"
    DP_SOLVE = "dp_solve"
    EVALUATION = "evaluation"


REQUIRED_FILES: dict[RunKind, tuple[str, ...]] = {
    RunKind.DP_SOLVE: (PROFILE_NPZ, QUALITY_JSON),
    RunKind.RL_TRAIN: (PROFILE_NPZ, QUALITY_JSON, RESULT_JSON, POLICY_ZIP),
    RunKind.EVALUATION: (PROFILE_NPZ, QUALITY_JSON),
}
EVALUATION_POLICY_RESULT_FILE: str = RESULT_JSON
BEST_REQUIRED_FILES: tuple[str, ...] = (
    PROFILE_NPZ,
    QUALITY_JSON,
    RESULT_JSON,
    POLICY_ZIP,
)


def required_files_for(
    kind: RunKind, *, policy_io_version: int | None
) -> tuple[str, ...]:
    """Return required artifact filenames for the given run configuration."""
    files = REQUIRED_FILES[kind]
    if kind == RunKind.EVALUATION and policy_io_version is not None:
        return (*files, EVALUATION_POLICY_RESULT_FILE)
    return files


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    kind: RunKind
    config: dict[str, Any]
    scenario_hash: str
    task: dict[str, Any]
    policy_io_version: int | None
    mtto_version: str
    created_at: str


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    actual_training_timesteps: int
    actual_training_rollouts: int | None
    actual_completed_episodes: int
    target_reached: bool
    stop_reason: str


@dataclass(frozen=True, slots=True)
class RLResult:
    termination_reason: TerminationReason | None
    truncated: bool
    total_reward: float
    steps: int
    final_position_m: float
    final_speed_mps: float
    final_time_s: float
    deterministic: bool
    policy_run_id: str
    policy_sha256: str
    training: TrainingOutcome | None


@dataclass(frozen=True, slots=True)
class RunPayload:
    profile: SpeedProfile
    quality: QualityReport
    result: RLResult | None = None
    best: RunPayload | None = None
    diagnostics: TrainingDiagnostics | None = None
    evaluations: EvaluationHistory | None = None


@dataclass(frozen=True, slots=True)
class CompletedRun:
    record: RunRecord
    payload: RunPayload


def canonical_json(value: Any) -> str:
    """Encode value as deterministic JSON."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file by streaming chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_dict(payload: object, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ArtifactError(f"{context} must be a dict, got {type(payload).__name__}")
    return payload


def _require_keys(
    payload: dict[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(payload.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        msg_parts = []
        if missing:
            msg_parts.append(f"missing keys: {missing}")
        if extra:
            msg_parts.append(f"extra keys: {extra}")
        raise ArtifactError(f"{context} key mismatch: {', '.join(msg_parts)}")


def _require_finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{field} must be a number, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val):
        raise ArtifactError(f"{field} must be finite, got {val}")
    return val


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{field} must be an integer, got {type(value).__name__}")
    return int(value)


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactError(f"{field} must be a boolean, got {type(value).__name__}")
    return bool(value)


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ArtifactError(f"{field} must be a string, got {type(value).__name__}")
    return value


def task_to_json(task: Task) -> dict[str, Any]:
    """Serialize Task to a plain JSON-compatible dictionary."""
    return dataclasses.asdict(task)


def task_from_json(payload: dict[str, Any]) -> Task:
    """Construct Task from a dictionary with strict validation."""
    data = _require_dict(payload, "Task")
    _require_keys(data, _TASK_KEYS, "Task")

    schedule_change_raw = data["schedule_change"]
    schedule_change: ScheduleChange | None = None
    if schedule_change_raw is not None:
        sc_data = _require_dict(schedule_change_raw, "Task.schedule_change")
        _require_keys(sc_data, _SCHEDULE_CHANGE_KEYS, "Task.schedule_change")
        trigger_pos = _require_finite_float(
            sc_data["trigger_position_m"], "Task.schedule_change.trigger_position_m"
        )
        new_time = _require_finite_float(
            sc_data["new_schedule_time_s"], "Task.schedule_change.new_schedule_time_s"
        )
        try:
            schedule_change = ScheduleChange(
                trigger_position_m=trigger_pos,
                new_schedule_time_s=new_time,
            )
        except (ValueError, TypeError) as exc:
            raise ArtifactError(f"Invalid ScheduleChange: {exc}") from exc

    schedule_time_raw = data["schedule_time_s"]
    schedule_time = (
        _require_finite_float(schedule_time_raw, "Task.schedule_time_s")
        if schedule_time_raw is not None
        else None
    )

    try:
        return Task(
            start_position_m=_require_finite_float(
                data["start_position_m"], "Task.start_position_m"
            ),
            target_position_m=_require_finite_float(
                data["target_position_m"], "Task.target_position_m"
            ),
            schedule_time_s=schedule_time,
            max_acc_change=_require_finite_float(
                data["max_acc_change"], "Task.max_acc_change"
            ),
            max_stop_error_m=_require_finite_float(
                data["max_stop_error_m"], "Task.max_stop_error_m"
            ),
            max_arr_time_error_s=_require_finite_float(
                data["max_arr_time_error_s"], "Task.max_arr_time_error_s"
            ),
            schedule_change=schedule_change,
        )
    except (ValueError, TypeError) as exc:
        raise ArtifactError(f"Invalid Task: {exc}") from exc


def write_profile(path: str | Path, profile: SpeedProfile) -> None:
    """Save SpeedProfile to a .npz archive strictly."""
    np.savez(
        path,
        position_m=profile.position_m,
        speed_mps=profile.speed_mps,
        time_s=profile.time_s,
        segment_acceleration_mps2=profile.segment_acceleration_mps2,
        propulsion_energy_kj=profile.propulsion_energy_kj,
        levitation_energy_kj=profile.levitation_energy_kj,
    )


def read_profile(path: str | Path) -> SpeedProfile:
    """Strictly load SpeedProfile from .npz archive."""
    try:
        with np.load(path, allow_pickle=False) as data:
            actual_keys = frozenset(data.files)
            if actual_keys != _PROFILE_KEYS:
                missing = sorted(_PROFILE_KEYS - actual_keys)
                extra = sorted(actual_keys - _PROFILE_KEYS)
                msg_parts = []
                if missing:
                    msg_parts.append(f"missing keys: {missing}")
                if extra:
                    msg_parts.append(f"extra keys: {extra}")
                raise ArtifactError(
                    f"{path} invalid SpeedProfile archive: {', '.join(msg_parts)}"
                )

            position_m = np.asarray(data["position_m"], dtype=np.float64)
            speed_mps = np.asarray(data["speed_mps"], dtype=np.float64)
            time_s = np.asarray(data["time_s"], dtype=np.float64)
            segment_acceleration_mps2 = np.asarray(
                data["segment_acceleration_mps2"], dtype=np.float64
            )
            propulsion_energy_kj = np.asarray(
                data["propulsion_energy_kj"], dtype=np.float64
            )
            levitation_energy_kj = np.asarray(
                data["levitation_energy_kj"], dtype=np.float64
            )

            try:
                return SpeedProfile(
                    position_m=position_m,
                    speed_mps=speed_mps,
                    time_s=time_s,
                    segment_acceleration_mps2=segment_acceleration_mps2,
                    propulsion_energy_kj=propulsion_energy_kj,
                    levitation_energy_kj=levitation_energy_kj,
                )
            except ValueError as exc:
                raise ArtifactError(f"{path}: invalid SpeedProfile: {exc}") from exc
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Failed to read SpeedProfile from {path}: {exc}") from exc


def write_quality(path: str | Path, report: QualityReport) -> None:
    """Save QualityReport to JSON with deterministic sorting and formatting."""
    payload = {
        "metrics": {
            "propulsion_energy_kj": float(report.metrics.propulsion_energy_kj),
            "levitation_energy_kj": float(report.metrics.levitation_energy_kj),
            "run_time_s": float(report.metrics.run_time_s),
            "stop_error_m": float(report.metrics.stop_error_m),
            "comfort_tav_mps2": float(report.metrics.comfort_tav_mps2),
            "comfort_rms_mps2": float(report.metrics.comfort_rms_mps2),
            "comfort_exceedance_pct": float(report.metrics.comfort_exceedance_pct),
            "arrival_time_error_s": (
                float(report.metrics.arrival_time_error_s)
                if report.metrics.arrival_time_error_s is not None
                else None
            ),
        },
        "audit": {
            "target_stopping_point": report.audit.target_stopping_point.tolist(),
            "request_pending": report.audit.request_pending.tolist(),
            "lower_limit_mps": report.audit.lower_limit_mps.tolist(),
            "upper_limit_mps": report.audit.upper_limit_mps.tolist(),
            "violations": [
                {
                    "node_index": int(v.node_index),
                    "position_m": float(v.position_m),
                    "speed_mps": float(v.speed_mps),
                    "kind": v.kind.value,
                    "margin_mps": float(v.margin_mps),
                    "category": v.category.value,
                }
                for v in report.audit.violations
            ],
            "events": [
                {
                    "node_index": int(e.node_index),
                    "kind": e.kind.value,
                    "time_s": float(e.time_s),
                    "position_m": float(e.position_m),
                    "target_stopping_point": int(e.target_stopping_point),
                }
                for e in report.audit.events
            ],
            "min_margin_mps": float(report.audit.min_margin_mps),
        },
        "completed": bool(report.completed),
        "precise_stop": bool(report.precise_stop),
        "safe": bool(report.safe),
        "punctual": bool(report.punctual) if report.punctual is not None else None,
        "feasible": bool(report.feasible),
    }
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    Path(path).write_text(content, encoding="utf-8")


def read_quality(path: str | Path) -> QualityReport:
    """Strictly load QualityReport from JSON."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        raw = json.loads(text)
    except Exception as exc:
        raise ArtifactError(f"Failed to read JSON from {path}: {exc}") from exc

    data = _require_dict(raw, f"{path}")
    _require_keys(data, _QUALITY_KEYS, f"{path}")

    # Metrics
    m_data = _require_dict(data["metrics"], f"{path}.metrics")
    _require_keys(m_data, _METRICS_KEYS, f"{path}.metrics")
    arrival_err_raw = m_data["arrival_time_error_s"]
    arrival_err = (
        _require_finite_float(arrival_err_raw, f"{path}.metrics.arrival_time_error_s")
        if arrival_err_raw is not None
        else None
    )
    metrics = QualityMetrics(
        propulsion_energy_kj=_require_finite_float(
            m_data["propulsion_energy_kj"], f"{path}.metrics.propulsion_energy_kj"
        ),
        levitation_energy_kj=_require_finite_float(
            m_data["levitation_energy_kj"], f"{path}.metrics.levitation_energy_kj"
        ),
        run_time_s=_require_finite_float(
            m_data["run_time_s"], f"{path}.metrics.run_time_s"
        ),
        stop_error_m=_require_finite_float(
            m_data["stop_error_m"], f"{path}.metrics.stop_error_m"
        ),
        comfort_tav_mps2=_require_finite_float(
            m_data["comfort_tav_mps2"], f"{path}.metrics.comfort_tav_mps2"
        ),
        comfort_rms_mps2=_require_finite_float(
            m_data["comfort_rms_mps2"], f"{path}.metrics.comfort_rms_mps2"
        ),
        comfort_exceedance_pct=_require_finite_float(
            m_data["comfort_exceedance_pct"], f"{path}.metrics.comfort_exceedance_pct"
        ),
        arrival_time_error_s=arrival_err,
    )

    # Audit
    a_data = _require_dict(data["audit"], f"{path}.audit")
    _require_keys(a_data, _AUDIT_KEYS, f"{path}.audit")

    target_sp_raw = a_data["target_stopping_point"]
    if not isinstance(target_sp_raw, list):
        raise ArtifactError(f"{path}.audit.target_stopping_point must be a list")
    target_sp = np.array(
        [
            _require_int(x, f"{path}.audit.target_stopping_point[]")
            for x in target_sp_raw
        ],
        dtype=np.int64,
    )

    req_pending_raw = a_data["request_pending"]
    if not isinstance(req_pending_raw, list):
        raise ArtifactError(f"{path}.audit.request_pending must be a list")
    req_pending = np.array(
        [_require_bool(x, f"{path}.audit.request_pending[]") for x in req_pending_raw],
        dtype=np.bool_,
    )

    lower_raw = a_data["lower_limit_mps"]
    if not isinstance(lower_raw, list):
        raise ArtifactError(f"{path}.audit.lower_limit_mps must be a list")
    lower_limits = np.array(
        [
            _require_finite_float(x, f"{path}.audit.lower_limit_mps[]")
            for x in lower_raw
        ],
        dtype=np.float64,
    )

    upper_raw = a_data["upper_limit_mps"]
    if not isinstance(upper_raw, list):
        raise ArtifactError(f"{path}.audit.upper_limit_mps must be a list")
    upper_limits = np.array(
        [
            _require_finite_float(x, f"{path}.audit.upper_limit_mps[]")
            for x in upper_raw
        ],
        dtype=np.float64,
    )

    violations_raw = a_data["violations"]
    if not isinstance(violations_raw, list):
        raise ArtifactError(f"{path}.audit.violations must be a list")
    violations: list[AuditViolation] = []
    for idx, v_item in enumerate(violations_raw):
        v_dict = _require_dict(v_item, f"{path}.audit.violations[{idx}]")
        _require_keys(v_dict, _VIOLATION_KEYS, f"{path}.audit.violations[{idx}]")
        kind_str = _require_str(v_dict["kind"], f"{path}.audit.violations[{idx}].kind")
        try:
            kind_enum = ViolationKind(kind_str)
        except ValueError as exc:
            raise ArtifactError(f"Invalid ViolationKind '{kind_str}'") from exc

        cat_str = _require_str(
            v_dict["category"], f"{path}.audit.violations[{idx}].category"
        )
        try:
            cat_enum = ViolationCategory(cat_str)
        except ValueError as exc:
            raise ArtifactError(f"Invalid ViolationCategory '{cat_str}'") from exc

        violations.append(
            AuditViolation(
                node_index=_require_int(
                    v_dict["node_index"], f"{path}.audit.violations[{idx}].node_index"
                ),
                position_m=_require_finite_float(
                    v_dict["position_m"], f"{path}.audit.violations[{idx}].position_m"
                ),
                speed_mps=_require_finite_float(
                    v_dict["speed_mps"], f"{path}.audit.violations[{idx}].speed_mps"
                ),
                kind=kind_enum,
                margin_mps=_require_finite_float(
                    v_dict["margin_mps"], f"{path}.audit.violations[{idx}].margin_mps"
                ),
                category=cat_enum,
            )
        )

    events_raw = a_data["events"]
    if not isinstance(events_raw, list):
        raise ArtifactError(f"{path}.audit.events must be a list")
    events: list[SpsEvent] = []
    for idx, e_item in enumerate(events_raw):
        e_dict = _require_dict(e_item, f"{path}.audit.events[{idx}]")
        _require_keys(e_dict, _EVENT_KEYS, f"{path}.audit.events[{idx}]")
        kind_str = _require_str(e_dict["kind"], f"{path}.audit.events[{idx}].kind")
        try:
            e_kind_enum = SpsEventKind(kind_str)
        except ValueError as exc:
            raise ArtifactError(f"Invalid SpsEventKind '{kind_str}'") from exc

        events.append(
            SpsEvent(
                node_index=_require_int(
                    e_dict["node_index"], f"{path}.audit.events[{idx}].node_index"
                ),
                kind=e_kind_enum,
                time_s=_require_finite_float(
                    e_dict["time_s"], f"{path}.audit.events[{idx}].time_s"
                ),
                position_m=_require_finite_float(
                    e_dict["position_m"], f"{path}.audit.events[{idx}].position_m"
                ),
                target_stopping_point=_require_int(
                    e_dict["target_stopping_point"],
                    f"{path}.audit.events[{idx}].target_stopping_point",
                ),
            )
        )

    min_margin = _require_finite_float(
        a_data["min_margin_mps"], f"{path}.audit.min_margin_mps"
    )

    audit = SafetyAudit(
        target_stopping_point=target_sp,
        request_pending=req_pending,
        lower_limit_mps=lower_limits,
        upper_limit_mps=upper_limits,
        violations=tuple(violations),
        events=tuple(events),
        min_margin_mps=min_margin,
    )

    punctual_raw = data["punctual"]
    punctual = (
        _require_bool(punctual_raw, f"{path}.punctual")
        if punctual_raw is not None
        else None
    )

    return QualityReport(
        metrics=metrics,
        audit=audit,
        completed=_require_bool(data["completed"], f"{path}.completed"),
        precise_stop=_require_bool(data["precise_stop"], f"{path}.precise_stop"),
        safe=_require_bool(data["safe"], f"{path}.safe"),
        punctual=punctual,
        feasible=_require_bool(data["feasible"], f"{path}.feasible"),
    )


def write_result(path: str | Path, result: RLResult) -> None:
    """Save RLResult to JSON strictly."""
    payload = {
        "termination_reason": (
            result.termination_reason.name
            if result.termination_reason is not None
            else None
        ),
        "truncated": bool(result.truncated),
        "total_reward": float(result.total_reward),
        "steps": int(result.steps),
        "final_position_m": float(result.final_position_m),
        "final_speed_mps": float(result.final_speed_mps),
        "final_time_s": float(result.final_time_s),
        "deterministic": bool(result.deterministic),
        "policy_run_id": str(result.policy_run_id),
        "policy_sha256": str(result.policy_sha256),
        "training": (
            {
                "actual_training_timesteps": int(
                    result.training.actual_training_timesteps
                ),
                "actual_training_rollouts": (
                    int(result.training.actual_training_rollouts)
                    if result.training.actual_training_rollouts is not None
                    else None
                ),
                "actual_completed_episodes": int(
                    result.training.actual_completed_episodes
                ),
                "target_reached": bool(result.training.target_reached),
                "stop_reason": str(result.training.stop_reason),
            }
            if result.training is not None
            else None
        ),
    }
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    Path(path).write_text(content, encoding="utf-8")


def read_result(path: str | Path) -> RLResult:
    """Strictly load RLResult from JSON."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        raw = json.loads(text)
    except Exception as exc:
        raise ArtifactError(f"Failed to read JSON from {path}: {exc}") from exc

    data = _require_dict(raw, f"{path}")
    _require_keys(data, _RESULT_KEYS, f"{path}")

    term_reason_raw = data["termination_reason"]
    termination_reason: TerminationReason | None = None
    if term_reason_raw is not None:
        name = _require_str(term_reason_raw, f"{path}.termination_reason")
        if name not in TerminationReason.__members__:
            raise ArtifactError(f"Unknown TerminationReason '{name}'")
        termination_reason = TerminationReason[name]

    training_raw = data["training"]
    training: TrainingOutcome | None = None
    if training_raw is not None:
        t_data = _require_dict(training_raw, f"{path}.training")
        _require_keys(t_data, _TRAINING_OUTCOME_KEYS, f"{path}.training")
        rollouts_raw = t_data["actual_training_rollouts"]
        rollouts = (
            _require_int(rollouts_raw, f"{path}.training.actual_training_rollouts")
            if rollouts_raw is not None
            else None
        )
        training = TrainingOutcome(
            actual_training_timesteps=_require_int(
                t_data["actual_training_timesteps"],
                f"{path}.training.actual_training_timesteps",
            ),
            actual_training_rollouts=rollouts,
            actual_completed_episodes=_require_int(
                t_data["actual_completed_episodes"],
                f"{path}.training.actual_completed_episodes",
            ),
            target_reached=_require_bool(
                t_data["target_reached"], f"{path}.training.target_reached"
            ),
            stop_reason=_require_str(
                t_data["stop_reason"], f"{path}.training.stop_reason"
            ),
        )

    return RLResult(
        termination_reason=termination_reason,
        truncated=_require_bool(data["truncated"], f"{path}.truncated"),
        total_reward=_require_finite_float(
            data["total_reward"], f"{path}.total_reward"
        ),
        steps=_require_int(data["steps"], f"{path}.steps"),
        final_position_m=_require_finite_float(
            data["final_position_m"], f"{path}.final_position_m"
        ),
        final_speed_mps=_require_finite_float(
            data["final_speed_mps"], f"{path}.final_speed_mps"
        ),
        final_time_s=_require_finite_float(
            data["final_time_s"], f"{path}.final_time_s"
        ),
        deterministic=_require_bool(data["deterministic"], f"{path}.deterministic"),
        policy_run_id=_require_str(data["policy_run_id"], f"{path}.policy_run_id"),
        policy_sha256=_require_str(data["policy_sha256"], f"{path}.policy_sha256"),
        training=training,
    )


def write_diagnostics(path: str | Path, diagnostics: TrainingDiagnostics) -> None:
    """Save TrainingDiagnostics to a .npz archive strictly."""
    data: dict[str, Any] = dict(diagnostics.reward.to_arrays())
    for key, val in diagnostics.safety.to_arrays().items():
        data[f"safety_{key}"] = val
    np.savez(path, **data)


def _parse_reward_diagnostics_from_mapping(
    data: Any, path: str | Path, prefix: str = ""
) -> RewardDiagnostics:
    schema_version = data[f"{prefix}schema_version"]
    if schema_version.dtype != np.int16 or schema_version.shape != (1,):
        raise ArtifactError(
            f"{prefix}schema_version must have dtype int16 and shape (1,)"
        )
    if int(schema_version[0]) != REWARD_DIAGNOSTICS_SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported reward diagnostics schema version: {schema_version}"
        )

    reward_names = data[f"{prefix}reward_names"]
    if reward_names.dtype.kind not in ("U", "S", "O") or reward_names.ndim != 1:
        raise ArtifactError(f"{prefix}reward_names must be a 1D string array")
    if tuple(str(x) for x in reward_names) != REWARD_NAMES:
        raise ArtifactError(f"{prefix}reward_names must match REWARD_NAMES")
    n_signals = reward_names.shape[0]

    rollout_end_step = data[f"{prefix}rollout_end_step"]
    if rollout_end_step.dtype != np.int64 or rollout_end_step.ndim != 1:
        raise ArtifactError("rollout_end_step must have dtype int64 and 1D shape")
    n_rollouts = rollout_end_step.shape[0]

    rollout_transition_count = data[f"{prefix}rollout_transition_count"]
    if rollout_transition_count.dtype != np.int64 or rollout_transition_count.shape != (
        n_rollouts,
    ):
        raise ArtifactError(
            "rollout_transition_count must have dtype int64 and shape (n_rollouts,)"
        )

    rollout_reward_sum = data[f"{prefix}rollout_reward_sum"]
    if rollout_reward_sum.dtype != np.float64 or rollout_reward_sum.shape != (
        n_rollouts,
        n_signals,
    ):
        raise ArtifactError(
            f"rollout_reward_sum must have dtype float64 and shape "
            f"({n_rollouts}, {n_signals})"
        )

    rollout_reward_abs_sum = data[f"{prefix}rollout_reward_abs_sum"]
    if rollout_reward_abs_sum.dtype != np.float64 or rollout_reward_abs_sum.shape != (
        n_rollouts,
        n_signals,
    ):
        raise ArtifactError(
            f"rollout_reward_abs_sum must have dtype float64 and shape "
            f"({n_rollouts}, {n_signals})"
        )

    rollout_reward_nonzero_count = data[f"{prefix}rollout_reward_nonzero_count"]
    if (
        rollout_reward_nonzero_count.dtype != np.int64
        or rollout_reward_nonzero_count.shape != (n_rollouts, n_signals)
    ):
        raise ArtifactError(
            f"rollout_reward_nonzero_count must have dtype int64 and shape "
            f"({n_rollouts}, {n_signals})"
        )

    rollout_reward_cross_product = data[f"{prefix}rollout_reward_cross_product"]
    if (
        rollout_reward_cross_product.dtype != np.float64
        or rollout_reward_cross_product.shape != (n_rollouts, n_signals, n_signals)
    ):
        raise ArtifactError(
            f"rollout_reward_cross_product must have dtype float64 and shape "
            f"({n_rollouts}, {n_signals}, {n_signals})"
        )

    episode_end_step = data[f"{prefix}episode_end_step"]
    if episode_end_step.dtype != np.int64 or episode_end_step.ndim != 1:
        raise ArtifactError("episode_end_step must have dtype int64 and 1D shape")
    n_episodes = episode_end_step.shape[0]

    episode_worker_rank = data[f"{prefix}episode_worker_rank"]
    if episode_worker_rank.dtype != np.int16 or episode_worker_rank.shape != (
        n_episodes,
    ):
        raise ArtifactError(
            "episode_worker_rank must have dtype int16 and shape (n_episodes,)"
        )

    episode_index = data[f"{prefix}episode_index"]
    if episode_index.dtype != np.int64 or episode_index.shape != (n_episodes,):
        raise ArtifactError(
            "episode_index must have dtype int64 and shape (n_episodes,)"
        )

    episode_length = data[f"{prefix}episode_length"]
    if episode_length.dtype != np.int32 or episode_length.shape != (n_episodes,):
        raise ArtifactError(
            "episode_length must have dtype int32 and shape (n_episodes,)"
        )

    episode_termination_reason = data[f"{prefix}episode_termination_reason"]
    if (
        episode_termination_reason.dtype != np.int8
        or episode_termination_reason.shape != (n_episodes,)
    ):
        raise ArtifactError(
            "episode_termination_reason must have dtype int8 and shape (n_episodes,)"
        )

    episode_complete = data[f"{prefix}episode_complete"]
    if episode_complete.dtype != np.bool_ or episode_complete.shape != (n_episodes,):
        raise ArtifactError(
            "episode_complete must have dtype bool and shape (n_episodes,)"
        )

    episode_reward_sums = data[f"{prefix}episode_reward_sums"]
    if episode_reward_sums.dtype != np.float64 or episode_reward_sums.shape != (
        n_episodes,
        n_signals,
    ):
        raise ArtifactError(
            f"episode_reward_sums must have dtype float64 and shape "
            f"({n_episodes}, {n_signals})"
        )

    component_slice = slice(0, TOTAL_REWARD_INDEX)
    if not np.allclose(
        rollout_reward_sum[:, TOTAL_REWARD_INDEX],
        rollout_reward_sum[:, component_slice].sum(axis=1),
        rtol=1e-6,
        atol=1e-3,
    ) or not np.allclose(
        episode_reward_sums[:, TOTAL_REWARD_INDEX],
        episode_reward_sums[:, component_slice].sum(axis=1),
        rtol=1e-6,
        atol=1e-3,
    ):
        raise ArtifactError("Reward diagnostics total does not equal component sum")

    return RewardDiagnostics(
        schema_version=schema_version,
        reward_names=reward_names,
        rollout_end_step=rollout_end_step,
        rollout_transition_count=rollout_transition_count,
        rollout_reward_sum=rollout_reward_sum,
        rollout_reward_abs_sum=rollout_reward_abs_sum,
        rollout_reward_nonzero_count=rollout_reward_nonzero_count,
        rollout_reward_cross_product=rollout_reward_cross_product,
        episode_end_step=episode_end_step,
        episode_worker_rank=episode_worker_rank,
        episode_index=episode_index,
        episode_length=episode_length,
        episode_termination_reason=episode_termination_reason,
        episode_complete=episode_complete,
        episode_reward_sums=episode_reward_sums,
    )


def _parse_safety_histogram_from_mapping(
    data: Any, path: str | Path, prefix: str = ""
) -> SafetyTruncationHistogram:
    bin_start_m = data[f"{prefix}bin_start_m"]
    if bin_start_m.dtype != np.float64 or bin_start_m.ndim != 1:
        raise ArtifactError("bin_start_m must have dtype float64 and 1D shape")
    n_bins = bin_start_m.shape[0]

    bin_end_m = data[f"{prefix}bin_end_m"]
    if bin_end_m.dtype != np.float64 or bin_end_m.shape != (n_bins,):
        raise ArtifactError("bin_end_m must have dtype float64 and shape (n_bins,)")

    safety_truncation_count = data[f"{prefix}safety_truncation_count"]
    if safety_truncation_count.dtype != np.int64 or safety_truncation_count.shape != (
        n_bins,
    ):
        raise ArtifactError(
            "safety_truncation_count must have dtype int64 and shape (n_bins,)"
        )

    low_safety_truncation_count = data[f"{prefix}low_safety_truncation_count"]
    if (
        low_safety_truncation_count.dtype != np.int64
        or low_safety_truncation_count.shape != (n_bins,)
    ):
        raise ArtifactError(
            "low_safety_truncation_count must have dtype int64 and shape (n_bins,)"
        )

    high_safety_truncation_count = data[f"{prefix}high_safety_truncation_count"]
    if (
        high_safety_truncation_count.dtype != np.int64
        or high_safety_truncation_count.shape != (n_bins,)
    ):
        raise ArtifactError(
            "high_safety_truncation_count must have dtype int64 and shape (n_bins,)"
        )

    global_safety_truncation_share = data[f"{prefix}global_safety_truncation_share"]
    if (
        global_safety_truncation_share.dtype != np.float64
        or global_safety_truncation_share.shape != (n_bins,)
    ):
        raise ArtifactError(
            "global_safety_truncation_share must have dtype float64 and shape (n_bins,)"
        )

    position_bin_size_m = data[f"{prefix}position_bin_size_m"]
    if position_bin_size_m.dtype != np.float64 or position_bin_size_m.shape != (1,):
        raise ArtifactError(
            "position_bin_size_m must have dtype float64 and shape (1,)"
        )

    return SafetyTruncationHistogram(
        bin_start_m=bin_start_m,
        bin_end_m=bin_end_m,
        safety_truncation_count=safety_truncation_count,
        low_safety_truncation_count=low_safety_truncation_count,
        high_safety_truncation_count=high_safety_truncation_count,
        global_safety_truncation_share=global_safety_truncation_share,
        position_bin_size_m=position_bin_size_m,
    )


def read_diagnostics(path: str | Path) -> TrainingDiagnostics:
    """Strictly load TrainingDiagnostics from .npz archive."""
    try:
        with np.load(path, allow_pickle=False) as data:
            actual_keys = frozenset(data.files)
            if actual_keys != _DIAGNOSTICS_KEYS:
                missing = sorted(_DIAGNOSTICS_KEYS - actual_keys)
                extra = sorted(actual_keys - _DIAGNOSTICS_KEYS)
                msg_parts = []
                if missing:
                    msg_parts.append(f"missing keys: {missing}")
                if extra:
                    msg_parts.append(f"extra keys: {extra}")
                msg_parts_str = ", ".join(msg_parts)
                raise ArtifactError(
                    f"{path} invalid TrainingDiagnostics archive: {msg_parts_str}"
                )
            reward = _parse_reward_diagnostics_from_mapping(data, path, prefix="")
            safety = _parse_safety_histogram_from_mapping(data, path, prefix="safety_")
            return TrainingDiagnostics(reward=reward, safety=safety)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(
            f"Failed to read TrainingDiagnostics from {path}: {exc}"
        ) from exc


def write_analysis_report(
    output_dir: str | Path,
    *,
    payload: dict[str, Any],
    markdown: str,
    csv_tables: dict[str, tuple[list[str], list[dict[str, Any]]]] | None = None,
) -> dict[str, str]:
    """Write analysis report outputs strictly to output_dir and return file paths."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "analysis_snapshot.json"
    markdown_path = out_path / "report.md"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)

    markdown_path.write_text(markdown, encoding="utf-8")

    output_paths: dict[str, str] = {
        "output_dir": str(out_path),
        "json_snapshot": str(json_path),
        "markdown_report": str(markdown_path),
    }

    if csv_tables:
        for table_name, (columns, rows) in csv_tables.items():
            csv_path = out_path / f"{table_name}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            output_paths[f"{table_name}_csv"] = str(csv_path)

    return output_paths


_EVALUATION_SPECS: dict[str, np.dtype] = {
    "training_steps": np.dtype(np.int64),
    "rollout_indices": np.dtype(np.int64),
    "total_reward": np.dtype(np.float64),
    "episode_steps": np.dtype(np.int64),
    "success": np.dtype(np.bool_),
    "safe": np.dtype(np.bool_),
    "feasible": np.dtype(np.bool_),
    "stop_error_m": np.dtype(np.float64),
    "time_error_s": np.dtype(np.float64),
    "total_energy_j": np.dtype(np.float64),
    "comfort_tav": np.dtype(np.float64),
    "completed_training_episodes": np.dtype(np.int64),
    "scheduled_completed_training_episodes": np.dtype(np.int64),
    "route_completion_ratio": np.dtype(np.float64),
    "safety_violation_positions_m": np.dtype(np.float64),
    "safety_violation_position_offsets": np.dtype(np.int64),
}


def write_evaluations(path: str | Path, history: EvaluationHistory) -> None:
    """Save EvaluationHistory to a .npz archive strictly."""
    data = {
        f.name: getattr(history, f.name) for f in dataclasses.fields(EvaluationHistory)
    }
    np.savez(path, **data)


def read_evaluations(path: str | Path) -> EvaluationHistory:
    """Strictly load EvaluationHistory from .npz archive."""
    try:
        with np.load(path, allow_pickle=False) as data:
            actual_keys = frozenset(data.files)
            if actual_keys != _EVALUATIONS_KEYS:
                missing = sorted(_EVALUATIONS_KEYS - actual_keys)
                extra = sorted(actual_keys - _EVALUATIONS_KEYS)
                msg_parts = []
                if missing:
                    msg_parts.append(f"missing keys: {missing}")
                if extra:
                    msg_parts.append(f"extra keys: {extra}")
                raise ArtifactError(
                    f"{path} invalid EvaluationHistory archive: {', '.join(msg_parts)}"
                )
            for name, expected_dtype in _EVALUATION_SPECS.items():
                arr = data[name]
                if arr.dtype != expected_dtype:
                    raise ArtifactError(
                        f"{path}: field {name} has dtype {arr.dtype}, "
                        f"expected {expected_dtype}"
                    )
            kwargs = {name: data[name] for name in _EVALUATIONS_KEYS}
            return EvaluationHistory(**kwargs)
    except ArtifactError:
        raise
    except (ValueError, TypeError) as exc:
        raise ArtifactError(f"{path}: invalid EvaluationHistory: {exc}") from exc
    except Exception as exc:
        raise ArtifactError(
            f"Failed to read EvaluationHistory from {path}: {exc}"
        ) from exc


def _run_record_json(record: RunRecord) -> str:
    payload = {
        "run_id": str(record.run_id),
        "kind": record.kind.value,
        "config": record.config,
        "scenario_hash": str(record.scenario_hash),
        "task": record.task,
        "policy_io_version": record.policy_io_version,
        "mtto_version": str(record.mtto_version),
        "created_at": str(record.created_at),
    }
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)


def write_run_record(path: str | Path, record: RunRecord) -> None:
    """Save RunRecord to JSON strictly."""
    content = _run_record_json(record)
    Path(path).write_text(content, encoding="utf-8")


def read_run_record(path: str | Path) -> RunRecord:
    """Strictly load RunRecord from JSON."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        raw = json.loads(text)
    except Exception as exc:
        raise ArtifactError(f"Failed to read JSON from {path}: {exc}") from exc

    data = _require_dict(raw, f"{path}")
    _require_keys(data, _RUN_RECORD_KEYS, f"{path}")

    kind_str = _require_str(data["kind"], f"{path}.kind")
    try:
        kind = RunKind(kind_str)
    except ValueError as exc:
        raise ArtifactError(f"Invalid RunKind '{kind_str}'") from exc

    pio_raw = data["policy_io_version"]
    policy_io_version = (
        _require_int(pio_raw, f"{path}.policy_io_version")
        if pio_raw is not None
        else None
    )

    return RunRecord(
        run_id=_require_str(data["run_id"], f"{path}.run_id"),
        kind=kind,
        config=_require_dict(data["config"], f"{path}.config"),
        scenario_hash=_require_str(data["scenario_hash"], f"{path}.scenario_hash"),
        task=_require_dict(data["task"], f"{path}.task"),
        policy_io_version=policy_io_version,
        mtto_version=_require_str(data["mtto_version"], f"{path}.mtto_version"),
        created_at=_require_str(data["created_at"], f"{path}.created_at"),
    )


def validate_payload(
    kind: RunKind, payload: RunPayload, *, policy_io_version: int | None
) -> None:
    """Validate in-memory run payload composition for the given run kind."""
    if kind == RunKind.RL_TRAIN:
        if payload.result is None:
            raise ArtifactError("rl_train payload must contain result")
        if payload.result.training is None:
            raise ArtifactError("rl_train result must contain training outcome")
        if payload.best is not None:
            if payload.best.result is None:
                raise ArtifactError("rl_train best payload must contain result")
            if payload.best.result.training is not None:
                raise ArtifactError(
                    "rl_train best result must not contain training outcome"
                )
            if payload.best.best is not None:
                raise ArtifactError("best payload must not nest another best payload")
            if (
                payload.best.diagnostics is not None
                or payload.best.evaluations is not None
            ):
                raise ArtifactError(
                    "rl_train best payload must not contain diagnostics or evaluations"
                )
    elif kind == RunKind.DP_SOLVE:
        if payload.result is not None:
            raise ArtifactError("dp_solve payload must not contain result")
        if payload.best is not None:
            raise ArtifactError("dp_solve payload must not contain best")
        if payload.diagnostics is not None or payload.evaluations is not None:
            raise ArtifactError(
                "dp_solve payload must not contain diagnostics or evaluations"
            )
    elif kind == RunKind.EVALUATION:
        if policy_io_version is not None:
            if payload.result is None:
                raise ArtifactError(
                    "evaluation with policy_io_version must contain result"
                )
            if payload.result.training is not None:
                raise ArtifactError(
                    "evaluation result must not contain training outcome"
                )
        else:
            if payload.result is not None:
                raise ArtifactError(
                    "evaluation without policy_io_version must not contain result"
                )
        if payload.best is not None:
            raise ArtifactError("evaluation payload must not contain best")
        if payload.diagnostics is not None or payload.evaluations is not None:
            raise ArtifactError(
                "evaluation payload must not contain diagnostics or evaluations"
            )


def write_run(run_dir: str | Path, record: RunRecord, payload: RunPayload) -> None:
    """Write run artifacts and atomically write run.json as final completion mark."""
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    run_json_path = target_dir / RUN_JSON
    if run_json_path.exists():
        raise ArtifactError(f"Cannot overwrite completed run: {run_json_path} exists")

    validate_payload(record.kind, payload, policy_io_version=record.policy_io_version)

    # Check policy.zip existence and SHA-256 for RL runs
    if POLICY_ZIP in REQUIRED_FILES[record.kind]:
        policy_path = target_dir / POLICY_ZIP
        if not policy_path.exists():
            raise ArtifactError(f"Required policy file not found: {policy_path}")
        assert payload.result is not None
        actual_sha = file_sha256(policy_path)
        if actual_sha != payload.result.policy_sha256:
            raise ArtifactError(
                f"Policy SHA-256 mismatch for {policy_path}: "
                f"expected {payload.result.policy_sha256}, got {actual_sha}"
            )
        if payload.result.policy_run_id != record.run_id:
            raise ArtifactError(
                f"policy_run_id mismatch for rl_train: "
                f"expected {record.run_id}, got {payload.result.policy_run_id}"
            )
        if payload.best is not None:
            best_policy = target_dir / BEST_DIR / POLICY_ZIP
            if not best_policy.exists():
                raise ArtifactError(
                    f"Required best policy file not found: {best_policy}"
                )
            assert payload.best.result is not None
            best_sha = file_sha256(best_policy)
            if best_sha != payload.best.result.policy_sha256:
                raise ArtifactError(
                    f"Best policy SHA-256 mismatch for {best_policy}: "
                    f"expected {payload.best.result.policy_sha256}, got {best_sha}"
                )

    # Write non-run.json artifacts
    write_profile(target_dir / PROFILE_NPZ, payload.profile)
    write_quality(target_dir / QUALITY_JSON, payload.quality)
    if payload.result is not None:
        write_result(target_dir / RESULT_JSON, payload.result)
    if payload.diagnostics is not None:
        write_diagnostics(target_dir / DIAGNOSTICS_NPZ, payload.diagnostics)
    if payload.evaluations is not None:
        write_evaluations(target_dir / EVALUATIONS_NPZ, payload.evaluations)

    if payload.best is not None:
        best_dir = target_dir / BEST_DIR
        best_dir.mkdir(parents=True, exist_ok=True)
        write_profile(best_dir / PROFILE_NPZ, payload.best.profile)
        write_quality(best_dir / QUALITY_JSON, payload.best.quality)
        if payload.best.result is not None:
            write_result(best_dir / RESULT_JSON, payload.best.result)

    # Strictly re-read all written files before writing run.json
    _ = read_profile(target_dir / PROFILE_NPZ)
    _ = read_quality(target_dir / QUALITY_JSON)
    if payload.result is not None:
        _ = read_result(target_dir / RESULT_JSON)
    if payload.diagnostics is not None:
        _ = read_diagnostics(target_dir / DIAGNOSTICS_NPZ)
    if payload.evaluations is not None:
        _ = read_evaluations(target_dir / EVALUATIONS_NPZ)
    if payload.best is not None:
        best_dir = target_dir / BEST_DIR
        _ = read_profile(best_dir / PROFILE_NPZ)
        _ = read_quality(best_dir / QUALITY_JSON)
        if payload.best.result is not None:
            _ = read_result(best_dir / RESULT_JSON)

    # Finally atomically write run.json
    temp_fd, temp_name = tempfile.mkstemp(dir=target_dir, prefix=".run_", suffix=".tmp")
    try:
        content = _run_record_json(record).encode("utf-8")
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, run_json_path)
    except Exception:
        if os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise


def read_completed_run(run_dir: str | Path) -> CompletedRun:
    """Read and strictly validate a completed run directory."""
    target_dir = Path(run_dir)
    run_json_path = target_dir / RUN_JSON
    if not run_json_path.exists():
        raise ArtifactError(f"Run is not completed: {run_json_path} does not exist")

    record = read_run_record(run_json_path)

    for fname in required_files_for(
        record.kind, policy_io_version=record.policy_io_version
    ):
        fpath = target_dir / fname
        if not fpath.exists():
            raise ArtifactError(f"Missing required artifact: {fpath}")

    profile = read_profile(target_dir / PROFILE_NPZ)
    quality = read_quality(target_dir / QUALITY_JSON)
    result = (
        read_result(target_dir / RESULT_JSON)
        if (target_dir / RESULT_JSON).exists()
        else None
    )

    best_payload: RunPayload | None = None
    best_dir = target_dir / BEST_DIR
    if best_dir.is_dir():
        for req_name in BEST_REQUIRED_FILES:
            fpath = best_dir / req_name
            if not fpath.exists():
                raise ArtifactError(f"Incomplete best/ directory: missing {fpath}")
        best_profile = read_profile(best_dir / PROFILE_NPZ)
        best_quality = read_quality(best_dir / QUALITY_JSON)
        best_result = read_result(best_dir / RESULT_JSON)
        best_payload = RunPayload(
            profile=best_profile,
            quality=best_quality,
            result=best_result,
        )

    diagnostics: TrainingDiagnostics | None = None
    diag_path = target_dir / DIAGNOSTICS_NPZ
    if diag_path.exists():
        diagnostics = read_diagnostics(diag_path)

    evaluations: EvaluationHistory | None = None
    eval_path = target_dir / EVALUATIONS_NPZ
    if eval_path.exists():
        evaluations = read_evaluations(eval_path)

    payload = RunPayload(
        profile=profile,
        quality=quality,
        result=result,
        best=best_payload,
        diagnostics=diagnostics,
        evaluations=evaluations,
    )
    validate_payload(record.kind, payload, policy_io_version=record.policy_io_version)
    return CompletedRun(record=record, payload=payload)
