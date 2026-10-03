from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

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
from mtto.io.artifacts import (
    BEST_DIR,
    BEST_REQUIRED_FILES,
    EVALUATION_POLICY_RESULT_FILE,
    POLICY_ZIP,
    PROFILE_NPZ,
    QUALITY_JSON,
    REQUIRED_FILES,
    RESULT_JSON,
    RUN_JSON,
    ArtifactError,
    RLResult,
    RunKind,
    RunPayload,
    RunRecord,
    TrainingOutcome,
    canonical_json,
    file_sha256,
    read_completed_run,
    read_profile,
    read_quality,
    read_result,
    read_run_record,
    required_files_for,
    task_from_json,
    task_to_json,
    validate_payload,
    write_profile,
    write_quality,
    write_result,
    write_run,
    write_run_record,
)
from mtto.rl.state import TerminationReason


def _sample_speed_profile(kind: str = "normal") -> SpeedProfile:
    if kind == "normal":
        return SpeedProfile(
            position_m=np.asarray([0.0, 50.0, 100.0], dtype=np.float64),
            speed_mps=np.asarray([0.0, 10.0, 15.0], dtype=np.float64),
            time_s=np.asarray([0.0, 10.0, 14.0], dtype=np.float64),
            segment_acceleration_mps2=np.asarray([1.0, 1.25], dtype=np.float64),
            propulsion_energy_kj=np.asarray([0.0, 100.0, 250.0], dtype=np.float64),
            levitation_energy_kj=np.asarray([0.0, 50.0, 100.0], dtype=np.float64),
        )
    if kind == "zero_len":
        return SpeedProfile(
            position_m=np.asarray([0.0, 0.0, 50.0], dtype=np.float64),
            speed_mps=np.asarray([0.0, 0.0, 10.0], dtype=np.float64),
            time_s=np.asarray([0.0, 0.0, 10.0], dtype=np.float64),
            segment_acceleration_mps2=np.asarray([0.0, 1.0], dtype=np.float64),
            propulsion_energy_kj=np.asarray([0.0, 0.0, 100.0], dtype=np.float64),
            levitation_energy_kj=np.asarray([0.0, 0.0, 50.0], dtype=np.float64),
        )
    if kind == "n_equals_1":
        return SpeedProfile(
            position_m=np.asarray([0.0], dtype=np.float64),
            speed_mps=np.asarray([0.0], dtype=np.float64),
            time_s=np.asarray([0.0], dtype=np.float64),
            segment_acceleration_mps2=np.asarray([], dtype=np.float64),
            propulsion_energy_kj=np.asarray([0.0], dtype=np.float64),
            levitation_energy_kj=np.asarray([0.0], dtype=np.float64),
        )
    raise ValueError(f"Unknown sample profile kind {kind}")


def _sample_quality_report(
    *,
    with_events: bool = True,
    punctual: bool | None = True,
    arrival_err: float | None = 0.5,
) -> QualityReport:
    violations = (
        (
            AuditViolation(
                node_index=1,
                position_m=50.0,
                speed_mps=12.0,
                kind=ViolationKind.OVER_UPPER_LIMIT,
                margin_mps=2.0,
                category=ViolationCategory.PRE_TIMEOUT,
            ),
        )
        if with_events
        else ()
    )
    events = (
        (
            SpsEvent(
                node_index=0,
                kind=SpsEventKind.REQUEST_START,
                time_s=0.0,
                position_m=0.0,
                target_stopping_point=1,
            ),
        )
        if with_events
        else ()
    )
    metrics = QualityMetrics(
        propulsion_energy_kj=250.0,
        levitation_energy_kj=100.0,
        run_time_s=14.0,
        stop_error_m=0.05,
        comfort_tav_mps2=0.2,
        comfort_rms_mps2=0.3,
        comfort_exceedance_pct=0.0,
        arrival_time_error_s=arrival_err,
    )
    audit = SafetyAudit(
        target_stopping_point=np.asarray([1, 1, 1], dtype=np.int64),
        request_pending=np.asarray([True, False, False], dtype=np.bool_),
        lower_limit_mps=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        upper_limit_mps=np.asarray([100.0, 100.0, 100.0], dtype=np.float64),
        violations=violations,
        events=events,
        min_margin_mps=10.0,
    )
    return QualityReport(
        metrics=metrics,
        audit=audit,
        completed=True,
        precise_stop=True,
        safe=len(violations) == 0,
        punctual=punctual,
        feasible=True,
    )


def _sample_rl_result(
    *,
    term_reason: TerminationReason | None = TerminationReason.STOPPED_IN_ZONE,
    with_training: bool = True,
) -> RLResult:
    training = (
        TrainingOutcome(
            actual_training_timesteps=10000,
            actual_training_rollouts=5,
            actual_completed_episodes=50,
            target_reached=True,
            stop_reason="max_timesteps_reached",
        )
        if with_training
        else None
    )
    return RLResult(
        termination_reason=term_reason,
        truncated=False,
        total_reward=150.5,
        steps=25,
        final_position_m=100.0,
        final_speed_mps=0.0,
        final_time_s=14.0,
        deterministic=True,
        policy_run_id="run-1234",
        policy_sha256="abc123sha256",
        training=training,
    )


def _sample_run_record(
    kind: RunKind = RunKind.RL_TRAIN,
    policy_io_version: int | None = 1,
) -> RunRecord:
    task = Task(
        start_position_m=0.0,
        target_position_m=100.0,
        schedule_time_s=14.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=ScheduleChange(
            trigger_position_m=50.0, new_schedule_time_s=16.0
        ),
    )
    return RunRecord(
        run_id="run-1234",
        kind=kind,
        config={"seed": 42, "lr": 0.001},
        scenario_hash="scenario-hash-abc",
        task=task_to_json(task),
        policy_io_version=policy_io_version,
        mtto_version="0.1.0",
        created_at="2026-09-27T12:00:00+00:00",
    )


@pytest.mark.parametrize("kind", ["normal", "zero_len", "n_equals_1"])
def test_profile_roundtrip(tmp_path: Path, kind: str) -> None:
    profile = _sample_speed_profile(kind)
    file_path = tmp_path / "profile.npz"
    write_profile(file_path, profile)
    loaded = read_profile(file_path)

    assert np.array_equal(loaded.position_m, profile.position_m)
    assert np.array_equal(loaded.speed_mps, profile.speed_mps)
    assert np.array_equal(loaded.time_s, profile.time_s)
    assert np.array_equal(
        loaded.segment_acceleration_mps2, profile.segment_acceleration_mps2
    )
    assert np.array_equal(loaded.propulsion_energy_kj, profile.propulsion_energy_kj)
    assert np.array_equal(loaded.levitation_energy_kj, profile.levitation_energy_kj)


@pytest.mark.parametrize(
    ("with_events", "punctual", "arrival_err"),
    [
        (True, True, 0.5),
        (False, False, -0.2),
        (True, None, None),
    ],
)
def test_quality_roundtrip(
    tmp_path: Path,
    with_events: bool,
    punctual: bool | None,
    arrival_err: float | None,
) -> None:
    report = _sample_quality_report(
        with_events=with_events, punctual=punctual, arrival_err=arrival_err
    )
    file_path = tmp_path / "quality.json"
    write_quality(file_path, report)
    loaded = read_quality(file_path)

    assert loaded.metrics == report.metrics
    assert np.array_equal(
        loaded.audit.target_stopping_point, report.audit.target_stopping_point
    )
    assert np.array_equal(loaded.audit.request_pending, report.audit.request_pending)
    assert np.array_equal(loaded.audit.lower_limit_mps, report.audit.lower_limit_mps)
    assert np.array_equal(loaded.audit.upper_limit_mps, report.audit.upper_limit_mps)
    assert loaded.audit.violations == report.audit.violations
    assert loaded.audit.events == report.audit.events
    assert loaded.audit.min_margin_mps == pytest.approx(report.audit.min_margin_mps)
    assert loaded.completed == report.completed
    assert loaded.precise_stop == report.precise_stop
    assert loaded.safe == report.safe
    assert loaded.punctual == report.punctual
    assert loaded.feasible == report.feasible
    assert not loaded.audit.lower_limit_mps.flags.writeable


@pytest.mark.parametrize(
    ("term_reason", "with_training"),
    [
        (TerminationReason.STOPPED_IN_ZONE, True),
        (TerminationReason.OVERRAN, True),
        (None, False),
    ],
)
def test_rl_result_roundtrip(
    tmp_path: Path,
    term_reason: TerminationReason | None,
    with_training: bool,
) -> None:
    result = _sample_rl_result(term_reason=term_reason, with_training=with_training)
    file_path = tmp_path / "result.json"
    write_result(file_path, result)
    loaded = read_result(file_path)
    assert loaded == result


@pytest.mark.parametrize(
    ("kind", "policy_io_version"),
    [
        (RunKind.RL_TRAIN, 1),
        (RunKind.DP_SOLVE, None),
        (RunKind.EVALUATION, 1),
        (RunKind.EVALUATION, None),
    ],
)
def test_run_record_roundtrip(
    tmp_path: Path,
    kind: RunKind,
    policy_io_version: int | None,
) -> None:
    record = _sample_run_record(kind=kind, policy_io_version=policy_io_version)
    file_path = tmp_path / "run.json"
    write_run_record(file_path, record)
    loaded = read_run_record(file_path)
    assert loaded == record


@pytest.mark.parametrize("with_schedule_change", [True, False])
def test_task_json_roundtrip(with_schedule_change: bool) -> None:
    sc = (
        ScheduleChange(trigger_position_m=30.0, new_schedule_time_s=25.0)
        if with_schedule_change
        else None
    )
    task = Task(
        start_position_m=10.0,
        target_position_m=120.0,
        schedule_time_s=20.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=5.0,
        schedule_change=sc,
    )
    payload = task_to_json(task)
    reconstructed = task_from_json(payload)
    assert reconstructed == task


def test_strict_profile_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "profile.npz"
    # Missing key
    np.savez(
        file_path,
        position_m=np.asarray([0.0, 10.0]),
        speed_mps=np.asarray([0.0, 5.0]),
    )
    with pytest.raises(ArtifactError, match="invalid SpeedProfile archive"):
        read_profile(file_path)

    # Extra key
    np.savez(
        file_path,
        position_m=np.asarray([0.0, 10.0]),
        speed_mps=np.asarray([0.0, 5.0]),
        time_s=np.asarray([0.0, 2.0]),
        segment_acceleration_mps2=np.asarray([2.5]),
        propulsion_energy_kj=np.asarray([0.0, 10.0]),
        levitation_energy_kj=np.asarray([0.0, 5.0]),
        unexpected_extra=np.asarray([1.0]),
    )
    with pytest.raises(ArtifactError, match="invalid SpeedProfile archive"):
        read_profile(file_path)

    # Non-finite values
    np.savez(
        file_path,
        position_m=np.asarray([0.0, np.nan]),
        speed_mps=np.asarray([0.0, 5.0]),
        time_s=np.asarray([0.0, 2.0]),
        segment_acceleration_mps2=np.asarray([2.5]),
        propulsion_energy_kj=np.asarray([0.0, 10.0]),
        levitation_energy_kj=np.asarray([0.0, 5.0]),
    )
    with pytest.raises(ArtifactError, match="must be finite"):
        read_profile(file_path)


def test_strict_quality_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "quality.json"
    valid_report = _sample_quality_report()
    write_quality(file_path, valid_report)
    valid_data = json.loads(file_path.read_text(encoding="utf-8"))

    # Missing field
    corrupt = dict(valid_data)
    del corrupt["safe"]
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="key mismatch"):
        read_quality(file_path)

    # Extra field
    corrupt = dict(valid_data)
    corrupt["extra"] = True
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="key mismatch"):
        read_quality(file_path)

    # Type error (bool instead of number in metrics)
    corrupt = json.loads(json.dumps(valid_data))
    corrupt["metrics"]["run_time_s"] = True
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="must be a number"):
        read_quality(file_path)

    # String instead of number in metrics
    corrupt = json.loads(json.dumps(valid_data))
    corrupt["metrics"]["stop_error_m"] = "invalid"
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="must be a number"):
        read_quality(file_path)


def test_strict_result_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "result.json"
    valid_result = _sample_rl_result()
    write_result(file_path, valid_result)
    valid_data = json.loads(file_path.read_text(encoding="utf-8"))

    # Unknown termination reason string
    corrupt = dict(valid_data)
    corrupt["termination_reason"] = "UNKNOWN_REASON"
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="Unknown TerminationReason"):
        read_result(file_path)

    # Missing training field
    corrupt = json.loads(json.dumps(valid_data))
    del corrupt["training"]["target_reached"]
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="key mismatch"):
        read_result(file_path)


def test_strict_run_record_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "run.json"
    valid_record = _sample_run_record()
    write_run_record(file_path, valid_record)
    valid_data = json.loads(file_path.read_text(encoding="utf-8"))

    corrupt = dict(valid_data)
    corrupt["kind"] = "unsupported_kind"
    file_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ArtifactError, match="Invalid RunKind"):
        read_run_record(file_path)


def test_validate_payload_matrix() -> None:
    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    result_with_training = _sample_rl_result(with_training=True)
    result_without_training = _sample_rl_result(with_training=False)

    # rl_train: must have result with training
    payload_rl_valid = RunPayload(
        profile=profile, quality=quality, result=result_with_training
    )
    validate_payload(RunKind.RL_TRAIN, payload_rl_valid, policy_io_version=1)

    payload_rl_no_result = RunPayload(profile=profile, quality=quality, result=None)
    with pytest.raises(ArtifactError, match="rl_train payload must contain result"):
        validate_payload(RunKind.RL_TRAIN, payload_rl_no_result, policy_io_version=1)

    payload_rl_no_training = RunPayload(
        profile=profile, quality=quality, result=result_without_training
    )
    with pytest.raises(
        ArtifactError, match="rl_train result must contain training outcome"
    ):
        validate_payload(RunKind.RL_TRAIN, payload_rl_no_training, policy_io_version=1)

    # rl_train with best: best.result must exist and its training must be None
    best_valid = RunPayload(
        profile=profile, quality=quality, result=result_without_training
    )
    payload_with_best = RunPayload(
        profile=profile,
        quality=quality,
        result=result_with_training,
        best=best_valid,
    )
    validate_payload(RunKind.RL_TRAIN, payload_with_best, policy_io_version=1)

    best_with_training = RunPayload(
        profile=profile, quality=quality, result=result_with_training
    )
    payload_with_bad_best = RunPayload(
        profile=profile,
        quality=quality,
        result=result_with_training,
        best=best_with_training,
    )
    with pytest.raises(
        ArtifactError, match="rl_train best result must not contain training outcome"
    ):
        validate_payload(RunKind.RL_TRAIN, payload_with_bad_best, policy_io_version=1)

    # dp_solve: must not have result or best
    payload_dp_valid = RunPayload(profile=profile, quality=quality, result=None)
    validate_payload(RunKind.DP_SOLVE, payload_dp_valid, policy_io_version=None)

    with pytest.raises(ArtifactError, match="dp_solve payload must not contain result"):
        validate_payload(RunKind.DP_SOLVE, payload_rl_valid, policy_io_version=None)

    with pytest.raises(ArtifactError, match="dp_solve payload must not contain best"):
        validate_payload(
            RunKind.DP_SOLVE,
            RunPayload(profile=profile, quality=quality, best=best_valid),
            policy_io_version=None,
        )

    # evaluation: policy_io_version non-None requires result without training
    payload_eval_valid = RunPayload(
        profile=profile, quality=quality, result=result_without_training
    )
    validate_payload(RunKind.EVALUATION, payload_eval_valid, policy_io_version=1)

    with pytest.raises(
        ArtifactError, match="evaluation with policy_io_version must contain result"
    ):
        validate_payload(RunKind.EVALUATION, payload_dp_valid, policy_io_version=1)

    with pytest.raises(
        ArtifactError, match="evaluation result must not contain training outcome"
    ):
        validate_payload(RunKind.EVALUATION, payload_rl_valid, policy_io_version=1)

    # evaluation without policy_io_version must not have result
    validate_payload(RunKind.EVALUATION, payload_dp_valid, policy_io_version=None)
    with pytest.raises(
        ArtifactError,
        match="evaluation without policy_io_version must not contain result",
    ):
        validate_payload(RunKind.EVALUATION, payload_eval_valid, policy_io_version=None)


def test_write_and_read_completed_run_rl_train(tmp_path: Path) -> None:
    run_dir = tmp_path / "rl_run"
    run_dir.mkdir(parents=True)
    policy_path = run_dir / POLICY_ZIP
    policy_path.write_bytes(b"dummy policy zip content")
    policy_sha = file_sha256(policy_path)

    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    result = _sample_rl_result(with_training=True)
    result = dataclasses.replace(
        result, policy_run_id="run-test", policy_sha256=policy_sha
    )

    record = _sample_run_record(kind=RunKind.RL_TRAIN, policy_io_version=1)
    record = dataclasses.replace(record, run_id="run-test")

    payload = RunPayload(profile=profile, quality=quality, result=result)
    write_run(run_dir, record, payload)

    completed = read_completed_run(run_dir)
    assert completed.record == record
    assert np.array_equal(completed.payload.profile.position_m, profile.position_m)
    assert completed.payload.quality.metrics == quality.metrics
    assert completed.payload.result == result


def test_write_and_read_completed_run_with_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "rl_run_best"
    run_dir.mkdir(parents=True)
    best_dir = run_dir / BEST_DIR
    best_dir.mkdir(parents=True)

    policy_path = run_dir / POLICY_ZIP
    policy_path.write_bytes(b"main policy zip")
    policy_sha = file_sha256(policy_path)

    best_policy = best_dir / POLICY_ZIP
    best_policy.write_bytes(b"best policy zip")
    best_policy_sha = file_sha256(best_policy)

    profile = _sample_speed_profile()
    quality = _sample_quality_report()

    result = _sample_rl_result(with_training=True)
    result = dataclasses.replace(
        result, policy_run_id="run-best-test", policy_sha256=policy_sha
    )

    best_result = _sample_rl_result(with_training=False)
    best_result = dataclasses.replace(
        best_result, policy_run_id="run-best-test", policy_sha256=best_policy_sha
    )
    best_payload = RunPayload(profile=profile, quality=quality, result=best_result)

    record = _sample_run_record(kind=RunKind.RL_TRAIN, policy_io_version=1)
    record = dataclasses.replace(record, run_id="run-best-test")

    payload = RunPayload(
        profile=profile, quality=quality, result=result, best=best_payload
    )
    write_run(run_dir, record, payload)

    completed = read_completed_run(run_dir)
    assert completed.payload.best is not None
    assert completed.payload.best.result == best_result


def test_write_and_read_completed_run_dp_solve(tmp_path: Path) -> None:
    run_dir = tmp_path / "dp_run"
    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    record = _sample_run_record(kind=RunKind.DP_SOLVE, policy_io_version=None)
    payload = RunPayload(profile=profile, quality=quality)

    write_run(run_dir, record, payload)
    completed = read_completed_run(run_dir)
    assert completed.record == record
    assert completed.payload.result is None


def test_write_and_read_completed_run_eval_without_policy_io(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval_run_no_pio"
    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    record = _sample_run_record(kind=RunKind.EVALUATION, policy_io_version=None)
    payload = RunPayload(profile=profile, quality=quality, result=None)

    write_run(run_dir, record, payload)
    assert not (run_dir / POLICY_ZIP).exists()
    assert not (run_dir / RESULT_JSON).exists()

    completed = read_completed_run(run_dir)
    assert completed.record == record
    assert completed.payload.result is None


def test_write_and_read_completed_run_eval_with_policy_io(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval_run_with_pio"
    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    result = _sample_rl_result(with_training=False)
    record = _sample_run_record(kind=RunKind.EVALUATION, policy_io_version=1)
    payload = RunPayload(profile=profile, quality=quality, result=result)

    write_run(run_dir, record, payload)
    # Crucial: evaluation run does NOT require policy.zip in its run_dir
    assert not (run_dir / POLICY_ZIP).exists()
    assert (run_dir / RESULT_JSON).exists()

    completed = read_completed_run(run_dir)
    assert completed.record == record
    assert completed.payload.result == result


def test_write_run_atomic_failure_does_not_produce_run_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _sample_speed_profile()
    quality = _sample_quality_report()

    # Step 1: Failure during write_quality
    run_dir_1 = tmp_path / "fail_quality"
    record_dp = _sample_run_record(kind=RunKind.DP_SOLVE, policy_io_version=None)
    payload_dp = RunPayload(profile=profile, quality=quality)

    def _broken_write_quality(path: str | Path, r: QualityReport) -> None:
        del path, r
        raise RuntimeError("Simulated failure during quality.json writing")

    monkeypatch.setattr("mtto.io.artifacts.write_quality", _broken_write_quality)
    with pytest.raises(RuntimeError, match="Simulated failure during quality.json"):
        write_run(run_dir_1, record_dp, payload_dp)
    assert not (run_dir_1 / RUN_JSON).exists()
    with pytest.raises(ArtifactError, match="Run is not completed"):
        read_completed_run(run_dir_1)

    # Step 2: Failure during write_result
    monkeypatch.undo()
    run_dir_2 = tmp_path / "fail_result"
    run_dir_2.mkdir(parents=True)
    policy_path = run_dir_2 / POLICY_ZIP
    policy_path.write_bytes(b"dummy weights")
    policy_sha = file_sha256(policy_path)

    record_rl = _sample_run_record(kind=RunKind.RL_TRAIN, policy_io_version=1)
    record_rl = dataclasses.replace(record_rl, run_id="run-atomic-rl")
    result_rl = _sample_rl_result(with_training=True)
    result_rl = dataclasses.replace(
        result_rl, policy_run_id="run-atomic-rl", policy_sha256=policy_sha
    )
    payload_rl = RunPayload(profile=profile, quality=quality, result=result_rl)

    def _broken_write_result(path: str | Path, res: RLResult) -> None:
        del path, res
        raise RuntimeError("Simulated failure during result.json writing")

    monkeypatch.setattr("mtto.io.artifacts.write_result", _broken_write_result)
    with pytest.raises(RuntimeError, match="Simulated failure during result.json"):
        write_run(run_dir_2, record_rl, payload_rl)
    assert not (run_dir_2 / RUN_JSON).exists()
    with pytest.raises(ArtifactError, match="Run is not completed"):
        read_completed_run(run_dir_2)


def test_write_run_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing_run"
    run_dir.mkdir(parents=True)
    existing_run_json = run_dir / RUN_JSON
    existing_run_json.write_text('{"existing": true}', encoding="utf-8")

    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    record = _sample_run_record(kind=RunKind.DP_SOLVE, policy_io_version=None)
    payload = RunPayload(profile=profile, quality=quality)

    with pytest.raises(ArtifactError, match="Cannot overwrite completed run"):
        write_run(run_dir, record, payload)

    assert existing_run_json.read_text(encoding="utf-8") == '{"existing": true}'


def test_policy_zip_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "policy_checks"
    run_dir.mkdir(parents=True)

    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    record = _sample_run_record(kind=RunKind.RL_TRAIN, policy_io_version=1)

    # Missing policy file
    result = _sample_rl_result(with_training=True)
    result = dataclasses.replace(
        result, policy_run_id=record.run_id, policy_sha256="abc"
    )
    payload = RunPayload(profile=profile, quality=quality, result=result)

    with pytest.raises(ArtifactError, match="Required policy file not found"):
        write_run(run_dir, record, payload)
    assert not (run_dir / RUN_JSON).exists()

    # Policy SHA mismatch
    policy_path = run_dir / POLICY_ZIP
    policy_path.write_bytes(b"actual content")
    with pytest.raises(ArtifactError, match="Policy SHA-256 mismatch"):
        write_run(run_dir, record, payload)
    assert not (run_dir / RUN_JSON).exists()

    # Policy run_id mismatch
    actual_sha = file_sha256(policy_path)
    result_wrong_id = dataclasses.replace(
        result, policy_run_id="wrong_id", policy_sha256=actual_sha
    )
    payload_wrong_id = RunPayload(
        profile=profile, quality=quality, result=result_wrong_id
    )
    with pytest.raises(ArtifactError, match="policy_run_id mismatch"):
        write_run(run_dir, record, payload_wrong_id)
    assert not (run_dir / RUN_JSON).exists()


def test_required_files_table() -> None:
    assert REQUIRED_FILES[RunKind.DP_SOLVE] == (PROFILE_NPZ, QUALITY_JSON)
    assert REQUIRED_FILES[RunKind.RL_TRAIN] == (
        PROFILE_NPZ,
        QUALITY_JSON,
        RESULT_JSON,
        POLICY_ZIP,
    )
    assert REQUIRED_FILES[RunKind.EVALUATION] == (PROFILE_NPZ, QUALITY_JSON)
    assert required_files_for(RunKind.DP_SOLVE, policy_io_version=None) == (
        PROFILE_NPZ,
        QUALITY_JSON,
    )
    assert required_files_for(RunKind.RL_TRAIN, policy_io_version=1) == (
        PROFILE_NPZ,
        QUALITY_JSON,
        RESULT_JSON,
        POLICY_ZIP,
    )
    assert required_files_for(RunKind.EVALUATION, policy_io_version=None) == (
        PROFILE_NPZ,
        QUALITY_JSON,
    )
    assert required_files_for(RunKind.EVALUATION, policy_io_version=1) == (
        PROFILE_NPZ,
        QUALITY_JSON,
        RESULT_JSON,
    )
    assert EVALUATION_POLICY_RESULT_FILE == RESULT_JSON
    assert BEST_REQUIRED_FILES == (
        PROFILE_NPZ,
        QUALITY_JSON,
        RESULT_JSON,
        POLICY_ZIP,
    )


@pytest.mark.parametrize(
    ("kind", "policy_io_version", "missing_file"),
    [
        (RunKind.DP_SOLVE, None, PROFILE_NPZ),
        (RunKind.DP_SOLVE, None, QUALITY_JSON),
        (RunKind.RL_TRAIN, 1, PROFILE_NPZ),
        (RunKind.RL_TRAIN, 1, QUALITY_JSON),
        (RunKind.RL_TRAIN, 1, RESULT_JSON),
        (RunKind.RL_TRAIN, 1, POLICY_ZIP),
        (RunKind.EVALUATION, 1, PROFILE_NPZ),
        (RunKind.EVALUATION, 1, QUALITY_JSON),
        (RunKind.EVALUATION, 1, RESULT_JSON),
        (RunKind.EVALUATION, None, PROFILE_NPZ),
        (RunKind.EVALUATION, None, QUALITY_JSON),
    ],
)
def test_read_completed_run_missing_required_files(
    tmp_path: Path,
    kind: RunKind,
    policy_io_version: int | None,
    missing_file: str,
) -> None:
    run_dir = tmp_path / f"missing_{kind}_{policy_io_version}_{missing_file}"
    run_dir.mkdir(parents=True)
    profile = _sample_speed_profile()
    quality = _sample_quality_report()

    if kind == RunKind.RL_TRAIN:
        policy_path = run_dir / POLICY_ZIP
        policy_path.write_bytes(b"dummy policy bytes")
        policy_sha = file_sha256(policy_path)
        result = _sample_rl_result(with_training=True)
        result = dataclasses.replace(
            result, policy_run_id="run-test", policy_sha256=policy_sha
        )
        payload = RunPayload(profile=profile, quality=quality, result=result)
    elif kind == RunKind.EVALUATION and policy_io_version is not None:
        result = _sample_rl_result(with_training=False)
        payload = RunPayload(profile=profile, quality=quality, result=result)
    else:
        payload = RunPayload(profile=profile, quality=quality, result=None)

    record = _sample_run_record(kind=kind, policy_io_version=policy_io_version)
    record = dataclasses.replace(record, run_id="run-test")
    write_run(run_dir, record, payload)

    # Delete the specified required file
    (run_dir / missing_file).unlink()
    with pytest.raises(ArtifactError, match="Missing required artifact"):
        read_completed_run(run_dir)


def test_read_completed_run_incomplete_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete_best"
    run_dir.mkdir(parents=True)
    policy_path = run_dir / POLICY_ZIP
    policy_path.write_bytes(b"policy")
    policy_sha = file_sha256(policy_path)

    best_dir = run_dir / BEST_DIR
    best_dir.mkdir(parents=True)
    best_policy = best_dir / POLICY_ZIP
    best_policy.write_bytes(b"best policy")
    best_sha = file_sha256(best_policy)

    profile = _sample_speed_profile()
    quality = _sample_quality_report()
    result = _sample_rl_result(with_training=True)
    result = dataclasses.replace(
        result, policy_run_id="run-id", policy_sha256=policy_sha
    )
    best_result = _sample_rl_result(with_training=False)
    best_result = dataclasses.replace(
        best_result, policy_run_id="run-id", policy_sha256=best_sha
    )

    record = _sample_run_record(kind=RunKind.RL_TRAIN, policy_io_version=1)
    record = dataclasses.replace(record, run_id="run-id")
    payload = RunPayload(
        profile=profile,
        quality=quality,
        result=result,
        best=RunPayload(profile=profile, quality=quality, result=best_result),
    )
    write_run(run_dir, record, payload)

    # Delete one file in best/
    (best_dir / RESULT_JSON).unlink()
    with pytest.raises(ArtifactError, match="Incomplete best/ directory"):
        read_completed_run(run_dir)


def test_deterministic_encoding(tmp_path: Path) -> None:
    record = _sample_run_record()
    p1 = tmp_path / "record1.json"
    p2 = tmp_path / "record2.json"
    write_run_record(p1, record)
    write_run_record(p2, record)
    assert p1.read_bytes() == p2.read_bytes()

    # canonical_json key ordering
    dict_a = {"b": 2, "a": 1, "c": {"y": 20, "x": 10}}
    dict_b = {"a": 1, "c": {"x": 10, "y": 20}, "b": 2}
    assert canonical_json(dict_a) == canonical_json(dict_b)
