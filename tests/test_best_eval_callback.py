import json
from pathlib import Path

import numpy as np

from model.ocs import TrainService
from rl.callbacks import (
    BestTrajectoryRecorder,
    PeriodicEvalCallback,
    SafetyViolationPositionRecorder,
)
from rl.evaluation import (
    ARRIVAL_STOP_SPEED_ABS_TOL_MPS,
    BEST_TRAJECTORY_SELECTION_RULE,
    PolicyEvaluationResult,
    classify_arrival_status,
    describe_best_update_reason,
    is_precise_arrival,
    is_punctual_arrival,
    is_successful_arrival,
)


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record(self, key: str, value: float, *_args, **_kwargs) -> None:
        self.records.append((key, value))

    def values_for(self, key: str) -> list[float]:
        return [value for record_key, value in self.records if record_key == key]


class DummyModel:
    def __init__(self, training_env: DummyTrainingEnv, logger: DummyLogger):
        self._training_env = training_env
        self.logger = logger

    def get_env(self) -> DummyTrainingEnv:
        return self._training_env

    def save(self, path: str) -> None:
        Path(f"{path}.zip").write_text("model", encoding="utf-8")


class DummyTrainingEnv:
    def save(self, path: str) -> None:
        Path(path).write_text("vecnormalize", encoding="utf-8")


class DummyEvalEnv:
    def close(self) -> None:
        return


def _build_train_service(*, schedule_time: float = 440.0) -> TrainService:
    return TrainService(
        start_position=0.0,
        start_speed=0.0,
        target_position=100.0,
        schedule_time=schedule_time,
        max_acc_change=0.75,
        max_arr_time_error_ratio=0.05,
        max_stop_error=0.3,
    )


def _build_result(
    *,
    success: bool,
    total_reward: float,
    total_energy_j: float = 12_345.0,
    stop_error_m: float | None = None,
    time_error_s: float | None = None,
    final_speed_mps: float | None = None,
    min_safety_margin_mps: float = 0.0,
    mean_safety_margin_mps: float = 0.0,
) -> PolicyEvaluationResult:
    resolved_stop_error = (
        float(stop_error_m) if stop_error_m is not None else (0.0 if success else 5.0)
    )
    resolved_time_error_s = (
        float(time_error_s) if time_error_s is not None else (0.0 if success else 20.0)
    )
    resolved_final_speed_mps = (
        float(final_speed_mps)
        if final_speed_mps is not None
        else (0.0 if success else 3.0)
    )
    train_service = _build_train_service()
    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=resolved_stop_error,
        time_error_s=resolved_time_error_s,
        final_speed_mps=resolved_final_speed_mps,
        train_service=train_service,
    )
    return PolicyEvaluationResult(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=total_reward,
        total_time_s=440.0 + resolved_time_error_s,
        target_time_s=440.0,
        total_energy_j=total_energy_j,
        total_energy_kj=total_energy_j / 1000.0,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=100.0 - resolved_stop_error,
        final_speed_mps=resolved_final_speed_mps,
        stop_error_m=resolved_stop_error,
        time_error_s=resolved_time_error_s,
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=22.0,
        comfort_tav=1.0,
        comfort_er_pct=2.0,
        comfort_rms=3.0,
        terminated=success,
        truncated=not success,
        episode_steps=10,
        trajectory_pos_m=np.asarray([0.0, 50.0, 100.0], dtype=np.float32),
        trajectory_speed_mps=np.asarray([0.0, 10.0, 0.0], dtype=np.float32),
        min_safety_margin_mps=min_safety_margin_mps,
        mean_safety_margin_mps=mean_safety_margin_mps,
    )


def _prepare_callback(
    tmp_path: Path,
    *,
    trigger_mode: str,
    trigger_interval: int,
    artifact_metadata: dict[str, object] | None = None,
) -> PeriodicEvalCallback:
    training_env = DummyTrainingEnv()
    logger = DummyLogger()
    callback = PeriodicEvalCallback(
        eval_env=DummyEvalEnv(),
        output_dir=str(tmp_path),
        artifact_metadata=artifact_metadata,
        eval_trigger_mode=trigger_mode,
        eval_trigger_interval=trigger_interval,
        deterministic=True,
    )
    callback.init_callback(DummyModel(training_env=training_env, logger=logger))  # type: ignore
    callback.locals = {}
    return callback


def _prepare_callback_with_recorders(
    *,
    trigger_mode: str,
    trigger_interval: int,
    recorders: list[object],
) -> PeriodicEvalCallback:
    training_env = DummyTrainingEnv()
    logger = DummyLogger()
    callback = PeriodicEvalCallback(
        eval_env=DummyEvalEnv(),
        recorders=recorders,
        eval_trigger_mode=trigger_mode,
        eval_trigger_interval=trigger_interval,
        deterministic=True,
    )
    callback.init_callback(DummyModel(training_env=training_env, logger=logger))  # type: ignore
    callback.locals = {}
    return callback


def test_periodic_eval_callback_triggers_on_episode_interval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    callback = _prepare_callback(
        tmp_path,
        trigger_mode="episodes",
        trigger_interval=3,
    )
    monkeypatch.setattr(
        "rl.callbacks.evaluate_policy_once",
        lambda *_args, **_kwargs: _build_result(success=True, total_reward=42.0),
    )

    callback.locals = {"dones": np.asarray([True, False], dtype=bool)}
    callback.num_timesteps = 10
    assert callback._on_step() is True
    assert callback.best_result is None

    callback.locals = {"dones": np.asarray([True, True], dtype=bool)}
    callback.num_timesteps = 20
    assert callback._on_step() is True

    assert callback.best_result is not None
    assert callback.best_trigger_interval == 3
    assert (tmp_path / "best_model.zip").exists()
    assert (tmp_path / "best_vecnormalize.pkl").exists()
    assert (tmp_path / "best_trajectory.npz").exists()
    metrics = json.loads(
        (tmp_path / "best_trajectory_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["eval_trigger_mode"] == "episodes"
    assert metrics["eval_trigger_interval"] == 3


def test_periodic_eval_callback_prefers_success_over_reward(
    tmp_path: Path,
    monkeypatch,
) -> None:
    callback = _prepare_callback(
        tmp_path,
        trigger_mode="steps",
        trigger_interval=1,
    )
    results = [
        _build_result(success=False, total_reward=100.0),
        _build_result(success=True, total_reward=1.0),
    ]
    monkeypatch.setattr(
        "rl.callbacks.evaluate_policy_once",
        lambda *_args, **_kwargs: results.pop(0),
    )

    callback.num_timesteps = 1
    assert callback._on_step() is True
    assert callback.best_result is not None
    assert callback.best_result.success is False

    callback.num_timesteps = 2
    assert callback._on_step() is True
    assert callback.best_result is not None
    assert callback.best_result.success is True
    assert callback.best_result.total_reward == 1.0

    metrics = json.loads(
        (tmp_path / "best_trajectory_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["success"] is True
    assert metrics["total_reward"] == 1.0


def test_periodic_eval_callback_persists_artifact_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    callback = _prepare_callback(
        tmp_path,
        trigger_mode="steps",
        trigger_interval=1,
        artifact_metadata={
            "reward_profile_name": "basic_safety",
            "experiment_tag": "trial_a",
        },
    )
    monkeypatch.setattr(
        "rl.callbacks.evaluate_policy_once",
        lambda *_args, **_kwargs: _build_result(success=True, total_reward=9.0),
    )

    callback.num_timesteps = 1
    assert callback._on_step() is True

    metrics = json.loads(
        (tmp_path / "best_trajectory_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["reward_profile_name"] == "basic_safety"
    assert metrics["experiment_tag"] == "trial_a"
    assert metrics["trajectory_source"] == "best"


def test_periodic_eval_callback_prefers_lower_energy_after_arrival_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    callback = _prepare_callback(
        tmp_path,
        trigger_mode="steps",
        trigger_interval=1,
    )
    results = [
        _build_result(success=True, total_reward=50.0, total_energy_j=6_000.0),
        _build_result(success=True, total_reward=10.0, total_energy_j=4_000.0),
    ]
    monkeypatch.setattr(
        "rl.callbacks.evaluate_policy_once",
        lambda *_args, **_kwargs: results.pop(0),
    )

    callback.num_timesteps = 1
    assert callback._on_step() is True
    callback.num_timesteps = 2
    assert callback._on_step() is True

    assert callback.best_result is not None
    assert callback.best_result.total_energy_j == 4_000.0

    metrics = json.loads(
        (tmp_path / "best_trajectory_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["best_update_reason"] == "lower_energy_after_arrival_requirements"
    assert metrics["success"] is True


def test_periodic_eval_callback_evaluates_once_per_trigger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    callback = _prepare_callback(
        tmp_path,
        trigger_mode="steps",
        trigger_interval=1,
    )
    call_count = 0

    def _fake_evaluate(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return _build_result(success=True, total_reward=42.0)

    monkeypatch.setattr("rl.callbacks.evaluate_policy_once", _fake_evaluate)

    callback.num_timesteps = 1
    assert callback._on_step() is True

    assert call_count == 1
    assert callback.best_result is not None


def test_safety_violation_position_recorder_records_bins(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "safety_violation_position_bins.npz"
    recorder = SafetyViolationPositionRecorder(
        output_path=str(output_path),
        position_bin_size_m=500.0,
    )
    recorder.init_callback(
        DummyModel(training_env=DummyTrainingEnv(), logger=DummyLogger())  # type: ignore[arg-type]
    )

    recorder.locals = {
        "infos": [
            {
                "state": {"position": 100.0},
                "constraint": {
                    "margin_to_vmax_mps": 2.0,
                    "margin_to_vmin_mps": 2.0,
                    "violation_code": 0,
                    "is_truncated": False,
                },
            },
            {
                "state": {"position": 600.0},
                "constraint": {
                    "margin_to_vmax_mps": -0.5,
                    "margin_to_vmin_mps": 1.0,
                    "violation_code": 3,
                    "is_truncated": False,
                },
            },
        ],
        "dones": np.asarray([False, False], dtype=bool),
    }
    assert recorder._on_step() is True

    recorder.locals = {
        "infos": [
            {
                "state": {"position": 650.0},
                "constraint": {
                    "margin_to_vmax_mps": 1.0,
                    "margin_to_vmin_mps": -0.2,
                    "violation_code": 2,
                    "is_truncated": True,
                },
            },
            {
                "state": {"position": 1100.0},
                "constraint": {
                    "margin_to_vmax_mps": 3.0,
                    "margin_to_vmin_mps": 2.5,
                    "violation_code": 1,
                    "is_truncated": True,
                },
            },
        ],
        "dones": np.asarray([True, True], dtype=bool),
    }
    assert recorder._on_step() is True

    recorder.locals = {
        "infos": [
            {
                "state": {"position": 1500.0},
                "constraint": {
                    "margin_to_vmax_mps": 4.0,
                    "margin_to_vmin_mps": 4.0,
                    "violation_code": 0,
                    "is_truncated": False,
                },
            }
        ],
        "dones": np.asarray([False], dtype=bool),
    }
    assert recorder._on_step() is True

    recorder._on_training_end()

    assert output_path.exists()
    with np.load(output_path) as data:
        np.testing.assert_allclose(
            data["bin_start_m"],
            [0.0, 500.0, 1000.0, 1500.0],
        )
        np.testing.assert_allclose(data["sample_exposure_count"], [1.0, 2.0, 1.0, 1.0])
        np.testing.assert_allclose(data["sample_violation_count"], [0.0, 2.0, 0.0, 0.0])
        np.testing.assert_allclose(
            data["sample_violation_rate"],
            [0.0, 1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(data["episode_exposure_count"], [1.0, 2.0, 1.0, 1.0])
        np.testing.assert_allclose(data["episode_violation_count"], [0.0, 2.0, 0.0, 0.0])
        np.testing.assert_allclose(
            data["episode_violation_rate"],
            [0.0, 1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(data["safety_truncation_count"], [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(data["position_bin_size_m"], [500.0])


def test_periodic_eval_default_recorder_does_not_write_safety_violation_bins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    callback = _prepare_callback(
        tmp_path,
        trigger_mode="steps",
        trigger_interval=1,
    )
    monkeypatch.setattr(
        "rl.callbacks.evaluate_policy_once",
        lambda *_args, **_kwargs: _build_result(success=True, total_reward=42.0),
    )

    callback.num_timesteps = 1
    assert callback._on_step() is True
    callback._on_training_end()

    assert not (tmp_path / "safety_violation_position_bins.npz").exists()


def test_describe_best_update_reason_prefers_higher_reward_when_no_success() -> None:
    higher_reward = _build_result(
        success=False,
        total_reward=12.0,
        total_energy_j=20_000.0,
        stop_error_m=8.0,
        time_error_s=30.0,
    )
    lower_reward = _build_result(
        success=False,
        total_reward=11.0,
        total_energy_j=1_000.0,
        stop_error_m=0.5,
        time_error_s=1.0,
    )

    assert (
        describe_best_update_reason(higher_reward, lower_reward)
        == "higher_total_reward_without_success"
    )
    assert describe_best_update_reason(lower_reward, higher_reward) is None


def test_to_metrics_includes_selection_rule() -> None:
    result = _build_result(success=True, total_reward=7.0, total_energy_j=3_000.0)

    metrics = result.to_metrics()

    assert metrics["selection_rule"] == BEST_TRAJECTORY_SELECTION_RULE
    assert metrics["success"] is True
    assert metrics["precise_arrival"] is True
    assert metrics["punctual_arrival"] is True
    assert metrics["strict_stop_error_limit_m"] == 0.3
    assert metrics["strict_time_error_limit_s"] == 22.0
    assert "strict_stop_requirement_met" not in metrics
    assert "strict_time_requirement_met" not in metrics
    assert metrics["selection_comparison_key"] == [
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        -3000.0,
    ]


def test_describe_best_update_reason_reports_success_upgrade() -> None:
    previous = _build_result(success=False, total_reward=100.0)
    candidate = _build_result(success=True, total_reward=1.0)

    assert (
        describe_best_update_reason(candidate, previous)
        == "success_replaces_reward_fallback"
    )


def test_describe_best_update_reason_prefers_precise_arrival() -> None:
    candidate = _build_result(
        success=True,
        total_reward=1.0,
        total_energy_j=20_000.0,
        stop_error_m=0.2,
        time_error_s=0.0,
    )
    previous = _build_result(
        success=True,
        total_reward=100.0,
        total_energy_j=1_000.0,
        stop_error_m=0.5,
        time_error_s=0.0,
    )

    assert (
        describe_best_update_reason(candidate, previous)
        == "precise_arrival_reached"
    )


def test_describe_best_update_reason_prefers_lower_stop_error_before_precise_arrival() -> (
    None
):
    candidate = _build_result(
        success=True,
        total_reward=1.0,
        total_energy_j=20_000.0,
        stop_error_m=0.6,
        time_error_s=0.0,
    )
    previous = _build_result(
        success=True,
        total_reward=100.0,
        total_energy_j=1_000.0,
        stop_error_m=0.7,
        time_error_s=0.0,
    )

    assert (
        describe_best_update_reason(candidate, previous)
        == "lower_stop_error_before_precise_arrival"
    )


def test_describe_best_update_reason_prefers_punctual_arrival() -> None:
    candidate = _build_result(
        success=True,
        total_reward=1.0,
        total_energy_j=20_000.0,
        stop_error_m=0.2,
        time_error_s=10.0,
    )
    previous = _build_result(
        success=True,
        total_reward=100.0,
        total_energy_j=1_000.0,
        stop_error_m=0.2,
        time_error_s=30.0,
    )

    assert (
        describe_best_update_reason(candidate, previous)
        == "punctual_arrival_reached"
    )


def test_describe_best_update_reason_prefers_lower_time_error_before_punctual_arrival() -> (
    None
):
    candidate = _build_result(
        success=True,
        total_reward=1.0,
        total_energy_j=20_000.0,
        stop_error_m=0.2,
        time_error_s=-30.0,
    )
    previous = _build_result(
        success=True,
        total_reward=100.0,
        total_energy_j=1_000.0,
        stop_error_m=0.2,
        time_error_s=40.0,
    )

    assert (
        describe_best_update_reason(candidate, previous)
        == "lower_time_error_before_punctual_arrival"
    )


def test_successful_arrival_matches_env_terminated_boundary() -> None:
    train_service = _build_train_service()
    assert is_successful_arrival(
        stop_error_m=3.0,
        final_speed_mps=ARRIVAL_STOP_SPEED_ABS_TOL_MPS,
        train_service=train_service,
    )
    assert not is_successful_arrival(
        stop_error_m=3.0001,
        final_speed_mps=0.0,
        train_service=train_service,
    )
    assert not is_successful_arrival(
        stop_error_m=0.0,
        final_speed_mps=ARRIVAL_STOP_SPEED_ABS_TOL_MPS + 0.001,
        train_service=train_service,
    )


def test_arrival_status_layers_require_previous_layer() -> None:
    train_service = _build_train_service(schedule_time=100.0)
    assert is_precise_arrival(
        success=True,
        stop_error_m=0.3,
        train_service=train_service,
    )
    assert not is_precise_arrival(
        success=False,
        stop_error_m=0.0,
        train_service=train_service,
    )
    assert is_punctual_arrival(
        precise_arrival=True,
        time_error_s=5.0,
        train_service=train_service,
    )
    assert not is_punctual_arrival(
        precise_arrival=False,
        time_error_s=0.0,
        train_service=train_service,
    )
    assert not is_punctual_arrival(
        precise_arrival=True,
        time_error_s=5.0001,
        train_service=train_service,
    )


def test_arrival_status_returns_false_when_schedule_time_invalid() -> None:
    train_service = _build_train_service(schedule_time=0.0)
    assert not is_punctual_arrival(
        precise_arrival=True,
        time_error_s=0.0,
        train_service=train_service,
    )
