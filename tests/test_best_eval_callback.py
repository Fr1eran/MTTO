import json
from pathlib import Path

import numpy as np

from rl.callbacks import BestTrajectoryEvalCallback
from rl.evaluation import PolicyEvaluationResult


class DummyLogger:
    def record(self, *_args, **_kwargs) -> None:
        return


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


def _build_result(*, success: bool, total_reward: float) -> PolicyEvaluationResult:
    return PolicyEvaluationResult(
        success=success,
        total_reward=total_reward,
        total_time_s=440.0,
        target_time_s=440.0,
        total_energy_j=12_345.0,
        total_energy_kj=12.345,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=100.0 if success else 95.0,
        final_speed_mps=0.0 if success else 3.0,
        stop_error_m=0.0 if success else 5.0,
        time_error_s=0.0 if success else 20.0,
        comfort_tav=1.0,
        comfort_er_pct=2.0,
        comfort_rms=3.0,
        terminated=success,
        truncated=not success,
        episode_steps=10,
        trajectory_pos_m=np.asarray([0.0, 50.0, 100.0], dtype=np.float32),
        trajectory_speed_mps=np.asarray([0.0, 10.0, 0.0], dtype=np.float32),
    )


def _prepare_callback(
    tmp_path: Path,
    *,
    trigger_mode: str,
    trigger_interval: int,
) -> BestTrajectoryEvalCallback:
    training_env = DummyTrainingEnv()
    logger = DummyLogger()
    callback = BestTrajectoryEvalCallback(
        eval_env=DummyEvalEnv(),
        output_dir=str(tmp_path),
        trigger_mode=trigger_mode,
        trigger_interval=trigger_interval,
        deterministic=True,
    )
    callback.init_callback(DummyModel(training_env=training_env, logger=logger))
    callback.locals = {}
    return callback


def test_best_eval_callback_triggers_on_episode_interval(
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
    assert callback.best_trigger_value == 3
    assert (tmp_path / "best_model.zip").exists()
    assert (tmp_path / "best_vecnormalize.pkl").exists()
    assert (tmp_path / "best_trajectory.npz").exists()
    metrics = json.loads(
        (tmp_path / "best_trajectory_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["trigger_mode"] == "episodes"
    assert metrics["trigger_value"] == 3


def test_best_eval_callback_prefers_success_over_reward(
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
