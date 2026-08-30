import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from stable_baselines3.common.base_class import BaseAlgorithm

from model.ocs import TrainService
from rl.callbacks import (
    BestEvaluationArtifactHandler,
    EvaluationHistoryArtifactHandler,
    RewardDiagnosticsArtifactCallback,
    SafetyTruncationPositionHistogramCallback,
    ScheduledPolicyEvaluationCallback,
)
from rl.evaluation import (
    BEST_TRAJECTORY_SELECTION_RULE,
    PolicyEvaluationResult,
    classify_arrival_status,
    describe_best_update_reason,
    get_strict_time_error_limit_s,
)
from rl.reward_diagnostics import REWARD_NAMES, REWARD_SIGNAL_COUNT


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record(self, key: str, value: float, *_args: object, **_kwargs: object) -> None:
        self.records.append((key, value))


class DummyModel:
    def __init__(self, training_env: object):
        self._training_env = training_env
        self.logger = DummyLogger()

    def get_env(self) -> object:
        return self._training_env

    def save(self, path: str) -> None:
        Path(f"{path}.zip").write_text("model", encoding="utf-8")


class DummyTrainingEnv:
    def __init__(
        self,
        *,
        batches: list[list[dict[str, object]]] | None = None,
        method_name: str = "drain_safety_truncations",
        num_envs: int = 1,
    ) -> None:
        self.batches = list(batches or [])
        self.method_name = method_name
        self.num_envs = num_envs

    def env_method(self, method_name: str, *_args: object, **_kwargs: object):
        assert method_name == self.method_name
        return self.batches.pop(0)


class DummyEvalEnv:
    def close(self) -> None:
        return


def _build_result(
    *,
    success: bool,
    total_reward: float,
    total_energy_j: float = 12_345.0,
    safety_positions: tuple[float, ...] = (),
) -> PolicyEvaluationResult:
    stop_error = 0.0 if success else 12.0
    time_error = 0.0 if success else 20.0
    final_speed = 0.0 if success else 3.0
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
    )
    success, precise, punctual = classify_arrival_status(
        stop_error_m=stop_error,
        time_error_s=time_error,
        final_speed_mps=final_speed,
        train_service=service,
        terminated=success,
        truncated=not success,
    )
    return PolicyEvaluationResult(
        success=success,
        precise_arrival=precise,
        punctual_arrival=punctual,
        total_reward=total_reward,
        total_time_s=440.0 + time_error,
        target_time_s=440.0,
        total_energy_j=total_energy_j,
        total_energy_kj=total_energy_j / 1000.0,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=100.0 - stop_error,
        final_speed_mps=final_speed,
        stop_error_m=stop_error,
        time_error_s=time_error,
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=10.0,
        comfort_tav=1.0,
        comfort_er_pct=2.0,
        comfort_rms=3.0,
        terminated=success,
        truncated=not success,
        episode_steps=10,
        trajectory_pos_m=np.asarray([0.0, 100.0], dtype=np.float32),
        trajectory_speed_mps=np.asarray([0.0, 0.0], dtype=np.float32),
        min_safety_margin_mps=0.0,
        mean_safety_margin_mps=0.0,
        safety_violation_positions_m=np.asarray(safety_positions, dtype=np.float32),
    )


def _init(callback: object, training_env: object) -> None:
    callback.init_callback(  # type: ignore[attr-defined]
        cast(BaseAlgorithm, cast(object, DummyModel(training_env)))
    )


def test_punctual_arrival_uses_train_service_absolute_limit() -> None:
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
    )

    assert service.max_arr_time_error_s == pytest.approx(10.0)
    assert get_strict_time_error_limit_s(service) == pytest.approx(10.0)
    assert classify_arrival_status(
        stop_error_m=0.0,
        time_error_s=9.999,
        final_speed_mps=0.0,
        train_service=service,
        terminated=True,
        truncated=False,
    ) == (True, True, True)
    assert classify_arrival_status(
        stop_error_m=0.0,
        time_error_s=10.0,
        final_speed_mps=0.0,
        train_service=service,
        terminated=True,
        truncated=False,
    ) == (True, True, False)

    custom_service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
        max_arr_time_error_s=5.0,
    )
    assert classify_arrival_status(
        stop_error_m=0.0,
        time_error_s=5.0,
        final_speed_mps=0.0,
        train_service=custom_service,
        terminated=True,
        truncated=False,
    ) == (True, True, False)


@pytest.mark.parametrize("invalid_limit", [0.0, -1.0, float("nan"), float("inf")])
def test_train_service_rejects_invalid_arrival_time_limit(
    invalid_limit: float,
) -> None:
    with pytest.raises(ValueError, match="max_arr_time_error_s"):
        TrainService(
            start_position=0.0,
            target_position=100.0,
            schedule_time=440.0,
            max_acc_change=0.75,
            max_stop_error=0.3,
            max_arr_time_error_s=invalid_limit,
        )


def test_scheduled_evaluation_shares_one_result_between_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    best = BestEvaluationArtifactHandler(output_dir=str(tmp_path / "best"))
    history_path = tmp_path / "evaluations.npz"
    history = EvaluationHistoryArtifactHandler(output_path=str(history_path))
    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        handlers=[best, history],
        evaluation_interval_rollouts=12,
        get_completed_training_episodes=lambda: 17,
    )
    _init(callback, DummyTrainingEnv())
    calls = 0

    def evaluate(*_args: object, **_kwargs: object) -> PolicyEvaluationResult:
        nonlocal calls
        calls += 1
        return _build_result(success=True, total_reward=5.0, safety_positions=(20.0,))

    monkeypatch.setattr("rl.callbacks.evaluate_policy_once", evaluate)
    assert callback._on_step() is True
    for rollout_index in range(1, 12):
        callback.num_timesteps = rollout_index * 10
        callback._on_rollout_end()
    assert calls == 0
    callback.num_timesteps = 120
    callback._on_rollout_end()
    callback._on_training_end()

    assert calls == 1
    assert best.best_result is not None
    metrics = json.loads((tmp_path / "best" / "metrics_best.json").read_text())
    assert metrics["evaluation_rollout_index"] == 12
    with np.load(history_path) as data:
        np.testing.assert_array_equal(data["training_steps"], [120])
        np.testing.assert_array_equal(data["rollout_indices"], [12])
        np.testing.assert_array_equal(data["completed_training_episodes"], [17])
        np.testing.assert_allclose(data["safety_violation_positions_m"], [20.0])


def test_scheduled_evaluation_repeats_at_rollout_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        handlers=[],
        evaluation_interval_rollouts=12,
    )
    _init(callback, DummyTrainingEnv())
    calls = 0

    def evaluate(*_args: object, **_kwargs: object) -> PolicyEvaluationResult:
        nonlocal calls
        calls += 1
        return _build_result(success=True, total_reward=5.0)

    monkeypatch.setattr("rl.callbacks.evaluate_policy_once", evaluate)
    for rollout_index in range(1, 25):
        callback.num_timesteps = rollout_index * 10
        callback._on_rollout_end()
        assert calls == rollout_index // 12


def test_scheduled_evaluation_rejects_nonpositive_rollout_interval() -> None:
    with pytest.raises(ValueError, match="evaluation_interval_rollouts"):
        _ = ScheduledPolicyEvaluationCallback(
            eval_env=DummyEvalEnv(),
            handlers=[],
            evaluation_interval_rollouts=0,
        )


def test_safety_histogram_has_no_step_hook_and_saves_rollout_data(
    tmp_path: Path,
) -> None:
    output = tmp_path / "safety_truncation_position_histogram.npz"
    env = DummyTrainingEnv(
        batches=[
            [
                {
                    "position_m": np.asarray([100.0, 650.0, 650.0]),
                    "violation_code": np.asarray([2, 3, 2]),
                }
            ],
            [
                {
                    "position_m": np.asarray([], dtype=np.float32),
                    "violation_code": np.asarray([], dtype=np.int8),
                }
            ],
        ]
    )
    callback = SafetyTruncationPositionHistogramCallback(
        output_path=str(output), position_bin_size_m=500.0
    )
    _init(callback, env)
    assert callback._on_step() is True
    callback._on_rollout_end()
    callback._on_training_end()
    with np.load(output) as data:
        np.testing.assert_allclose(data["bin_start_m"], [0.0, 500.0])
        np.testing.assert_array_equal(data["safety_truncation_count"], [1, 2])


def _reward_batch(*, count: int, complete: bool) -> dict[str, object]:
    sums = np.arange(REWARD_SIGNAL_COUNT, dtype=np.float64)
    sums[-1] = sums[:-1].sum()
    return {
        "transition_count": np.asarray([count], dtype=np.int64),
        "reward_sum": sums,
        "reward_abs_sum": np.abs(sums),
        "reward_nonzero_count": np.full(REWARD_SIGNAL_COUNT, count, dtype=np.int64),
        "reward_cross_product": np.outer(sums, sums),
        "episode_end_worker_step": np.asarray([count], dtype=np.int64),
        "episode_worker_rank": np.asarray([0], dtype=np.int16),
        "episode_index": np.asarray([0], dtype=np.int64),
        "episode_length": np.asarray([count], dtype=np.int32),
        "episode_terminated": np.asarray([complete]),
        "episode_truncated": np.asarray([False]),
        "episode_complete": np.asarray([complete]),
        "episode_violation_code": np.asarray([0], dtype=np.int8),
        "episode_reward_sums": sums.reshape(1, -1),
    }


def test_reward_artifact_callback_drains_on_rollout_and_training_end(
    tmp_path: Path,
) -> None:
    empty = _reward_batch(count=0, complete=False)
    for key in tuple(empty):
        if key.startswith("episode_"):
            value = np.asarray(empty[key])
            empty[key] = value[:0]
    env = DummyTrainingEnv(
        method_name="drain_reward_diagnostics",
        batches=[[empty], [_reward_batch(count=4, complete=True)]],
    )
    output = tmp_path / "reward_diagnostics.npz"
    callback = RewardDiagnosticsArtifactCallback(output_path=str(output))
    _init(callback, env)
    callback.num_timesteps = 4
    callback._on_rollout_end()
    callback._on_training_end()
    assert callback.completed_episode_count == 1
    with np.load(output, allow_pickle=False) as data:
        assert tuple(data["reward_names"].tolist()) == REWARD_NAMES
        np.testing.assert_array_equal(data["rollout_transition_count"], [4])
        np.testing.assert_array_equal(data["episode_complete"], [True])


def test_best_selection_and_metrics_contract() -> None:
    failed = _build_result(success=False, total_reward=100.0)
    succeeded = _build_result(success=True, total_reward=1.0)
    assert (
        describe_best_update_reason(succeeded, failed)
        == "success_replaces_reward_fallback"
    )
    assert succeeded.to_metrics()["selection_rule"] == BEST_TRAJECTORY_SELECTION_RULE
