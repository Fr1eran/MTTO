"""PPO construction, callbacks, learning-rate schedules and SB3 termination tests."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from mtto.domain.safeguard import ViolationKind
from mtto.domain.scenario import Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import (
    AuditViolation,
    QualityMetrics,
    QualityReport,
    SafetyAudit,
    ViolationCategory,
)
from mtto.rl.evaluate import RLRun, calculate_route_completion_ratio
from mtto.rl.ppo import (
    CompletedEpisodeProgress,
    ScheduledPolicyEvaluationCallback,
    StopTrainingOnCompletedEpisodes,
)
from mtto.rl.state import TerminationReason
from tests.golden.drive import build_env


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record(self, key: str, value: float, *_args: object, **_kwargs: object) -> None:
        self.records.append((key, value))


class DummyModel:
    def __init__(
        self,
        training_env: object = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._training_env = training_env
        self._parameters = parameters or {}
        self.logger = DummyLogger()

    def get_env(self) -> object:
        return self._training_env

    def get_parameters(self) -> dict[str, Any]:
        return self._parameters


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
    def __init__(self) -> None:
        self.scenario = None
        self.task = Task(
            start_position_m=0.0,
            target_position_m=100.0,
            schedule_time_s=440.0,
            max_jerk_mps3=0.75,
            max_stop_error_m=0.3,
            max_arr_time_error_s=10.0,
        )

    def close(self) -> None:
        pass


def _dummy_run(reward: float = 0.0) -> RLRun:
    profile = SpeedProfile.from_arrays(
        [0.0, 100.0],
        [0.0, 0.0],
        [0.0, 440.0],
        [0.0, 0.0],
        [0.0, 0.0],
    )
    return RLRun(
        profile=profile,
        termination_reason=TerminationReason.STOPPED_IN_ZONE,
        total_reward=reward,
        steps=10,
        deterministic=True,
    )


def _make_report(
    *,
    feasible: bool = False,
    completed: bool = False,
    precise_stop: bool = False,
    safe: bool = False,
    punctual: bool = False,
    stop_error_m: float = 10.0,
    arrival_time_error_s: float = 5.0,
    total_energy_kj: float = 100.0,
) -> QualityReport:
    metrics = QualityMetrics(
        propulsion_energy_kj=total_energy_kj,
        levitation_energy_kj=0.0,
        run_time_s=100.0,
        stop_error_m=stop_error_m,
        comfort_tav_mps2=0.5,
        comfort_rms_mps2=0.5,
        comfort_exceedance_pct=0.0,
        arrival_time_error_s=arrival_time_error_s,
    )
    violation = AuditViolation(
        node_index=0,
        position_m=10.0,
        speed_mps=5.0,
        kind=ViolationKind.OVER_UPPER_LIMIT,
        margin_mps=-1.0,
        category=ViolationCategory.PRE_TIMEOUT,
    )
    audit = SafetyAudit(
        target_stopping_point=np.zeros(0, dtype=np.int64),
        request_pending=np.zeros(0, dtype=np.bool_),
        lower_limit_mps=np.zeros(0, dtype=np.float64),
        upper_limit_mps=np.zeros(0, dtype=np.float64),
        violations=() if safe else (violation,),
        events=(),
        min_margin_mps=1.0 if safe else -1.0,
    )
    return QualityReport(
        metrics=metrics,
        audit=audit,
        completed=completed,
        precise_stop=precise_stop,
        safe=safe,
        punctual=punctual,
        feasible=feasible,
    )


def _init(
    callback: object,
    training_env: object,
    parameters: dict[str, Any] | None = None,
) -> None:
    model = DummyModel(training_env, parameters=parameters)
    callback.init_callback(  # type: ignore[attr-defined]
        cast(BaseAlgorithm, cast(object, model))
    )


def test_callbacks_do_not_write_files_and_have_consistent_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    from mtto.rl.env import make_env
    from mtto.rl.ppo import (
        RewardDiagnosticsCallback,
        SafetyTruncationHistogramCallback,
        ScheduledPolicyEvaluationCallback,
        build_ppo,
    )
    from mtto.workflows.train import build_env_references
    from paper.figures import load_paper_scenario, load_paper_task

    scenario = load_paper_scenario()
    task = load_paper_task()
    lookup, normalization = build_env_references(scenario, task)

    train_env = VecMonitor(
        DummyVecEnv(
            [
                lambda: make_env(
                    scenario,
                    task,
                    0.998,
                    1.0,
                    lookup,
                    normalization,
                    compact_training_info=True,
                    enable_safety_truncation_tracking=True,
                    reward_diagnostics_worker_rank=0,
                    reward_diagnostics_rollout_capacity=64,
                )
            ]
        )
    )
    eval_env = make_env(
        scenario,
        task,
        0.998,
        1.0,
        lookup,
        normalization,
        enable_trajectory_tracking=True,
    )

    reward_cb = RewardDiagnosticsCallback()
    safety_cb = SafetyTruncationHistogramCallback(position_bin_size_m=5000.0)
    eval_cb = ScheduledPolicyEvaluationCallback(
        eval_env=eval_env,
        evaluation_interval_rollouts=1,
    )

    monkeypatch.chdir(tmp_path)
    model = build_ppo(
        venv=train_env,
        n_steps=64,
        gamma=0.998,
        learning_rate=3e-4,
        batch_size=64,
    )
    assert model.policy.log_std.detach().eq(-1.0).all()
    model.learn(
        total_timesteps=128,
        callback=CallbackList([reward_cb, safety_cb, eval_cb]),
    )

    # 1. 断言回调完成后临时目录中没有回调产生的文件
    assert list(tmp_path.iterdir()) == []

    # 2. diagnostics、histogram、history、best 均可取得且形状一致
    diag = reward_cb.diagnostics
    hist = safety_cb.histogram
    history = eval_cb.history
    best = eval_cb.best

    assert diag is not None
    assert hist is not None
    assert history is not None
    assert best is not None

    assert diag.rollout_end_step.shape == (2,)
    assert diag.rollout_transition_count.shape == (2,)
    assert diag.rollout_reward_sum.shape == (2, diag.reward_names.size)
    assert history.training_steps.shape == (2,)
    assert history.rollout_indices.shape == (2,)
    assert history.total_reward.shape == (2,)
    assert hist.bin_start_m.shape == hist.safety_truncation_count.shape
    assert best.training_step in history.training_steps
    assert best.quality is not None
    assert best.run is not None
    assert isinstance(best.parameters, dict)


def test_best_policy_selection_and_parameter_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtto.rl.ppo as ppo_module

    model_params = {"weights": np.array([1.0, 2.0], dtype=np.float64)}
    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        evaluation_interval_rollouts=1,
    )
    _init(callback, DummyTrainingEnv(), parameters=model_params)

    dummy_run = _dummy_run(reward=10.0)
    monkeypatch.setattr(ppo_module, "run_policy", lambda *a, **k: dummy_run)

    # 1. first_evaluation: 初始候选模型评估，生成首次 best
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(completed=False, stop_error_m=50.0),
    )
    callback.num_timesteps = 10
    callback._on_rollout_end()
    assert callback.best is not None
    assert callback.best.update_reason == "first_evaluation"
    assert callback.best.parameters["weights"][0] == 1.0

    # 验证深拷贝：修改模型权重，回调保存的参数不受影响
    model_params["weights"][0] = 999.0
    assert callback.best.parameters["weights"][0] == 1.0

    # 2. None 分支：更差的候选不触发择优更新
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(completed=False, stop_error_m=100.0),
    )
    callback.num_timesteps = 20
    callback._on_rollout_end()
    assert callback.best.update_reason == "first_evaluation"
    assert callback.best.parameters["weights"][0] == 1.0

    # 3. safe_success_reached: 从未完成到安全完成
    model_params["weights"][0] = 3.0
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(
            completed=True, safe=True, punctual=False, stop_error_m=5.0
        ),
    )
    callback.num_timesteps = 30
    callback._on_rollout_end()
    assert callback.best.update_reason == "safe_success_reached"
    assert callback.best.parameters["weights"][0] == 3.0

    # 4. better_constraint_fallback: 同样完成但约束违背指标更好（停车误差更小）
    model_params["weights"][0] = 4.0
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(
            completed=True, safe=True, punctual=False, stop_error_m=1.0
        ),
    )
    callback.num_timesteps = 40
    callback._on_rollout_end()
    assert callback.best.update_reason == "better_constraint_fallback"
    assert callback.best.parameters["weights"][0] == 4.0

    # 5. strict_feasibility_reached: 首次达成严格可行
    model_params["weights"][0] = 5.0
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(
            feasible=True,
            completed=True,
            precise_stop=True,
            safe=True,
            punctual=True,
            total_energy_kj=80.0,
        ),
    )
    callback.num_timesteps = 50
    callback._on_rollout_end()
    assert callback.best.update_reason == "strict_feasibility_reached"
    assert callback.best.parameters["weights"][0] == 5.0

    # 6. lower_energy_among_feasible: 可行解中能耗更低
    model_params["weights"][0] = 6.0
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(
            feasible=True,
            completed=True,
            precise_stop=True,
            safe=True,
            punctual=True,
            total_energy_kj=50.0,
        ),
    )
    callback.num_timesteps = 60
    callback._on_rollout_end()
    assert callback.best.update_reason == "lower_energy_among_feasible"
    assert callback.best.parameters["weights"][0] == 6.0

    # 7. 可行解中能耗更高的候选：不触发更新
    model_params["weights"][0] = 7.0
    monkeypatch.setattr(
        ppo_module,
        "assess",
        lambda *a, **k: _make_report(
            feasible=True,
            completed=True,
            precise_stop=True,
            safe=True,
            punctual=True,
            total_energy_kj=70.0,
        ),
    )
    callback.num_timesteps = 70
    callback._on_rollout_end()
    assert callback.best.update_reason == "lower_energy_among_feasible"
    assert callback.best.parameters["weights"][0] == 6.0


def test_completed_episode_stop_callback_drives_shared_progress() -> None:
    progress = CompletedEpisodeProgress(target_episodes=3)
    callback = StopTrainingOnCompletedEpisodes(progress)
    callback.locals = {"dones": np.asarray([True, False])}

    assert callback._on_step() is True
    assert callback.n_episodes == 1
    assert progress.fraction == pytest.approx(1.0 / 3.0)

    callback.locals = {"dones": np.asarray([True, True])}
    assert callback._on_step() is False
    assert callback.n_episodes == 3
    assert progress.fraction == pytest.approx(1.0)


def test_scheduled_evaluation_repeats_at_rollout_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtto.rl.ppo as ppo_module

    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        evaluation_interval_rollouts=12,
    )
    _init(callback, DummyTrainingEnv())
    calls = 0

    def evaluate(*_args: object, **_kwargs: object) -> RLRun:
        nonlocal calls
        calls += 1
        return _dummy_run(reward=5.0)

    monkeypatch.setattr(ppo_module, "run_policy", evaluate)
    monkeypatch.setattr(ppo_module, "assess", lambda *a, **k: _make_report())
    for rollout_index in range(1, 25):
        callback.num_timesteps = rollout_index * 10
        callback._on_rollout_end()
        assert calls == rollout_index // 12


def test_episode_schedule_evaluates_on_next_rollout_start_after_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtto.rl.ppo as ppo_module

    completed = [99]
    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        evaluation_interval_episodes=100,
        get_completed_training_episodes=lambda: completed[0],
        max_completed_episodes_exclusive=5000,
    )
    _init(callback, DummyTrainingEnv())
    calls: list[tuple[int, int]] = []

    def evaluate(*_args: object, **_kwargs: object) -> RLRun:
        calls.append((callback._rollouts_completed, completed[0]))
        return _dummy_run(reward=-1.0)

    monkeypatch.setattr(ppo_module, "run_policy", evaluate)
    monkeypatch.setattr(ppo_module, "assess", lambda *a, **k: _make_report())

    callback._on_rollout_start()
    completed[0] = 107
    callback.num_timesteps = 8192
    callback._on_rollout_end()
    assert calls == []
    callback._on_rollout_start()

    assert calls == [(1, 107)]


def test_episode_schedule_collapses_crossed_thresholds_and_skips_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtto.rl.ppo as ppo_module

    completed = [350]
    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        evaluation_interval_episodes=100,
        get_completed_training_episodes=lambda: completed[0],
        max_completed_episodes_exclusive=5000,
    )
    _init(callback, DummyTrainingEnv())
    calls: list[int] = []

    def evaluate(*_args: object, **_kwargs: object) -> RLRun:
        calls.append(completed[0])
        return _dummy_run(reward=-1.0)

    monkeypatch.setattr(ppo_module, "run_policy", evaluate)
    monkeypatch.setattr(ppo_module, "assess", lambda *a, **k: _make_report())

    callback._on_rollout_start()
    callback._on_rollout_start()
    completed[0] = 5000
    callback._on_rollout_start()
    callback._on_training_end()

    assert calls == [350]
    assert callback._last_scheduled_completed_episodes == 300
    history = callback.history
    np.testing.assert_array_equal(history.scheduled_completed_training_episodes, [300])
    np.testing.assert_array_equal(history.completed_training_episodes, [350])
    np.testing.assert_array_equal(history.training_steps, [0])
    np.testing.assert_array_equal(history.rollout_indices, [0])


def test_scheduled_evaluation_skips_terminal_budget_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtto.rl.ppo as ppo_module

    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        evaluation_interval_rollouts=100,
        max_rollouts_exclusive=500,
    )
    _init(callback, DummyTrainingEnv())
    calls: list[int] = []

    def evaluate(*_args: object, **_kwargs: object) -> RLRun:
        calls.append(callback._rollouts_completed)
        return _dummy_run(reward=5.0)

    monkeypatch.setattr(ppo_module, "run_policy", evaluate)
    monkeypatch.setattr(ppo_module, "assess", lambda *a, **k: _make_report())
    for rollout_index in range(1, 501):
        callback.num_timesteps = rollout_index * 10
        callback._on_rollout_end()

    assert calls == [100, 200, 300, 400]


def test_optional_boundary_evaluations_include_latest_final_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtto.rl.ppo as ppo_module

    completed = [0]
    callback = ScheduledPolicyEvaluationCallback(
        eval_env=DummyEvalEnv(),
        evaluation_interval_rollouts=4,
        get_completed_training_episodes=lambda: completed[0],
        evaluate_at_boundaries=True,
    )
    _init(callback, DummyTrainingEnv())

    def evaluate(*_args: object, **_kwargs: object) -> RLRun:
        return _dummy_run(reward=float(completed[0]))

    monkeypatch.setattr(ppo_module, "run_policy", evaluate)
    monkeypatch.setattr(ppo_module, "assess", lambda *a, **k: _make_report())
    callback.num_timesteps = 0
    callback._on_training_start()
    for index in range(1, 5):
        completed[0] = index
        callback.num_timesteps = index * 10
        callback._on_rollout_end()
    completed[0] = 6
    callback.num_timesteps = 60
    callback._on_training_end()

    history = callback.history
    np.testing.assert_array_equal(history.completed_training_episodes, [0, 4, 6])
    np.testing.assert_array_equal(history.training_steps, [0, 40, 60])
    np.testing.assert_array_equal(history.total_reward, [0, 4, 6])


def test_scheduled_evaluation_rejects_nonpositive_rollout_interval() -> None:
    with pytest.raises(ValueError, match="evaluation_interval_rollouts"):
        _ = ScheduledPolicyEvaluationCallback(
            eval_env=DummyEvalEnv(),
            evaluation_interval_rollouts=0,
        )


def test_evaluation_schedule_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ScheduledPolicyEvaluationCallback(
            eval_env=DummyEvalEnv(),
            evaluation_interval_rollouts=12,
            evaluation_interval_episodes=100,
            get_completed_training_episodes=lambda: 0,
        )


@pytest.mark.parametrize(
    ("start", "target", "final", "expected"),
    (
        (0.0, 100.0, 40.0, 0.4),
        (100.0, 0.0, 60.0, 0.4),
        (0.0, 100.0, -20.0, 0.0),
        (100.0, 0.0, 120.0, 0.0),
        (0.0, 100.0, 130.0, 1.0),
        (100.0, 0.0, -30.0, 1.0),
    ),
)
def test_route_completion_ratio_uses_clipped_net_directional_progress(
    start: float, target: float, final: float, expected: float
) -> None:
    assert calculate_route_completion_ratio(
        start_position_m=start,
        target_position_m=target,
        final_position_m=final,
    ) == pytest.approx(expected)


def test_learning_rate_anneals_by_completed_episodes_not_sb3_progress() -> None:
    from mtto.rl.ppo import (
        CompletedEpisodeProgress,
        completed_episode_cosine_annealing_schedule,
    )

    progress = CompletedEpisodeProgress(target_episodes=5000)
    schedule = completed_episode_cosine_annealing_schedule(progress)
    assert schedule(0.0) == pytest.approx(3e-4)
    assert schedule(1.0) == pytest.approx(3e-4)

    progress.completed_episodes = 2500
    assert schedule(0.0) == pytest.approx((3e-4 + 1e-5) / 2.0)
    progress.completed_episodes = 5000
    assert schedule(1.0) == pytest.approx(1e-5)
    progress.completed_episodes = 6000
    assert schedule(0.5) == pytest.approx(1e-5)


def test_learning_rate_anneals_by_environment_step_progress() -> None:
    from mtto.rl.ppo import environment_step_cosine_annealing_schedule

    schedule = environment_step_cosine_annealing_schedule()
    assert schedule(1.0) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx((3e-4 + 1e-5) / 2.0)
    assert schedule(0.0) == pytest.approx(1e-5)


def test_training_and_evaluation_environments_share_references(
    paper_scenario, paper_task
) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    from mtto.rl.env import make_env
    from mtto.workflows.train import build_env_references

    lookup, normalization = build_env_references(paper_scenario, paper_task)
    initializers = [
        (
            lambda rank=rank: make_env(
                paper_scenario,
                paper_task,
                0.998,
                1.0,
                lookup,
                normalization,
                compact_training_info=True,
                reward_diagnostics_worker_rank=rank,
                reward_diagnostics_rollout_capacity=4,
            )
        )
        for rank in range(2)
    ]
    train = DummyVecEnv(initializers)
    evaluation = make_env(
        paper_scenario,
        paper_task,
        0.998,
        1.0,
        lookup,
        normalization,
    )
    try:
        assert all(env.srtsp_lookup is lookup for env in train.envs)
        assert all(env.normalization is normalization for env in train.envs)
        assert evaluation.srtsp_lookup is lookup
        assert evaluation.normalization is normalization
    finally:
        train.close()
        evaluation.close()


class ActionReplayWrapper(gym.ActionWrapper):
    """Replay a fixed sequence of actions, defaulting to zero if exhausted."""

    def __init__(self, env: gym.Env, actions: np.ndarray) -> None:
        super().__init__(env)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.index = 0

    def action(self, action: np.ndarray) -> np.ndarray:
        if self.index < len(self.actions):
            act = np.asarray([self.actions[self.index]], dtype=np.float32)
            self.index += 1
            return act
        return action


class RewardSpyWrapper(gym.Wrapper):
    """Spy wrapper to record raw float step rewards from the environment."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.raw_rewards: list[float] = []

    def step(self, action: np.ndarray):
        obs, reward, term, trunc, info = self.env.step(action)
        self.raw_rewards.append(float(reward))
        return obs, reward, term, trunc, info


@pytest.mark.parametrize("reason", list(TerminationReason))
def test_domain_failures_terminate_without_timelimit_truncation(
    monkeypatch: pytest.MonkeyPatch, reason: TerminationReason
) -> None:
    env = build_env("basic_safety_punctuality")
    transition = env.transition
    monkeypatch.setattr(
        env,
        "transition",
        lambda *args: replace(transition(*args), termination_reason=reason),
    )
    vec = DummyVecEnv([lambda: env])
    vec.reset()
    _, _, dones, infos = vec.step(np.asarray([[0.5]], dtype=np.float32))
    vec.close()

    assert bool(dones[0]) is True
    assert infos[0]["termination_reason"] == reason.name
    assert not infos[0].get("TimeLimit.truncated", False)


def test_external_timelimit_truncates_with_timelimit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_env("basic_safety_punctuality")
    # Isolate the external cutoff from any domain termination.
    transition = env.transition
    monkeypatch.setattr(
        env,
        "transition",
        lambda *args: replace(transition(*args), termination_reason=None),
    )
    vec = DummyVecEnv([lambda: TimeLimit(env, max_episode_steps=3)])
    vec.reset()

    done = False
    last_info = None
    for _ in range(3):
        _, _, dones, infos = vec.step(np.asarray([[0.5]], dtype=np.float32))
        done = bool(dones[0])
        last_info = infos[0]

    vec.close()
    assert done is True
    assert last_info is not None
    assert last_info.get("TimeLimit.truncated") is True
    assert last_info["termination_reason"] is None


def test_sb3_ppo_rollout_buffer_no_bootstrap_on_domain_failure() -> None:
    # Full braking from standstill stops short on the first step.
    spy = RewardSpyWrapper(
        ActionReplayWrapper(
            build_env("basic_safety_punctuality"), np.full(4, -1.0, dtype=np.float32)
        )
    )
    vec = DummyVecEnv([lambda: spy])
    model = PPO("MlpPolicy", vec, n_steps=4, batch_size=4, seed=42, device="cpu")
    _, callback = model._setup_learn(total_timesteps=4)
    model.collect_rollouts(vec, callback, model.rollout_buffer, 4)
    vec.close()

    raw_failure_reward = spy.raw_rewards[0]
    rollout_step_reward = float(model.rollout_buffer.rewards[0, 0])
    assert rollout_step_reward == pytest.approx(raw_failure_reward, abs=1e-5)
