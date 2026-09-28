"""MTTOEnv construction, transitions, schedule changes and termination reasons."""

from dataclasses import replace
from typing import TypedDict, Unpack

import numpy as np
import pytest
from gymnasium.utils.env_checker import (
    check_env,
)

from mtto.domain.dynamics import calc_levi_deceleration_scalar_numba
from mtto.domain.energy import segment_energy
from mtto.domain.kinematics import Motion
from mtto.domain.line import get_slope_scalar_numba
from mtto.domain.safeguard import build_safeguard
from mtto.domain.scenario import Scenario, ScheduleChange, StopState, Task
from mtto.domain.srtsp import (
    SrtspLookup,
    lookup_upper_speed_or_zero,
    min_remaining_time_s,
)
from mtto.rl.diagnostics import (
    REWARD_SIGNAL_COUNT,
    RewardDiagnosticsAccumulator,
    SafetyTruncationBuffer,
)
from mtto.rl.env import MTTOEnv
from mtto.rl.evaluate import run_policy
from mtto.rl.observation import ObservationBuilder
from mtto.rl.rewards import RewardConfig
from mtto.rl.state import State, StepResult, TerminationReason
from mtto.workflows.train import build_env_references
from tests.golden.drive import build_env


def _result(
    previous: State,
    next_state: State,
    acceleration: float,
    distance: float,
    duration: float,
    energy: float,
    reason: TerminationReason | None,
) -> StepResult:
    return StepResult(
        step_end_state=next_state,
        next_state=next_state,
        commanded_acceleration_mps2=acceleration,
        motion=Motion(next_state.v_mps, distance, duration),
        propulsion_delta_kj=energy,
        levitation_delta_kj=0.0,
        termination_reason=reason,
        violation=None,
    )


class _MTTOEnvOverrides(TypedDict, total=False):
    enable_trajectory_tracking: bool
    safety_truncation_buffer: SafetyTruncationBuffer | None


@pytest.fixture(scope="module")
def mtto_env(paper_scenario):
    custom_safeguard = build_safeguard(
        params=replace(paper_scenario.safeguard.params, factor=0.95),
        line=paper_scenario.line,
        levi_curves=paper_scenario.safeguard.levi_curves,
        brake_curves=paper_scenario.safeguard.brake_curves,
        min_curves=paper_scenario.safeguard.min_curves,
        max_curves=paper_scenario.safeguard.max_curves,
    )
    scenario = replace(paper_scenario, safeguard=custom_safeguard)
    task = Task(
        start_position_m=135.0,
        target_position_m=29270.046,
        schedule_time_s=440.0,
        max_acc_change=0.75,
        max_stop_error_m=2.0,
        max_arr_time_error_s=60.0,
    )

    srtsp_lookup, normalization = build_env_references(scenario, task, 10.0)
    return MTTOEnv(
        scenario=scenario,
        task=task,
        gamma=0.995,
        step_distance=10.0,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        reward_config=RewardConfig(),
    )


def test_punctuality_schedule_change_and_global_context_reference(mtto_env):
    midpoint = (mtto_env.task.start_position_m + mtto_env.task.target_position_m) / 2
    task = replace(
        mtto_env.task,
        schedule_change=ScheduleChange(
            trigger_position_m=mtto_env.task.start_position_m + 10.0,
            new_schedule_time_s=mtto_env.task.schedule_time_s + 20.0,
        ),
    )
    env = _build_env_like(
        mtto_env,
        task=task,
        reward_config=RewardConfig(enable_potential_punctuality=True),
    )
    env.reset()
    calc = env.reward_calculator
    assert calc.potential_punctuality(env.state, env.task) == pytest.approx(0)
    old_reference = calc.reference_punctuality_slack(
        midpoint, env.state.schedule_time_s, env.task
    )
    old_minimum = env.normalization.initial_min_operation_time_s
    action = np.asarray(
        [env.observation_builder.normalize_acc_to_action(0.5)], dtype=np.float32
    )
    _, _, _, _, _ = env.step(action)
    assert env.state.schedule_changed is True
    assert env.state.schedule_time_s == pytest.approx(
        task.schedule_change.new_schedule_time_s
    )
    assert calc.reference_punctuality_slack(
        midpoint, env.state.schedule_time_s, env.task
    ) == pytest.approx(old_reference + 10)
    assert env.normalization.initial_min_operation_time_s == old_minimum
    assert env.state.slack_time_s == pytest.approx(
        env.state.schedule_time_s
        - env.state.t_s
        - min_remaining_time_s(
            env.vehicle,
            env.track,
            env.safeguard.params.factor,
            env.state.s_m,
            env.state.v_mps,
            env.task.target_position_m,
        )
    )


def test_punctuality_gym_and_shared_replay_reward_match(mtto_env):
    env = _build_env_like(
        mtto_env, reward_config=RewardConfig(enable_potential_punctuality=True)
    )
    env.reset()
    acceleration = env.observation_builder.denormalize_action(1.0)
    transition = env.transition(env.state, acceleration)
    expected = env.reward_calculator.calculate(env.state, transition, env.task)
    _, reward, terminated, truncated, _ = env.step(np.asarray([1.0]))
    assert reward == pytest.approx(expected.total)
    assert (terminated, truncated) == (
        transition.termination_reason is not None,
        False,
    )


def test_punctuality_training_adapter_and_ppo_smoke(mtto_env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = _build_env_like(
        mtto_env, reward_config=RewardConfig(enable_potential_punctuality=True)
    )
    vec = DummyVecEnv([lambda: env])
    env.compact_training_info = True
    vec.reset()
    _, _, done, infos = vec.step(np.asarray([[-1.0]]))
    assert done[0]
    assert infos[0]["termination_reason"] is not None
    assert not infos[0].get("TimeLimit.truncated", False)
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        gamma=env.gamma,
        seed=17,
        device="cpu",
        policy_kwargs={"net_arch": [16]},
    )
    model.learn(total_timesteps=32)
    assert model.num_timesteps == 32
    for parameter in model.policy.parameters():
        assert np.isfinite(parameter.detach().numpy()).all()
    assert np.isfinite(model.rollout_buffer.rewards).all()
    assert np.isfinite(model.rollout_buffer.returns).all()
    vec.close()


def test_punctuality_external_time_limit_still_bootstraps(mtto_env, monkeypatch):
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = _build_env_like(
        mtto_env, reward_config=RewardConfig(enable_potential_punctuality=True)
    )
    vec = DummyVecEnv([lambda: TimeLimit(env, max_episode_steps=1)])
    vec.reset()
    # Isolate an external cutoff from this fixture's first-step safety failure.
    transition = env.transition
    monkeypatch.setattr(
        env,
        "transition",
        lambda *args: replace(
            transition(*args),
            termination_reason=None,
        ),
    )
    _, _, done, infos = vec.step(np.asarray([[1.0]]))
    assert done[0]
    assert infos[0]["termination_reason"] is None
    assert infos[0]["TimeLimit.truncated"]
    vec.close()


def _build_env_like(
    source_env: MTTOEnv,
    *,
    scenario: Scenario | None = None,
    task: Task | None = None,
    reward_config: RewardConfig | None = None,
    **kwargs: Unpack[_MTTOEnvOverrides],
) -> MTTOEnv:
    scenario = scenario if scenario is not None else source_env.scenario
    task = task or replace(source_env.task)
    srtsp_lookup, normalization = build_env_references(
        scenario, task, source_env.step_distance
    )
    return MTTOEnv(
        scenario=scenario,
        task=task,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        gamma=source_env.gamma,
        step_distance=source_env.step_distance,
        reward_config=reward_config if reward_config is not None else RewardConfig(),
        **kwargs,
    )


def test_reset(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (12,)
    np.testing.assert_allclose(obs[0], 1.0)  # remaining_distance
    np.testing.assert_allclose(obs[1], 0.0)  # current_speed
    np.testing.assert_allclose(obs[2], 0.0)  # current_acc
    np.testing.assert_allclose(
        obs[3],
        mtto_env.observation_builder.normalize_acc_to_action(
            mtto_env.observation_builder.calc_coasting_acc(mtto_env.state)
        ),
    )  # suggested_dec_normalized
    np.testing.assert_allclose(obs[4], 1.0)  # remaining_schedule_time
    np.testing.assert_allclose(
        obs[5],
        mtto_env.state.slack_time_s / mtto_env.state.schedule_time_s,
    )  # time_redundancy
    np.testing.assert_allclose(
        obs[6],
        mtto_env.state.max_speed_mps / mtto_env.vehicle.max_speed,
    )  # current_max_speed
    np.testing.assert_allclose(
        obs[7],
        mtto_env.state.lower_limit_mps / mtto_env.vehicle.max_speed,
    )  # current_min_speed
    np.testing.assert_allclose(obs[8], 0.0)  # current_slope
    np.testing.assert_allclose(
        obs[9],
        mtto_env.observation_builder.get_lookahead_avg_slope(mtto_env.state.s_m)
        / mtto_env.vehicle.max_slope_capacity,
    )  # lookahead_avg_slope
    np.testing.assert_allclose(
        obs[10],
        mtto_env.observation_builder.get_lookahead_avg_upper_speed(mtto_env.state.s_m)
        / mtto_env.vehicle.max_speed,
    )  # lookahead_avg_upper_speed
    np.testing.assert_allclose(obs[11], 0.0)  # approach_progress
    assert info == {}


def test_multiple_environments_share_references(mtto_env: MTTOEnv) -> None:
    first = MTTOEnv(
        mtto_env.scenario,
        mtto_env.task,
        mtto_env.gamma,
        mtto_env.step_distance,
        mtto_env.srtsp_lookup,
        mtto_env.normalization,
    )
    second = MTTOEnv(
        mtto_env.scenario,
        mtto_env.task,
        mtto_env.gamma,
        mtto_env.step_distance,
        mtto_env.srtsp_lookup,
        mtto_env.normalization,
    )
    assert first.srtsp_lookup is second.srtsp_lookup is mtto_env.srtsp_lookup
    assert first.normalization is second.normalization is mtto_env.normalization
    second_state = second.state
    _ = first.step(np.asarray([0.0], dtype=np.float32))
    assert second.state is second_state


@pytest.mark.parametrize(
    "action, expected_acceleration",
    [(-1.0, "max_dec"), (0.0, None), (1.0, "max_acc")],
)
def test_observation_builder_denormalizes_actions(
    mtto_env: MTTOEnv,
    action: float,
    expected_acceleration: str | None,
) -> None:
    actual = mtto_env.observation_builder.denormalize_action(action)
    expected = (
        (mtto_env.vehicle.max_acc + mtto_env.vehicle.max_dec) / 2.0
        if expected_acceleration is None
        else getattr(mtto_env.vehicle, expected_acceleration)
    )

    assert actual == pytest.approx(expected)
    assert mtto_env.observation_builder.normalize_acc_to_action(
        actual
    ) == pytest.approx(action)


def test_suggested_dec_uses_coasting_acc_outside_final_approach(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    mtto_env.state = replace(mtto_env.state, v_mps=30.0, slope_permille=1.0)

    obs = mtto_env.observation_builder.build(mtto_env.state)

    coasting_dec = calc_levi_deceleration_scalar_numba(
        mtto_env.state.v_mps,
        mtto_env.state.slope_permille,
        mtto_env.vehicle.mass,
        mtto_env.vehicle.numoftrainsets,
    )
    expected_coasting_acc = float(
        np.clip(
            -coasting_dec,
            mtto_env.vehicle.max_dec,
            mtto_env.vehicle.max_acc,
        )
    )
    np.testing.assert_allclose(
        obs[3],
        mtto_env.observation_builder.normalize_acc_to_action(expected_coasting_acc),
    )


def test_suggested_dec_uses_required_stop_dec_in_final_approach(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    mtto_env.state = replace(
        mtto_env.state,
        s_m=mtto_env.task.target_position_m - 1000.0,
        v_mps=20.0,
        slope_permille=1.0,
    )

    obs = mtto_env.observation_builder.build(mtto_env.state)

    required_dec = -(mtto_env.state.v_mps**2) / (2.0 * 1000.0)
    np.testing.assert_allclose(
        obs[3],
        mtto_env.observation_builder.normalize_acc_to_action(required_dec),
    )
    np.testing.assert_allclose(
        obs[11],
        mtto_env.observation_builder.calc_approach_progress(1000.0),
    )  # approach_progress


def test_lookup_upper_speed_or_zero_returns_zero_outside_lut(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    monkeypatch.setattr(
        mtto_env,
        "srtsp_lookup",
        SrtspLookup(100.0, 10.0, np.asarray([0.0, 10.0, 20.0], dtype=np.float32)),
    )

    assert lookup_upper_speed_or_zero(mtto_env.srtsp_lookup, 99.0) == pytest.approx(0.0)
    assert lookup_upper_speed_or_zero(mtto_env.srtsp_lookup, 110.0) == pytest.approx(
        10.0
    )
    assert lookup_upper_speed_or_zero(mtto_env.srtsp_lookup, 121.0) == pytest.approx(
        0.0
    )


def test_lookahead_cache_matches_window_average(mtto_env: MTTOEnv) -> None:
    _ = mtto_env.reset()
    builder = mtto_env.observation_builder
    step_index = 3
    position_m = mtto_env.task.start_position_m + step_index * mtto_env.step_distance
    offsets = np.linspace(
        builder.step_distance_m,
        builder.lookahead_distance_m,
        builder.lookahead_num_samples,
    )
    expected_slope = np.mean(
        [
            get_slope_scalar_numba(
                position_m + float(offset),
                mtto_env.track.slopes,
                mtto_env.track.slope_intervals,
            )
            for offset in offsets
        ]
    )
    expected_upper_speed = np.mean(
        [
            lookup_upper_speed_or_zero(
                mtto_env.srtsp_lookup, position_m + float(offset)
            )
            for offset in offsets
        ]
    )

    np.testing.assert_allclose(
        builder.get_lookahead_avg_slope(position_m), expected_slope
    )
    np.testing.assert_allclose(
        builder.get_lookahead_avg_upper_speed(position_m), expected_upper_speed
    )


def test_lookahead_features_are_read_only_and_not_recomputed_during_build(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    builder = mtto_env.observation_builder

    def fail(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("lookahead feature was recomputed during build")

    monkeypatch.setattr("mtto.rl.observation.get_slope_scalar_numba", fail)
    monkeypatch.setattr(builder, "srtsp_lookup", fail)

    observation = builder.build(mtto_env.state)

    assert observation.shape == (12,)
    assert builder._lookahead_avg_slope_by_step.flags.writeable is False
    assert builder._lookahead_avg_upper_speed_by_step.flags.writeable is False


@pytest.mark.parametrize("position_offset_m", (-1.0, 10**9))
def test_lookahead_cache_rejects_out_of_route_position(
    mtto_env: MTTOEnv,
    position_offset_m: float,
) -> None:
    position_m = mtto_env.task.start_position_m + position_offset_m
    with pytest.raises(ValueError, match="outside the task route"):
        _ = mtto_env.observation_builder.get_lookahead_avg_slope(position_m)


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    mec1, lec1 = segment_energy(
        mtto_env.scenario.energy,
        mtto_env.vehicle,
        mtto_env.track,
        begin_pos=mtto_env.state.s_m,
        begin_speed=mtto_env.state.v_mps,
        acc=0.0,
        distance=0.0,
        direction=1,
        operation_time=0.0,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = segment_energy(
        mtto_env.scenario.energy,
        mtto_env.vehicle,
        mtto_env.track,
        begin_pos=mtto_env.state.s_m,
        begin_speed=mtto_env.state.v_mps,
        acc=1.0,
        distance=100.0,
        direction=1,
        operation_time=14.142,
    )
    energy_consumption2 = mec2 + lec2

    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_whole_env(mtto_env: MTTOEnv):
    check_env(mtto_env)


def test_step_info_excludes_reward_diagnostics(mtto_env: MTTOEnv):
    _ = mtto_env.reset()
    action = mtto_env.action_space.sample()
    _, _, terminated, truncated, info = mtto_env.step(action)

    assert "episode" in info
    assert "outcome" in info
    assert info["outcome"] == {
        "termination_reason": info["termination_reason"],
    }
    assert info["safety_margin_mps"] == pytest.approx(
        min(
            mtto_env.state.max_speed_mps - mtto_env.state.v_mps,
            mtto_env.state.v_mps - mtto_env.state.lower_limit_mps,
        )
    )
    assert "rewards" not in info


def test_compact_training_info_is_empty_and_worker_accumulator_records(
    mtto_env: MTTOEnv,
):
    mtto_env.reward_diagnostics_accumulator = RewardDiagnosticsAccumulator(
        worker_rank=0, rollout_capacity=2
    )
    mtto_env.compact_training_info = True
    try:
        _ = mtto_env.reset()
        action = mtto_env.action_space.sample()
        _, _, _, _, info = mtto_env.step(action)
        assert info == {"termination_reason": mtto_env.outcome.termination_reason}
        batch = mtto_env.drain_reward_diagnostics()
        np.testing.assert_array_equal(batch["transition_count"], [1])
        assert batch["reward_sum"].shape == (REWARD_SIGNAL_COUNT,)
    finally:
        mtto_env.compact_training_info = False
        mtto_env.reward_diagnostics_accumulator = None


def test_no_trajectory_tracking_data_when_disabled(mtto_env: MTTOEnv):
    assert mtto_env.enable_trajectory_tracking is False

    _ = mtto_env.reset()
    action = mtto_env.action_space.sample()
    _ = mtto_env.step(action)

    assert mtto_env.trajectory_pos is None
    assert mtto_env.trajectory_speed_mps is None


def test_trajectory_tracking_can_be_enabled_without_rendering(mtto_env: MTTOEnv):
    mtto_env.enable_trajectory_tracking = True
    try:
        _ = mtto_env.reset()
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 1
        assert len(mtto_env.trajectory_speed_mps) == 1

        action = np.asarray([1.0], dtype=np.float32)
        _ = mtto_env.step(action)

        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 2
        assert len(mtto_env.trajectory_speed_mps) == 2
        assert mtto_env.trajectory_pos[-1] == pytest.approx(float(mtto_env.state.s_m))
        assert mtto_env.trajectory_speed_mps[-1] == pytest.approx(
            abs(float(mtto_env.state.v_mps))
        )
    finally:
        mtto_env.enable_trajectory_tracking = False
        _ = mtto_env.reset()


def _patch_step_dependencies_for_outcome_tests(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    *,
    next_speed: float,
) -> None:
    def _advance(state: State, acceleration: float):
        next_state = replace(
            state,
            v_mps=next_speed,
            commanded_acceleration_mps2=acceleration,
            stop_error_m=abs(mtto_env.task.target_position_m - state.s_m),
            step=state.step + 1,
        )
        success = next_speed == 0.0 and next_state.stop_error_m <= 9.0
        failed_stop = next_speed == 0.0 and not success
        reason = (
            TerminationReason.STOPPED_IN_ZONE
            if success
            else (TerminationReason.STOPPED_SHORT if failed_stop else None)
        )
        return _result(
            state,
            next_state,
            acceleration,
            0.0,
            0.0,
            0.0,
            reason,
        )

    monkeypatch.setattr(mtto_env, "transition", _advance)


def test_step_failed_stop_terminates_with_fixed_penalty(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)
    previous_state = mtto_env.state

    _, reward, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    current_state = mtto_env.state
    expected_safety = mtto_env.reward_calculator.gamma * (
        mtto_env.reward_calculator._potential_safety(
            speed_mps=current_state.v_mps,
            min_speed_mps=current_state.lower_limit_mps,
            max_speed_mps=current_state.max_speed_mps,
        )
    ) - mtto_env.reward_calculator._potential_safety(
        speed_mps=previous_state.v_mps,
        min_speed_mps=previous_state.lower_limit_mps,
        max_speed_mps=previous_state.max_speed_mps,
    )
    assert reward == pytest.approx(-10.0 + expected_safety)
    assert info["outcome"] == {"termination_reason": "STOPPED_SHORT"}
    assert info["termination_reason"] == "STOPPED_SHORT"
    assert "constraint" not in info


def test_step_buffers_speed_safety_truncation_inside_worker(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = SafetyTruncationBuffer()
    env = _build_env_like(mtto_env, safety_truncation_buffer=buffer)
    _ = env.reset()

    def _advance(state: State, acceleration: float):
        next_state = replace(
            state,
            s_m=1234.0,
            commanded_acceleration_mps2=acceleration,
            step=state.step + 1,
        )
        return _result(
            state,
            next_state,
            acceleration,
            0.0,
            0.0,
            0.0,
            TerminationReason.UNDER_LOWER_LIMIT,
        )

    monkeypatch.setattr(env, "transition", _advance)

    _, _, terminated, truncated, info = env.step(np.asarray([0.0], dtype=np.float32))
    batch = env.drain_safety_truncations()

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "UNDER_LOWER_LIMIT"
    assert "safety" not in info
    np.testing.assert_allclose(batch["position_m"], [1234.0])
    np.testing.assert_array_equal(batch["termination_reason"], [4])
    assert env.drain_safety_truncations()["position_m"].size == 0


def test_step_success_is_terminated_without_truncation(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)
    mtto_env.state = replace(
        mtto_env.state,
        s_m=mtto_env.task.target_position_m,
        v_mps=0.0,
        t_s=mtto_env.task.schedule_time_s,
        stop_error_m=0.0,
    )

    _, _, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "STOPPED_IN_ZONE"
    assert info["outcome"] == {"termination_reason": "STOPPED_IN_ZONE"}
    assert "constraint" not in info


def _advance_with_terminal_state(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_error_m: float,
    speed_mps: float,
) -> _result:
    sim = mtto_env
    monkeypatch.setattr(sim, "task", replace(sim.task, max_stop_error_m=0.3))
    state = replace(sim.initial_state(), v_mps=max(speed_mps, 0.0))

    monkeypatch.setattr(
        "mtto.rl.env.run_distance",
        lambda *_args: (speed_mps, sim.step_distance, 1.0),
    )
    monkeypatch.setattr(
        "mtto.rl.env.segment_energy", lambda *_args, **_kwargs: (0.0, 0.0)
    )
    monkeypatch.setattr(
        sim.sps,
        "advance",
        lambda *_args, **_kwargs: state.sps,
    )

    def _build_state(**kwargs: object) -> State:
        return replace(
            state,
            s_m=sim.task.target_position_m - stop_error_m,
            v_mps=speed_mps,
            commanded_acceleration_mps2=float(kwargs["commanded_acceleration_mps2"]),
            t_s=float(kwargs["t_s"]),
            propulsion_energy_kj=float(kwargs["propulsion_energy_kj"]),
            step=int(kwargs["step"]),
            stop_error_m=stop_error_m,
            lower_limit_mps=-100.0,
            upper_limit_mps=100.0,
            srtsp_limit_mps=100.0,
            sps=kwargs["sps"],
        )

    monkeypatch.setattr(sim, "_build_state", _build_state)
    return sim.transition(state, 0.0)


@pytest.mark.parametrize(
    ("stop_error_m", "speed_mps"),
    [(8.999, -0.01), (9.0, 0.0), (9.0, 0.01)],
)
def test_transition_succeeds_within_relaxed_stop_tolerance(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    stop_error_m: float,
    speed_mps: float,
) -> None:
    transition = _advance_with_terminal_state(
        mtto_env,
        monkeypatch,
        stop_error_m=stop_error_m,
        speed_mps=speed_mps,
    )

    assert transition.termination_reason is TerminationReason.STOPPED_IN_ZONE


def test_transition_rejects_stop_outside_relaxed_tolerance(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transition = _advance_with_terminal_state(
        mtto_env,
        monkeypatch,
        stop_error_m=9.001,
        speed_mps=0.0,
    )

    assert transition.termination_reason is TerminationReason.STOPPED_SHORT


@pytest.mark.parametrize(
    ("speed_mps", "expected_reason"),
    [
        (-100.001, TerminationReason.UNDER_LOWER_LIMIT),
        (100.001, TerminationReason.OVER_UPPER_LIMIT),
    ],
)
def test_transition_still_terminates_speed_bound_violations(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    speed_mps: float,
    expected_reason: TerminationReason,
) -> None:
    transition = _advance_with_terminal_state(
        mtto_env,
        monkeypatch,
        stop_error_m=10.0,
        speed_mps=speed_mps,
    )

    assert transition.termination_reason is expected_reason


def test_relaxed_success_does_not_imply_precise_arrival(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transition = _advance_with_terminal_state(
        mtto_env,
        monkeypatch,
        stop_error_m=1.0,
        speed_mps=0.0,
    )

    next_state = transition.next_state
    completed = (
        mtto_env.task.stop_state(next_state.s_m, next_state.v_mps)
        == StopState.STOPPED_IN_ZONE
    )
    precise_arrival = completed and (
        next_state.stop_error_m <= mtto_env.task.max_stop_error_m
    )

    assert completed is True
    assert precise_arrival is False


@pytest.mark.parametrize("step_offset", [0, 1])
def test_transition_does_not_truncate_at_or_over_required_transition_budget(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    step_offset: int,
) -> None:
    sim = mtto_env
    state = replace(
        sim.initial_state(),
        v_mps=10.0,
        step=sim.normalization.required_episode_steps - 1 + step_offset,
    )

    def _build_state(**kwargs: object):
        return replace(
            state,
            s_m=kwargs["s_m"],
            v_mps=10.0,
            commanded_acceleration_mps2=kwargs["commanded_acceleration_mps2"],
            t_s=kwargs["t_s"],
            propulsion_energy_kj=kwargs["propulsion_energy_kj"],
            step=kwargs["step"],
            sps=kwargs["sps"],
            lower_limit_mps=0.0,
            upper_limit_mps=100.0,
            srtsp_limit_mps=100.0,
            stop_error_m=10.0,
        )

    monkeypatch.setattr(sim, "_build_state", _build_state)
    transition = sim.transition(state, 0.0)

    assert transition.next_state.step == (
        sim.normalization.required_episode_steps + step_offset
    )
    assert transition.termination_reason is None


def test_transition_allows_success_on_required_transition_budget(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = mtto_env
    state = replace(
        sim.initial_state(),
        step=sim.normalization.required_episode_steps - 1,
    )

    def _build_state(**kwargs: object):
        return replace(
            state,
            s_m=mtto_env.task.target_position_m,
            v_mps=0.0,
            commanded_acceleration_mps2=kwargs["commanded_acceleration_mps2"],
            t_s=kwargs["t_s"],
            propulsion_energy_kj=kwargs["propulsion_energy_kj"],
            step=kwargs["step"],
            sps=kwargs["sps"],
            lower_limit_mps=0.0,
            upper_limit_mps=100.0,
            srtsp_limit_mps=100.0,
            stop_error_m=0.0,
        )

    monkeypatch.setattr(sim, "_build_state", _build_state)
    transition = sim.transition(state, 0.0)

    assert transition.next_state.step == sim.normalization.required_episode_steps
    assert transition.termination_reason is TerminationReason.STOPPED_IN_ZONE


def test_transition_shortens_final_step_and_rejects_nonzero_terminal_speed(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = mtto_env
    remaining_distance_m = sim.step_distance / 2.0
    state = replace(
        sim.initial_state(),
        s_m=(sim.task.target_position_m - remaining_distance_m),
        v_mps=10.0,
        stop_error_m=remaining_distance_m,
    )

    def _transition(
        speed_mps: float, acceleration_mps2: float, distance_m: float
    ) -> tuple[float, float, float]:
        del speed_mps, acceleration_mps2
        return 10.0, distance_m, distance_m / 10.0

    def _build_state(**kwargs: object) -> State:
        return replace(
            state,
            s_m=float(kwargs["s_m"]),
            v_mps=10.0,
            t_s=float(kwargs["t_s"]),
            propulsion_energy_kj=float(kwargs["propulsion_energy_kj"]),
            step=int(kwargs["step"]),
            stop_error_m=0.0,
            lower_limit_mps=0.0,
            upper_limit_mps=100.0,
            srtsp_limit_mps=100.0,
            sps=kwargs["sps"],
        )

    monkeypatch.setattr("mtto.rl.env.run_distance", _transition)
    monkeypatch.setattr(
        "mtto.rl.env.segment_energy", lambda *_args, **_kwargs: (0.0, 0.0)
    )
    monkeypatch.setattr(
        sim.sps,
        "advance",
        lambda *_args, **_kwargs: state.sps,
    )
    monkeypatch.setattr(sim, "_build_state", _build_state)

    transition = sim.transition(state, 0.0)

    assert transition.motion.distance_m == pytest.approx(remaining_distance_m)
    assert transition.next_state.s_m == pytest.approx(sim.task.target_position_m)
    assert transition.termination_reason is TerminationReason.OVERRAN


def test_terminal_punctuality_reward_favors_smaller_time_error(
    mtto_env: MTTOEnv,
) -> None:
    terminal_state = replace(
        mtto_env.state,
        s_m=mtto_env.task.target_position_m,
        v_mps=0.0,
        stop_error_m=0.0,
        t_s=mtto_env.task.schedule_time_s,
    )
    punctual_reward = mtto_env.reward_calculator.calculate(
        mtto_env.state,
        _result(
            mtto_env.state,
            terminal_state,
            0.0,
            0.0,
            0.0,
            0.0,
            TerminationReason.STOPPED_IN_ZONE,
        ),
        mtto_env.task,
    )
    late_state = replace(
        terminal_state,
        t_s=terminal_state.t_s + 60.0,
    )
    late_reward = mtto_env.reward_calculator.calculate(
        terminal_state,
        _result(
            terminal_state,
            late_state,
            0.0,
            0.0,
            0.0,
            0.0,
            TerminationReason.STOPPED_IN_ZONE,
        ),
        mtto_env.task,
    )

    assert punctual_reward.terminal_punctuality > late_reward.terminal_punctuality


def test_schedule_change_updates_time_context_without_dp_reference(
    mtto_env: MTTOEnv,
) -> None:
    task = replace(
        mtto_env.task,
        schedule_change=ScheduleChange(
            trigger_position_m=mtto_env.task.start_position_m + 10.0,
            new_schedule_time_s=445.0,
        ),
    )
    env = _build_env_like(mtto_env, task=task)
    _ = env.reset()
    assert env.state.schedule_time_s == pytest.approx(mtto_env.task.schedule_time_s)
    assert env.state.schedule_changed is False

    action = np.asarray(
        [env.observation_builder.normalize_acc_to_action(0.5)], dtype=np.float32
    )
    observation, _, _, _, _ = env.step(action)

    assert env.state.schedule_time_s == pytest.approx(445.0)
    assert env.state.schedule_changed is True
    assert observation.shape == env.observation_space.shape


def test_external_rollout_uses_same_transition_and_reward_path(
    mtto_env: MTTOEnv,
) -> None:
    action = np.asarray([-1.0], dtype=np.float32)
    _ = mtto_env.reset()
    _, env_reward, env_terminated, env_truncated, _ = mtto_env.step(action)

    def _policy(_obs: object) -> np.ndarray:
        return action

    run = run_policy(_policy, mtto_env)

    # From standstill, maximum braking produces the same immediate failed-stop
    # termination in both execution paths.
    assert run.steps == 1
    assert run.total_reward == pytest.approx(env_reward)
    assert run.termination_reason is TerminationReason.STOPPED_SHORT
    assert run.termination_reason.name == mtto_env.outcome.termination_reason
    assert run.termination_reason is not TerminationReason.STOPPED_IN_ZONE


def test_observation_builder_supports_out_buffer(mtto_env: MTTOEnv) -> None:
    _ = mtto_env.reset()
    builder = mtto_env.observation_builder
    custom_buf = np.zeros(ObservationBuilder.OBSERVATION_DIM, dtype=np.float32)

    ret = builder.build(mtto_env.state, out=custom_buf)

    assert ret is custom_buf
    assert custom_buf.shape == (ObservationBuilder.OBSERVATION_DIM,)
    assert np.any(custom_buf != 0.0)


def test_observation_builder_returns_copy_when_out_is_none(
    mtto_env: MTTOEnv,
) -> None:
    _ = mtto_env.reset()
    builder = mtto_env.observation_builder

    obs1 = builder.build(mtto_env.state)
    assert obs1 is not builder._obs_buffer
    obs1_snapshot = obs1.copy()

    # Updating the internal scratch buffer must not mutate an earlier return.
    mtto_env.state = replace(mtto_env.state, v_mps=50.0)
    obs2 = builder.build(mtto_env.state)
    assert obs2 is not builder._obs_buffer
    assert obs1 is not obs2
    np.testing.assert_array_equal(obs1, obs1_snapshot)
    assert obs2[1] == pytest.approx(50.0 / mtto_env.vehicle.max_speed)


def test_env_reset_and_step_return_observation_copies(mtto_env: MTTOEnv) -> None:
    obs_reset, _ = mtto_env.reset()
    assert obs_reset is not mtto_env._observation_buffer
    reset_snapshot = obs_reset.copy()

    obs_step, _, _, _, _ = mtto_env.step(np.asarray([0.0], dtype=np.float32))
    assert obs_step is not mtto_env._observation_buffer
    assert obs_reset is not obs_step
    np.testing.assert_array_equal(obs_reset, reset_snapshot)


@pytest.mark.parametrize(
    ("initial_speed", "command", "expected_distance", "expected_speed"),
    [
        (10.0, 5e-7, 10.0, 10.0),
        (0.0, -0.5, 0.0, 0.0),
        (2.0, -1.0, 2.0, 0.0),
    ],
)
def test_commanded_acceleration_stays_distinct_from_motion(
    mtto_env: MTTOEnv,
    initial_speed: float,
    command: float,
    expected_distance: float,
    expected_speed: float,
) -> None:
    previous = replace(mtto_env.initial_state(), v_mps=initial_speed)
    result = mtto_env.transition(previous, command)
    assert result.commanded_acceleration_mps2 == command
    assert result.step_end_state.commanded_acceleration_mps2 == command
    assert result.motion.distance_m == pytest.approx(expected_distance)
    assert result.motion.v1_mps == pytest.approx(expected_speed)
    if expected_distance == 0.0:
        assert result.motion.duration_s == 0.0
        assert (
            result.propulsion_delta_kj,
            result.levitation_delta_kj,
        ) == segment_energy(
            mtto_env.scenario.energy,
            mtto_env.vehicle,
            mtto_env.track,
            begin_pos=previous.s_m,
            begin_speed=previous.v_mps,
            acc=command,
            distance=0.0,
            direction=1,
            operation_time=0.0,
        )


@pytest.fixture(scope="module")
def base_components(paper_scenario):
    safeguard = build_safeguard(
        params=replace(paper_scenario.safeguard.params, factor=0.95),
        line=paper_scenario.line,
        levi_curves=paper_scenario.safeguard.levi_curves,
        brake_curves=paper_scenario.safeguard.brake_curves,
        min_curves=paper_scenario.safeguard.min_curves,
        max_curves=paper_scenario.safeguard.max_curves,
    )
    return replace(paper_scenario, safeguard=safeguard)


def _env(scenario, trigger: float | None, new_time: float = 480.0) -> MTTOEnv:
    task = Task(
        start_position_m=135.0,
        target_position_m=29270.046,
        schedule_time_s=440.0,
        max_acc_change=0.75,
        max_stop_error_m=2.0,
        max_arr_time_error_s=60.0,
        schedule_change=(
            ScheduleChange(trigger, new_time) if trigger is not None else None
        ),
    )
    lookup, normalization = build_env_references(scenario, task, 10.0)
    return MTTOEnv(scenario, task, 0.995, 10.0, lookup, normalization)


@pytest.mark.parametrize("new_time", [420.0, 460.0, 500.0])
def test_schedule_change_start_point_trigger(base_components, new_time):
    env = _env(base_components, 135.0, new_time)
    state = env.initial_state()
    assert state.schedule_time_s == pytest.approx(new_time)
    assert state.schedule_changed is True


@pytest.mark.parametrize("delta_time_s", [-20.0, 20.0, 50.0])
def test_schedule_change_mid_run_crossing(base_components, delta_time_s):
    env = _env(base_components, 145.0, 440.0 + delta_time_s)
    before = env.initial_state()
    assert before.schedule_time_s == pytest.approx(440.0)
    assert before.schedule_changed is False
    result = env.transition(before, 0.5)
    assert result.step_end_state.schedule_time_s == pytest.approx(440.0)
    assert result.step_end_state.schedule_changed is False
    assert result.next_state.schedule_time_s == pytest.approx(440.0 + delta_time_s)
    assert result.next_state.schedule_changed is True
    reward = env.reward_calculator.calculate(before, result, env.task)
    old_result = replace(result, next_state=result.step_end_state)
    assert reward == env.reward_calculator.calculate(before, old_result, env.task)
    following = env.transition(result.next_state, 0.5)
    assert following.next_state.schedule_time_s == pytest.approx(440.0 + delta_time_s)
    assert following.next_state.schedule_changed is True


@pytest.mark.parametrize(
    "reason",
    list(TerminationReason),
)
def test_schedule_change_not_applied_on_termination(
    base_components, monkeypatch, reason
):
    env = _env(base_components, 145.0)
    before = env.initial_state()
    build_state = env._build_state

    def terminal_state(**kwargs):
        state = build_state(**kwargs)
        if reason is TerminationReason.STOPPED_IN_ZONE:
            return replace(state, s_m=env.task.target_position_m, v_mps=0.0)
        if reason is TerminationReason.STOPPED_SHORT:
            return replace(state, v_mps=0.0)
        if reason is TerminationReason.OVERRAN:
            return replace(state, s_m=env.task.target_position_m, v_mps=10.0)
        if reason is TerminationReason.UNDER_LOWER_LIMIT:
            return replace(state, lower_limit_mps=state.v_mps + 1.0)
        if reason is TerminationReason.OVER_UPPER_LIMIT:
            return replace(state, upper_limit_mps=state.v_mps - 1.0)
        return replace(state, srtsp_limit_mps=state.v_mps - 1.0)

    monkeypatch.setattr(env, "_build_state", terminal_state)
    result = env.transition(before, 0.5)
    assert result.termination_reason is reason
    assert result.next_state is result.step_end_state
    assert result.next_state.schedule_time_s == pytest.approx(440.0)
    assert result.next_state.schedule_changed is False


@pytest.mark.parametrize(
    ("offset", "applied"),
    [(0.0, False), (5.0, True), (10.0, True), (10.001, False)],
)
def test_schedule_change_boundary_conditions(base_components, offset, applied):
    env = _env(base_components, 135.0 + offset)
    before = env.initial_state()
    if offset == 0.0:
        before = replace(before, schedule_time_s=440.0, schedule_changed=False)
    result = env.transition(before, 0.5)
    assert result.next_state.schedule_changed is applied
    assert result.next_state.schedule_time_s == pytest.approx(
        480.0 if applied else 440.0
    )
    if not applied:
        assert result.next_state is result.step_end_state


def test_env_rejects_task_without_schedule_time(base_components):
    task = Task(
        start_position_m=135.0,
        target_position_m=29270.046,
        schedule_time_s=None,
        max_acc_change=0.75,
        max_stop_error_m=2.0,
        max_arr_time_error_s=60.0,
    )
    reference = _env(base_components, None)
    with pytest.raises(ValueError, match="schedule_time_s"):
        MTTOEnv(
            base_components,
            task,
            0.995,
            10.0,
            reference.srtsp_lookup,
            reference.normalization,
        )


@pytest.fixture(scope="module")
def env():
    env = build_env("basic_safety_punctuality")
    return env


def _advance_with_state(
    env,
    next_state: State,
    monkeypatch: pytest.MonkeyPatch,
):
    state = env.initial_state()
    monkeypatch.setattr(
        "mtto.rl.env.run_distance",
        lambda *_args: (next_state.v_mps, env.step_distance, 1.0),
    )
    monkeypatch.setattr(
        "mtto.rl.env.segment_energy", lambda *_args, **_kwargs: (0.0, 0.0)
    )
    monkeypatch.setattr(env.sps, "advance", lambda *_args, **_kwargs: next_state.sps)
    monkeypatch.setattr(env, "_build_state", lambda **_kwargs: next_state)
    return env.transition(state, 0.0)


@pytest.mark.parametrize(
    ("pos_offset", "speed_mps", "overrides", "expected_reason"),
    [
        # 1. STOPPED_IN_ZONE: train stopped within stopping zone
        (0.0, 0.0, {}, TerminationReason.STOPPED_IN_ZONE),
        # 2. STOPPED_SHORT: train stopped short of stopping zone
        (-50.0, 0.0, {}, TerminationReason.STOPPED_SHORT),
        # 3. OVERRAN: target position reached but train is not stopped
        (0.0, 15.0, {}, TerminationReason.OVERRAN),
        # 4. UNDER_LOWER_LIMIT: train speed below lower limit
        (
            -5000.0,
            5.0,
            {"lower_limit_mps": 10.0},
            TerminationReason.UNDER_LOWER_LIMIT,
        ),
        # 5. OVER_UPPER_LIMIT: train speed above safeguard upper limit
        (
            -5000.0,
            60.0,
            {
                "lower_limit_mps": 0.0,
                "upper_limit_mps": 50.0,
                "srtsp_limit_mps": 55.0,
            },
            TerminationReason.OVER_UPPER_LIMIT,
        ),
        # 6. OVER_SRTSP: train speed above SRTSP limit while <= safeguard limit
        (
            -5000.0,
            55.0,
            {
                "lower_limit_mps": 0.0,
                "upper_limit_mps": 60.0,
                "srtsp_limit_mps": 50.0,
            },
            TerminationReason.OVER_SRTSP,
        ),
        # Normal step: in motion within all limits, not at target
        (
            -5000.0,
            25.0,
            {
                "lower_limit_mps": 0.0,
                "upper_limit_mps": 50.0,
                "srtsp_limit_mps": 50.0,
            },
            None,
        ),
    ],
)
def test_individual_termination_reasons(
    env,
    monkeypatch: pytest.MonkeyPatch,
    pos_offset: float,
    speed_mps: float,
    overrides: dict[str, float],
    expected_reason: TerminationReason | None,
):
    target_pos = env.task.target_position_m
    initial_state = env.initial_state()
    base_state = env._build_state(
        s_m=target_pos + pos_offset,
        v_mps=speed_mps,
        commanded_acceleration_mps2=0.0,
        t_s=100.0,
        propulsion_energy_kj=100.0,
        levitation_energy_kj=0.0,
        step=10,
        sps=initial_state.sps,
        schedule_time_s=initial_state.schedule_time_s,
        schedule_changed=initial_state.schedule_changed,
    )
    if overrides:
        base_state = replace(base_state, **overrides)

    transition = _advance_with_state(env, base_state, monkeypatch)
    assert transition.termination_reason is expected_reason


@pytest.mark.parametrize(
    ("description", "pos_offset", "speed_mps", "overrides", "expected_reason"),
    [
        (
            "stopped_in_zone takes precedence over low speed violation",
            0.0,
            0.0,
            {"lower_limit_mps": 5.0},
            TerminationReason.STOPPED_IN_ZONE,
        ),
        (
            "stopped_short takes precedence over low speed violation",
            -50.0,
            0.0,
            {"lower_limit_mps": 5.0},
            TerminationReason.STOPPED_SHORT,
        ),
        (
            "overran takes precedence over guard upper limit violation",
            0.0,
            60.0,
            {
                "upper_limit_mps": 50.0,
                "srtsp_limit_mps": 55.0,
            },
            TerminationReason.OVERRAN,
        ),
        (
            "overran takes precedence over srtsp limit violation",
            0.0,
            55.0,
            {
                "upper_limit_mps": 60.0,
                "srtsp_limit_mps": 50.0,
            },
            TerminationReason.OVERRAN,
        ),
        (
            "over_upper_limit takes precedence when exceeding both guard and srtsp",
            -5000.0,
            65.0,
            {
                "upper_limit_mps": 50.0,
                "srtsp_limit_mps": 60.0,
            },
            TerminationReason.OVER_UPPER_LIMIT,
        ),
    ],
)
def test_termination_priority_resolution(
    env,
    monkeypatch: pytest.MonkeyPatch,
    description: str,
    pos_offset: float,
    speed_mps: float,
    overrides: dict[str, float],
    expected_reason: TerminationReason,
):
    del description
    target_pos = env.task.target_position_m
    initial_state = env.initial_state()
    base_state = env._build_state(
        s_m=target_pos + pos_offset,
        v_mps=speed_mps,
        commanded_acceleration_mps2=0.0,
        t_s=100.0,
        propulsion_energy_kj=100.0,
        levitation_energy_kj=0.0,
        step=10,
        sps=initial_state.sps,
        schedule_time_s=initial_state.schedule_time_s,
        schedule_changed=initial_state.schedule_changed,
    )
    if overrides:
        base_state = replace(base_state, **overrides)

    transition = _advance_with_state(env, base_state, monkeypatch)
    assert transition.termination_reason is expected_reason
