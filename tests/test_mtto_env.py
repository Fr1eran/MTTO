from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo, calc_levi_deceleration_scalar_numba
from rl import env_factory
from rl.evaluation import evaluate_operational_policy_once, evaluate_policy_once
from rl.mtto_env import MTTOEnv, RewardConfig
from rl.operational_state import OperationalTransition, ViolationCode
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)
@pytest.fixture(scope="module")
def mtto_env():
    # 读取线路数据
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
    longyang_start_position, putong_end_position = load_stations_goal_positions()
    levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
        load_safeguard_curves(
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        )
    )

    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.95,
    )

    track = TrackInfo(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )

    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    train_service = TrainService(
        start_position=longyang_start_position,
        start_speed=0.0,
        target_position=putong_end_position,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=60.0,
        max_stop_error=2.0,
    )

    maglevttoenv = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=0.995,
        max_step_distance=10.0,
        reward_config=RewardConfig(),
    )
    return maglevttoenv


def _clone_train_service(train_service: TrainService) -> TrainService:
    return TrainService(
        start_position=train_service.start_position,
        start_speed=train_service.start_speed,
        target_position=train_service.target_position,
        schedule_time=train_service.schedule_time,
        max_acc_change=train_service.max_acc_change,
        max_arr_time_error_ratio=train_service.max_arr_time_error_ratio,
        max_stop_error=train_service.max_stop_error,
    )


def _build_env_like(
    source_env: MTTOEnv,
    *,
    train_service: TrainService | None = None,
    reward_config: RewardConfig | None = None,
    **kwargs,
) -> MTTOEnv:
    return MTTOEnv(
        vehicle=source_env.vehicle,
        track=source_env.track,
        safeguard_utility=source_env.safeguard_utility,
        train_service=train_service or _clone_train_service(source_env.train_service),
        gamma=source_env.gamma,
        max_step_distance=source_env.max_step_distance,
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
        obs[3], mtto_env.observation_builder.normalize_acc_to_action(
            mtto_env.observation_builder.calc_coasting_acc(mtto_env.state)
        )
    )  # suggested_dec_normalized
    np.testing.assert_allclose(obs[4], 1.0)  # remaining_schedule_time
    np.testing.assert_allclose(
        obs[5],
        mtto_env.state.redundant_operation_time_s
        / mtto_env.train_service.schedule_time,
    )  # time_redundancy
    np.testing.assert_allclose(
        obs[6],
        mtto_env.state.max_speed_mps / mtto_env.vehicle.max_speed,
    )  # current_max_speed
    np.testing.assert_allclose(
        obs[7],
        mtto_env.state.min_speed_mps / mtto_env.vehicle.max_speed,
    )  # current_min_speed
    np.testing.assert_allclose(obs[8], 0.0)  # current_slope
    np.testing.assert_allclose(
        obs[9],
        mtto_env.observation_builder.calc_lookahead_avg_slope(mtto_env.state)
        / mtto_env.vehicle.max_slope_capacity,
    )  # lookahead_avg_slope
    np.testing.assert_allclose(
        obs[10],
        mtto_env.observation_builder.calc_lookahead_avg_upper_speed(mtto_env.state)
        / mtto_env.vehicle.max_speed,
    )  # lookahead_avg_upper_speed
    np.testing.assert_allclose(obs[11], 0.0)  # approach_progress
    assert info == {}


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
    assert mtto_env.observation_builder.normalize_acc_to_action(actual) == pytest.approx(
        action
    )


@pytest.mark.parametrize(
    "member",
    [
        "current_position",
        "current_speed",
        "current_acc",
        "current_operation_time",
        "current_redundant_operation_time",
        "current_energy_consumption",
        "current_slope",
        "current_min_speed",
        "current_max_speed",
        "current_stopping_point_index",
        "current_steps",
        "stop_error",
        "sps",
        "ors",
        "ecc",
        "whole_distance",
        "direction",
        "max_episode_steps",
        "required_episode_steps",
        "max_energy_consumption",
        "min_operation_time",
        "max_redundant_operation_time",
        "_upper_speed_lut_pos_min",
        "_upper_speed_lut_step",
        "_upper_speed_lut_speed_arr",
        "_get_action_denormalized",
        "_update_motion",
        "_get_upper_speed",
        "_get_upper_speed_or_zero",
        "_normalize_acc_to_action",
        "_calc_coasting_acc",
        "_calc_lookahead_avg_slope",
        "_calc_lookahead_avg_upper_speed",
        "_calc_approach_progress",
        "_get_reward_goal",
        "_get_obs",
        "_last_violation_code",
    ],
)
def test_mtto_env_does_not_expose_removed_compatibility_members(
    mtto_env: MTTOEnv,
    member: str,
) -> None:
    assert not hasattr(mtto_env, member)


@pytest.mark.parametrize(
    "member",
    [
        "VIOLATION_CODE_ONGOING",
        "VIOLATION_CODE_FAILED_STOP",
        "VIOLATION_CODE_SPEED_LOW",
        "VIOLATION_CODE_SPEED_HIGH",
        "VIOLATION_CODE_STEP_LIMIT",
    ],
)
def test_mtto_env_does_not_expose_legacy_violation_constants(member: str) -> None:
    assert not hasattr(MTTOEnv, member)


def test_suggested_dec_uses_coasting_acc_outside_final_approach(mtto_env: MTTOEnv):
    mtto_env.reset()
    mtto_env.state = replace(mtto_env.state, speed_mps=30.0, slope_permille=1.0)

    obs = mtto_env.observation_builder.build(mtto_env.state)

    coasting_dec = calc_levi_deceleration_scalar_numba(
        mtto_env.state.speed_mps,
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
    mtto_env.reset()
    mtto_env.state = replace(
        mtto_env.state,
        position_m=mtto_env.train_service.target_position - 1000.0,
        speed_mps=20.0,
        slope_permille=1.0,
    )

    obs = mtto_env.observation_builder.build(mtto_env.state)

    required_dec = -(mtto_env.state.speed_mps**2) / (2.0 * 1000.0)
    np.testing.assert_allclose(
        obs[3],
        mtto_env.observation_builder.normalize_acc_to_action(required_dec),
    )
    np.testing.assert_allclose(
        obs[11],
        mtto_env.observation_builder.calc_approach_progress(1000.0),
    )  # approach_progress


def test_get_upper_speed_or_zero_returns_zero_outside_lut(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.reset()
    monkeypatch.setattr(mtto_env.stepper, "_upper_speed_lut_pos_min", 100.0)
    monkeypatch.setattr(mtto_env.stepper, "_upper_speed_lut_step", 10.0)
    monkeypatch.setattr(
        mtto_env.stepper,
        "_upper_speed_lut_speed_arr",
        np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
    )

    assert mtto_env.stepper.get_upper_speed_or_zero(99.0) == pytest.approx(0.0)
    assert mtto_env.stepper.get_upper_speed_or_zero(110.0) == pytest.approx(10.0)
    assert mtto_env.stepper.get_upper_speed_or_zero(121.0) == pytest.approx(0.0)


def test_lookahead_avg_upper_speed_uses_window_average(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.reset()
    mtto_env.state = replace(mtto_env.state, position_m=0.0, speed_mps=5.0)
    monkeypatch.setattr(mtto_env.stepper, "_upper_speed_lut_pos_min", 0.0)
    monkeypatch.setattr(mtto_env.stepper, "_upper_speed_lut_step", 10.0)
    monkeypatch.setattr(
        mtto_env.stepper,
        "_upper_speed_lut_speed_arr",
        np.asarray([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32),
    )

    avg_upper_speed = mtto_env.observation_builder.calc_lookahead_avg_upper_speed(
        mtto_env.state,
        lookahead_distance_m=100.0,
        num_samples=2,
    )

    np.testing.assert_allclose(avg_upper_speed, 10.0)


def test_lookahead_avg_slope_uses_window_average(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.reset()
    mtto_env.state = replace(mtto_env.state, position_m=0.0)
    monkeypatch.setattr(mtto_env.track, "slopes", np.asarray([0.0, 4.0]))
    monkeypatch.setattr(mtto_env.track, "slope_intervals", np.asarray([0.0, 50.0]))

    avg_slope = mtto_env.observation_builder.calc_lookahead_avg_slope(
        mtto_env.state,
        lookahead_distance_m=100.0,
        num_samples=2,
    )

    np.testing.assert_allclose(avg_slope, 2.0)


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    mec1, lec1 = mtto_env.stepper.ecc.calc_energy(
        begin_pos=mtto_env.state.position_m,
        begin_speed=mtto_env.state.speed_mps,
        acc=0.0,
        distance=0.0,
        direction=mtto_env.stepper.direction,
        operation_time=0.0,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = mtto_env.stepper.ecc.calc_energy(
        begin_pos=mtto_env.state.position_m,
        begin_speed=mtto_env.state.speed_mps,
        acc=1.0,
        distance=100.0,
        direction=mtto_env.stepper.direction,
        operation_time=14.142,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption2 = mec2 + lec2

    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


























def test_whole_env(mtto_env: MTTOEnv):
    check_env(mtto_env)


def test_step_without_diagnostics_keeps_tb_dicts_empty(mtto_env: MTTOEnv):
    mtto_env.enable_diagnostics = False
    try:
        mtto_env.reset()
        action = mtto_env.action_space.sample()
        _, _, terminated, truncated, info = mtto_env.step(action)

        assert mtto_env.rewards_info == {}
        assert mtto_env.constraint_info == {}
        assert mtto_env.event_info == {}

        expected_runtime_keys = {
            "energy_consumption",
            "operation_time",
            "redundant_operation_time",
            "position",
            "stopping_point_index",
        }
        runtime_namespace = "basic" if "basic" in info else "runtime"
        assert runtime_namespace in info
        runtime = info[runtime_namespace]
        assert isinstance(runtime, dict)
        assert expected_runtime_keys.issubset(set(runtime.keys()))
        assert expected_runtime_keys.issubset(set(mtto_env.basic_info.keys()))
        assert info["outcome"] == {
            "terminated": terminated,
            "truncated": truncated,
        }
        assert "rewards" not in info
        assert "state" not in info
        assert "constraint" not in info
        assert "event" not in info
        assert "tb_diagnostics" not in info
    finally:
        mtto_env.enable_diagnostics = True


def test_step_with_diagnostics_puts_namespaces_at_info_top_level(
    mtto_env: MTTOEnv,
):
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1

    mtto_env.reset()
    action = mtto_env.action_space.sample()
    _, _, _, _, info = mtto_env.step(action)

    assert "tb_diagnostics" not in info
    runtime_namespace = "basic" if "basic" in info else "runtime"
    assert runtime_namespace in info
    assert "outcome" in info
    assert "rewards" in info
    assert "state" not in info
    assert "constraint" in info
    assert "event" in info
    assert set(info["constraint"]) == {
        "margin_to_vmax_mps",
        "margin_to_vmin_mps",
        "violation_code",
        "speed_limit_segment",
    }

    runtime = info[runtime_namespace]
    assert isinstance(runtime, dict)
    assert "position" in runtime
    assert "stopping_position" not in runtime

    expected_runtime_keys = {
        "energy_consumption",
        "operation_time",
        "position",
        "stopping_point_index",
    }
    assert expected_runtime_keys.issubset(set(runtime.keys()))


def test_no_trajectory_tracking_data_when_disabled(mtto_env: MTTOEnv):
    assert mtto_env.render_mode is None
    assert mtto_env.enable_trajectory_tracking is False

    mtto_env.reset()
    action = mtto_env.action_space.sample()
    mtto_env.step(action)

    assert mtto_env.trajectory_pos is None
    assert mtto_env.trajectory_speed_mps is None


def test_trajectory_tracking_can_be_enabled_without_rendering(mtto_env: MTTOEnv):
    mtto_env.enable_trajectory_tracking = True
    try:
        mtto_env.reset()
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 1
        assert len(mtto_env.trajectory_speed_mps) == 1

        action = np.asarray([1.0], dtype=np.float32)
        mtto_env.step(action)

        assert mtto_env.render_mode is None
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) == 2
        assert len(mtto_env.trajectory_speed_mps) == 2
        assert mtto_env.trajectory_pos[-1] == pytest.approx(
            float(mtto_env.state.position_m)
        )
        assert mtto_env.trajectory_speed_mps[-1] == pytest.approx(
            abs(float(mtto_env.state.speed_mps))
        )
    finally:
        mtto_env.enable_trajectory_tracking = False
        mtto_env.reset()


def _patch_step_dependencies_for_outcome_tests(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
    *,
    next_speed: float,
) -> None:
    def _advance(state, acceleration):
        next_state = replace(
            state,
            speed_mps=next_speed,
            acceleration_mps2=acceleration,
            stop_error_m=abs(mtto_env.train_service.target_position - state.position_m),
            step_count=state.step_count + 1,
        )
        success = next_speed == 0.0 and next_state.stop_error_m <= 9.0
        failed_stop = next_speed == 0.0 and not success
        return OperationalTransition(
            state, next_state, acceleration, 0.0, 0.0, 0.0,
            success, failed_stop,
            ViolationCode.ONGOING if success else ViolationCode.FAILED_STOP,
        )

    monkeypatch.setattr(mtto_env.stepper, "advance", _advance)


def test_step_failed_stop_is_truncated_with_fixed_penalty(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1
    mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)

    _, reward, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is False
    assert truncated is True
    assert reward == pytest.approx(-2.0)
    assert info["outcome"] == {"terminated": False, "truncated": True}
    assert "constraint" in info
    assert info["constraint"]["violation_code"] == int(ViolationCode.FAILED_STOP)


def test_step_success_is_terminated_without_truncation(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1
    mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)
    mtto_env.state = replace(
        mtto_env.state,
        position_m=mtto_env.train_service.target_position,
        speed_mps=0.0,
        operation_time_s=mtto_env.train_service.schedule_time,
        stop_error_m=0.0,
    )

    _, _, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert "constraint" in info
    assert info["constraint"]["violation_code"] == int(ViolationCode.ONGOING)


def test_stepper_truncates_at_required_transition_budget(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stepper = mtto_env.stepper
    state = replace(
        stepper.reset(),
        speed_mps=10.0,
        step_count=stepper.required_episode_steps - 1,
    )

    def _build_state(**kwargs):
        return replace(
            state,
            position_m=kwargs["position_m"],
            speed_mps=10.0,
            acceleration_mps2=kwargs["acceleration_mps2"],
            operation_time_s=kwargs["operation_time_s"],
            energy_consumption_kj=kwargs["energy_consumption_kj"],
            step_count=kwargs["step_count"],
            sps_state=kwargs["sps_state"],
            min_speed_mps=0.0,
            max_speed_mps=100.0,
            stop_error_m=10.0,
        )

    monkeypatch.setattr(stepper, "_build_state", _build_state)
    transition = stepper.advance(state, 0.0)

    assert transition.next_state.step_count == stepper.required_episode_steps
    assert transition.terminated is False
    assert transition.truncated is True
    assert transition.violation_code is ViolationCode.STEP_LIMIT


def test_stepper_allows_success_on_required_transition_budget(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stepper = mtto_env.stepper
    state = replace(
        stepper.reset(),
        step_count=stepper.required_episode_steps - 1,
    )

    def _build_state(**kwargs):
        return replace(
            state,
            position_m=mtto_env.train_service.target_position,
            speed_mps=0.0,
            acceleration_mps2=kwargs["acceleration_mps2"],
            operation_time_s=kwargs["operation_time_s"],
            energy_consumption_kj=kwargs["energy_consumption_kj"],
            step_count=kwargs["step_count"],
            sps_state=kwargs["sps_state"],
            min_speed_mps=0.0,
            max_speed_mps=100.0,
            stop_error_m=0.0,
        )

    monkeypatch.setattr(stepper, "_build_state", _build_state)
    transition = stepper.advance(state, 0.0)

    assert transition.next_state.step_count == stepper.required_episode_steps
    assert transition.terminated is True
    assert transition.truncated is False
    assert transition.violation_code is ViolationCode.ONGOING



















def test_terminal_punctuality_reward_favors_smaller_time_error(
    mtto_env: MTTOEnv,
) -> None:
    terminal_state = replace(
        mtto_env.state,
        position_m=mtto_env.train_service.target_position,
        speed_mps=0.0,
        stop_error_m=0.0,
        operation_time_s=mtto_env.train_service.schedule_time,
    )
    punctual_reward = mtto_env.reward_calculator.calculate(
        OperationalTransition(
            mtto_env.state,
            terminal_state,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ViolationCode.ONGOING,
        )
    )
    late_state = replace(
        terminal_state,
        operation_time_s=terminal_state.operation_time_s + 60.0,
    )
    late_reward = mtto_env.reward_calculator.calculate(
        OperationalTransition(
            terminal_state,
            late_state,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ViolationCode.ONGOING,
        )
    )

    assert punctual_reward.terminal_punctuality > late_reward.terminal_punctuality


def test_change_schedule_time_updates_time_context_without_dp_reference(
    mtto_env: MTTOEnv,
) -> None:
    mtto_env.reset()
    observation = mtto_env.change_schedule_time(445.0)

    assert mtto_env.train_service.schedule_time == pytest.approx(445.0)
    assert observation.shape == mtto_env.observation_space.shape


def test_external_rollout_uses_same_transition_and_reward_path(mtto_env: MTTOEnv) -> None:
    action = np.asarray([-1.0], dtype=np.float32)
    mtto_env.reset()
    _, env_reward, env_terminated, env_truncated, _ = mtto_env.step(action)

    result = evaluate_operational_policy_once(
        lambda _obs: action,
        stepper=mtto_env.stepper,
        reward_calculator=mtto_env.reward_calculator,
        observation_builder=mtto_env.observation_builder,
    )

    # From standstill, maximum braking produces the same immediate failed-stop
    # truncation in both execution paths.
    assert result.episode_steps == 1
    assert result.total_reward == pytest.approx(env_reward)
    assert result.terminated is env_terminated
    assert result.truncated is env_truncated
    assert result.success is False


def test_gym_evaluation_uses_terminal_flags_without_persisting_violation_code(
    mtto_env: MTTOEnv,
) -> None:
    class BrakingPolicy:
        def predict(self, _obs, deterministic: bool = True):
            return np.asarray([-1.0], dtype=np.float32), None

    result = evaluate_policy_once(BrakingPolicy(), mtto_env)

    assert result.terminated is False
    assert result.truncated is True
    assert result.success is False
    assert "violation_code" not in result.to_metrics()
    assert "terminated" not in result.to_metrics()
    assert "truncated" not in result.to_metrics()
