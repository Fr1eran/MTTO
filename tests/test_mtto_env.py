import numpy as np
import pytest
from rl.mtto_env import MTTOEnv, RewardConfig
from model.vehicle import VehicleInfo
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)

from gymnasium.utils.env_checker import check_env


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


def test_reset(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (14,)
    np.testing.assert_allclose(obs[0], 1.0)  # remaining_distance
    np.testing.assert_allclose(obs[1], 0.0)  # current_speed
    np.testing.assert_allclose(obs[2], 0.0)  # current_acc
    np.testing.assert_allclose(obs[3], 1.0)  # remaining_schedule_time
    np.testing.assert_allclose(obs[5], 0.0)  # current_slope
    np.testing.assert_allclose(obs[6], 0.0)  # current_max_speed
    np.testing.assert_allclose(obs[7], 0.0)  # current_min_speed
    np.testing.assert_allclose(obs[8], 0.0)  # next_slope
    np.testing.assert_allclose(
        obs[9], 0.032199375331401825, rtol=1e-4
    )  # next_max_speed
    np.testing.assert_allclose(obs[10], 0.0)  # next_min_speed
    assert info == {}


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    mec1, lec1 = mtto_env.ecc.calc_energy(
        begin_pos=mtto_env.current_position,
        begin_speed=mtto_env.current_speed,
        acc=0.0,
        distance=0.0,
        direction=mtto_env.direction,
        operation_time=0.0,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = mtto_env.ecc.calc_energy(
        begin_pos=mtto_env.current_position,
        begin_speed=mtto_env.current_speed,
        acc=1.0,
        distance=100.0,
        direction=mtto_env.direction,
        operation_time=14.142,
        vehicle=mtto_env.vehicle,
        track=mtto_env.track,
    )
    energy_consumption2 = mec2 + lec2

    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_reference_remaining_operation_time_matches_endpoints(mtto_env: MTTOEnv):
    mtto_env.reset()

    start_remaining = mtto_env._get_reference_remaining_operation_time(
        mtto_env.train_service.start_position
    )
    target_remaining = mtto_env._get_reference_remaining_operation_time(
        mtto_env.train_service.target_position
    )

    assert start_remaining == pytest.approx(
        mtto_env.ref_total_operation_time,
        rel=1e-3,
    )
    assert target_remaining == pytest.approx(0.0, abs=0.5)


def test_punctuality_reward_depends_on_position_and_time_only(mtto_env: MTTOEnv):
    mtto_env.reset()

    mtto_env.current_position = mtto_env.train_service.start_position + 5000.0
    mtto_env.current_operation_time = 120.0
    mtto_env.last_state["pos"] = mtto_env.current_position - 100.0
    mtto_env.last_state["operation_time"] = 118.0

    mtto_env.current_speed = 5.0
    mtto_env.last_state["speed"] = 8.0
    reward_low_speed = mtto_env._get_reward_punctuality_dense()

    mtto_env.current_speed = 25.0
    mtto_env.last_state["speed"] = 30.0
    reward_high_speed = mtto_env._get_reward_punctuality_dense()

    np.testing.assert_allclose(reward_low_speed, reward_high_speed)


def test_whole_env(mtto_env: MTTOEnv):
    check_env(mtto_env)


def test_step_without_diagnostics_keeps_tb_dicts_empty(mtto_env: MTTOEnv):
    mtto_env.enable_diagnostics = False
    try:
        mtto_env.reset()
        action = mtto_env.action_space.sample()
        _, _, _, _, info = mtto_env.step(action)

        assert mtto_env.rewards_info == {}
        assert mtto_env.state_info == {}
        assert mtto_env.constraint_info == {}
        assert mtto_env.event_info == {}

        expected_runtime_keys = {
            "energy_consumption",
            "operation_time",
            "position",
            "stopping_point_index",
        }
        runtime_namespace = "basic" if "basic" in info else "runtime"
        assert runtime_namespace in info
        runtime = info[runtime_namespace]
        assert isinstance(runtime, dict)
        assert expected_runtime_keys.issubset(set(runtime.keys()))
        assert expected_runtime_keys.issubset(set(mtto_env.basic_info.keys()))
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
    assert "rewards" in info
    assert "state" in info
    assert "constraint" in info
    assert "event" in info

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
            float(mtto_env.current_position)
        )
        assert mtto_env.trajectory_speed_mps[-1] == pytest.approx(
            abs(float(mtto_env.current_speed))
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
    monkeypatch.setattr(mtto_env, "_update_motion", lambda: (next_speed, 0.0, 0.0))
    monkeypatch.setattr(mtto_env.ecc, "calc_energy", lambda **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(mtto_env, "_get_upper_speed", lambda _pos: 10_000.0)
    monkeypatch.setattr(
        mtto_env.safeguard_utility,
        "get_min_and_max_speed",
        lambda **_kwargs: (0.0, 10_000.0),
    )
    monkeypatch.setattr(
        mtto_env.sps,
        "step_to_next_stopping_point",
        lambda **_kwargs: -1,
    )


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
    assert reward == pytest.approx(-10.0)
    assert "constraint" in info
    assert info["constraint"]["violation_code"] == MTTOEnv.VIOLATION_CODE_FAILED_STOP


def test_step_success_is_terminated_without_truncation(
    mtto_env: MTTOEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtto_env.enable_diagnostics = True
    mtto_env.diagnostics_interval_steps = 1
    mtto_env.reset()
    _patch_step_dependencies_for_outcome_tests(mtto_env, monkeypatch, next_speed=0.0)
    mtto_env.current_position = mtto_env.train_service.target_position
    mtto_env.current_speed = 0.0
    mtto_env.current_operation_time = mtto_env.train_service.schedule_time

    _, _, terminated, truncated, info = mtto_env.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert "constraint" in info
    assert info["constraint"]["violation_code"] == MTTOEnv.VIOLATION_CODE_ONGOING


def test_potential_punctuality_v3_is_finite_for_extreme_negative_input(
    mtto_env: MTTOEnv,
) -> None:
    val = mtto_env._potential_punctuality_v3(-100.0)
    assert np.isfinite(val)


def test_potential_punctuality_v4_peaks_at_zero_time_error(
    mtto_env: MTTOEnv,
) -> None:
    peak = mtto_env._potential_punctuality_v4(0.0)
    early = mtto_env._potential_punctuality_v4(0.1)
    late = mtto_env._potential_punctuality_v4(-0.1)

    assert peak > early
    assert peak > late
    assert late < early


def test_potential_punctuality_v4_late_side_drops_monotonically(
    mtto_env: MTTOEnv,
) -> None:
    slightly_late = mtto_env._potential_punctuality_v4(-0.05)
    clearly_late = mtto_env._potential_punctuality_v4(-0.2)

    assert clearly_late < slightly_late


def test_potential_punctuality_v4_is_finite_for_extreme_input(
    mtto_env: MTTOEnv,
) -> None:
    assert np.isfinite(mtto_env._potential_punctuality_v4(-100.0))
    assert np.isfinite(mtto_env._potential_punctuality_v4(100.0))


def test_punctuality_dense_reward_is_stable_for_extreme_time_redundancy(
    mtto_env: MTTOEnv,
) -> None:
    mtto_env.reset()
    mtto_env.current_redundant_operation_time = -100.0
    mtto_env.last_state["time_redundancy"] = 0.0
    val = mtto_env._get_reward_punctuality_dense()

    assert np.isfinite(val)
    assert abs(val) < 1e4
