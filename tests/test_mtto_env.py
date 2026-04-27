import numpy as np
import pytest
from rl.mtto_env import MTTOEnv
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
    )
    return maglevttoenv


def test_reset(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    assert isinstance(obs, dict), "obs should be a dictionary"
    expected_keys = [
        "remaining_distance",
        "current_speed",
        "current_acc",
        "remaining_schedule_time",
        "current_slope",
        "current_max_speed",
        "current_min_speed",
        "next_slope",
        "next_max_speed",
        "next_min_speed",
        "current_latest_traction_intervention_point",
        "current_latest_braking_intervention_point",
    ]
    for key in expected_keys:
        assert key in obs, f"Missing key in obs: {key}"
    # Check types of some important fields
    assert isinstance(obs["remaining_distance"], np.ndarray)
    assert isinstance(obs["current_speed"], np.ndarray)
    assert isinstance(obs["current_acc"], np.ndarray)
    assert isinstance(obs["remaining_schedule_time"], np.ndarray)
    assert isinstance(obs["current_slope"], np.ndarray)
    assert isinstance(obs["current_max_speed"], np.ndarray)
    assert isinstance(obs["current_min_speed"], np.ndarray)
    assert isinstance(obs["next_slope"], np.ndarray)
    assert isinstance(obs["next_max_speed"], np.ndarray)
    assert isinstance(obs["next_min_speed"], np.ndarray)
    assert isinstance(obs["current_latest_traction_intervention_point"], np.ndarray)
    assert isinstance(obs["current_latest_braking_intervention_point"], np.ndarray)
    np.testing.assert_allclose(obs["remaining_distance"], 0.9953664)
    np.testing.assert_allclose(obs["current_speed"], 0.0)
    np.testing.assert_allclose(obs["current_acc"], 0.0)
    np.testing.assert_allclose(obs["remaining_schedule_time"], 1.0)
    np.testing.assert_allclose(obs["current_slope"], 0.0)
    np.testing.assert_allclose(obs["current_max_speed"], 0.0)
    np.testing.assert_allclose(obs["current_min_speed"], 0.0)
    np.testing.assert_allclose(obs["next_slope"], 0.0)
    np.testing.assert_allclose(obs["next_max_speed"], 0.032199375331401825, rtol=1e-6)
    np.testing.assert_allclose(obs["next_min_speed"], 0.0)
    assert info == {}


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    mec1, lec1 = mtto_env.ecc.calc_energy(
        begin_pos=mtto_env.current_pos,
        begin_speed=mtto_env.current_speed,
        acc=0.0,
        distance=0.0,
        direction=mtto_env.direction,
        operation_time=0.0,
        vehicle=mtto_env.vehicle,
        trackprofile=mtto_env.trackprofile,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = mtto_env.ecc.calc_energy(
        begin_pos=mtto_env.current_pos,
        begin_speed=mtto_env.current_speed,
        acc=1.0,
        distance=100.0,
        direction=mtto_env.direction,
        operation_time=14.142,
        vehicle=mtto_env.vehicle,
        trackprofile=mtto_env.trackprofile,
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
    assert "docking_position" not in runtime

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
        action = mtto_env.action_space.sample()
        mtto_env.step(action)

        assert mtto_env.render_mode is None
        assert mtto_env.trajectory_pos is not None
        assert mtto_env.trajectory_speed_mps is not None
        assert len(mtto_env.trajectory_pos) >= 1
        assert len(mtto_env.trajectory_speed_mps) >= 1
    finally:
        mtto_env.enable_trajectory_tracking = False
        mtto_env.reset()
