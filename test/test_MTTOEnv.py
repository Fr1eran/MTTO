import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl.MTTOEnv import MTTOEnv
from model.Vehicle import Vehicle
from model.SafeGuard import SafeGuardUtility
from model.Track import Track
from model.Task import Task
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
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )

    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.95,
    )

    track = Track(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )

    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    task = Task(
        start_position=longyang_start_position,
        start_speed=0.0,
        target_position=putong_end_position,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error=120,
        max_stop_error=0.3,
    )

    maglevttoenv = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        task=task,
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
    assert "current_energy_consumption" in info, (
        "Missing key in info: current_energy_consumption"
    )
    assert "current_operation_time" in info, (
        "Missing key in info: current_operation_time"
    )
    assert "docking_position" in info, "Missing key in info: docking_position"
    assert isinstance(info["current_energy_consumption"], float)
    assert isinstance(info["current_operation_time"], float)
    assert isinstance(info["docking_position"], float)
    np.testing.assert_allclose(info["current_energy_consumption"], 0.0)
    np.testing.assert_allclose(info["current_operation_time"], 0.0)
    np.testing.assert_allclose(info["docking_position"], 135.0)


def test_cal_energy_consumption(mtto_env: MTTOEnv):
    obs, info = mtto_env.reset()
    mec1, lec1 = mtto_env.ecc.CalcEnergy(
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

    mec2, lec2 = mtto_env.ecc.CalcEnergy(
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
