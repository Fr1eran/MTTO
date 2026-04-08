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
    load_auxiliary_parking_areas,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_station_zp_positions,
)

from gymnasium.utils.env_checker import check_env


@pytest.fixture(scope="module")
def mttoenv():
    # 读取线路数据
    slopes, slope_intervals = load_slopes()
    s_limits, s_intervals = load_speed_limits(to_mps=True)
    aps, dps = load_auxiliary_parking_areas()
    ly_zp, pa_zp = load_station_zp_positions()
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )

    sgu = SafeGuardUtility(
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.95,
    )

    track = Track(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        ASA_aps=aps,
        ASA_dps=dps,
    )

    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    task = Task(
        start_position=ly_zp,
        start_speed=0.0,
        target_position=pa_zp,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error=120,
        max_stop_error=0.3,
    )

    maglevttoenv = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguardutility=sgu,
        task=task,
        gamma=0.995,
        max_step_distance=10.0,
    )
    return maglevttoenv


def test_reset(mttoenv: MTTOEnv):
    obs, info = mttoenv.reset()
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
        "current_latest_traction_intervation_point",
        "current_latest_braking_intervation_point",
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
    assert isinstance(obs["current_latest_traction_intervation_point"], np.ndarray)
    assert isinstance(obs["current_latest_braking_intervation_point"], np.ndarray)
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


def test_cal_energy_consumption(mttoenv: MTTOEnv):
    obs, info = mttoenv.reset()
    mec1, lec1 = mttoenv.ecc.CalcEnergy(
        begin_pos=mttoenv.current_pos,
        begin_speed=mttoenv.current_speed,
        acc=0.0,
        distance=0.0,
        direction=mttoenv.direction,
        operation_time=0.0,
        vehicle=mttoenv.vehicle,
        trackprofile=mttoenv.trackprofile,
    )
    energy_consumption1 = mec1 + lec1

    mec2, lec2 = mttoenv.ecc.CalcEnergy(
        begin_pos=mttoenv.current_pos,
        begin_speed=mttoenv.current_speed,
        acc=1.0,
        distance=100.0,
        direction=mttoenv.direction,
        operation_time=14.142,
        vehicle=mttoenv.vehicle,
        trackprofile=mttoenv.trackprofile,
    )
    energy_consumption2 = mec2 + lec2

    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_whole_env(mttoenv: MTTOEnv):
    check_env(mttoenv)
