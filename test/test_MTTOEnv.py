import json
import os
import sys
import pickle

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl.MTTOEnv import MTTOEnv
from model.Vehicle import Vehicle
from model.SafeGuard import SafeGuardUtility
from model.Track import Track
from model.Task import Task

from gymnasium.utils.env_checker import check_env


@pytest.fixture(scope="module")
def mttoenv():
    # 读取线路数据
    with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
        slope_data = json.load(f)
        slopes = slope_data["slopes"]
        slope_intervals = slope_data["intervals"]
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        sl_data = json.load(f)
        s_limits = sl_data["speed_limits"]
        s_intervals = sl_data["intervals"]
    with open("data/rail/raw/auxiliary_parking_areas.json", "r", encoding="utf-8") as f:
        apa_data = json.load(f)
        aps = apa_data["accessible_points"]
        dps = apa_data["dangerous_points"]
    # 读取车站数据
    with open("data/rail/raw/stations.json", "r", encoding="utf-8") as f:
        stations_data = json.load(f)
        # ly_begin = stations_data["LY"]["begin"]
        ly_zp = stations_data["LY"]["zp"]
        # ly_end = stations_data["LY"]["end"]
        # pa_begin = stations_data["PA"]["begin"]
        pa_zp = stations_data["PA"]["zp"]
        # pa_end = stations_data["PA"]["end"]
    # 读取防护曲线
    with open("data/rail/safeguard/min_curves_list.pkl", "rb") as f:
        min_curves_list = pickle.load(f)
    with open("data/rail/safeguard/max_curves_list.pkl", "rb") as f:
        max_curves_list = pickle.load(f)

    sgu = SafeGuardUtility(
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        gamma=0.95,
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
        safeguardutil=sgu,
        task=task,
        gamma=0.995,
        max_step_distance=10.0,
    )
    return maglevttoenv


def test_reset(mttoenv: MTTOEnv):
    obs, info = mttoenv.reset()
    assert isinstance(obs, dict), "obs should be a dictionary"
    expected_keys = [
        "agent_remaining_distance",
        "agent_current_speed",
        "agent_current_acc",
        "agent_remaining_schedule_time",
        "current_slope",
        "current_max_speed",
        "current_min_speed",
        "next_slope",
        "next_max_speed",
        "next_min_speed",
    ]
    for key in expected_keys:
        assert key in obs, f"Missing key in obs: {key}"
    # Check types of some important fields
    assert isinstance(obs["agent_remaining_distance"], np.ndarray)
    assert isinstance(obs["agent_current_speed"], np.ndarray)
    assert isinstance(obs["agent_current_acc"], np.ndarray)
    assert isinstance(obs["agent_remaining_schedule_time"], np.ndarray)
    assert isinstance(obs["current_slope"], np.ndarray)
    assert isinstance(obs["current_max_speed"], np.ndarray)
    assert isinstance(obs["current_min_speed"], np.ndarray)
    assert isinstance(obs["next_slope"], np.ndarray)
    assert isinstance(obs["next_max_speed"], np.ndarray)
    assert isinstance(obs["next_min_speed"], np.ndarray)
    np.testing.assert_allclose(obs["agent_remaining_distance"], 0.9953664)
    np.testing.assert_allclose(obs["agent_current_speed"], 0.0)
    np.testing.assert_allclose(obs["agent_current_acc"], 0.0)
    np.testing.assert_allclose(obs["agent_remaining_schedule_time"], 1.0)
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
    energy_consumption1 = mttoenv._cal_energy_consumption(
        acc=0.0, displacement=0.0, travel_time=0.0
    )
    energy_consumption2 = mttoenv._cal_energy_consumption(
        acc=1.0, displacement=100.0, travel_time=14.142
    )
    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_whole_env(mttoenv: MTTOEnv):
    check_env(mttoenv)
