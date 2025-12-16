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
    )

    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    task = Task(
        starting_position=ly_zp,
        starting_velocity=0.0,
        destination=pa_zp,
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
        ds=10.0,
    )
    return maglevttoenv


def test_reset(mttoenv: MTTOEnv):
    obs, info = mttoenv.reset()
    assert isinstance(obs, dict), "obs should be a dictionary"
    expected_keys = [
        "agent_remainning_displacement",
        "agent_current_speed",
        "agent_remainning_schedule_time",
        "current_slope",
        "next_dslope",
        "displacement_between_next_dslope",
        "current_vlimit",
        "next_dvlimit",
        "displacement_between_next_dvlimit",
    ]
    for key in expected_keys:
        assert key in obs, f"Missing key in obs: {key}"
    # Check types of some important fields
    assert isinstance(obs["agent_remainning_displacement"], np.ndarray)
    assert isinstance(obs["agent_current_speed"], np.ndarray)
    assert isinstance(obs["agent_remainning_schedule_time"], np.ndarray)
    assert isinstance(obs["current_slope"], np.ndarray)
    assert isinstance(obs["next_dslope"], np.ndarray)
    assert isinstance(obs["displacement_between_next_dslope"], np.ndarray)
    assert isinstance(obs["current_vlimit"], np.ndarray)
    assert isinstance(obs["next_dvlimit"], np.ndarray)
    assert isinstance(obs["displacement_between_next_dvlimit"], np.ndarray)
    np.testing.assert_allclose(obs["agent_remainning_displacement"], 29270.046 - 135.0)
    np.testing.assert_allclose(obs["agent_current_speed"], 0.0)
    np.testing.assert_allclose(obs["agent_remainning_schedule_time"], 440.0)
    np.testing.assert_allclose(obs["current_slope"], 0.0)
    np.testing.assert_allclose(obs["next_dslope"], -0.0154)
    np.testing.assert_allclose(obs["displacement_between_next_dslope"], 2748.4972)
    np.testing.assert_allclose(obs["current_vlimit"], 60.0)
    np.testing.assert_allclose(obs["next_dvlimit"], 40.0)
    np.testing.assert_allclose(obs["displacement_between_next_dvlimit"], 105.0)
    assert "total_energy_consumption" in info, (
        "Missing key in info: total_energy_consumption"
    )
    assert isinstance(info["total_energy_consumption"], float)
    np.testing.assert_allclose(info["total_energy_consumption"], 0.0)


def test_cal_energy_consumption(mttoenv: MTTOEnv):
    obs, info = mttoenv.reset()
    energy_consumption1 = mttoenv.cal_energy_consumption(
        acc=0.0, displacement=0.0, travel_time=0.0
    )
    energy_consumption2 = mttoenv.cal_energy_consumption(
        acc=1.0, displacement=100.0, travel_time=14.142
    )
    print(f"acc=0.0 step energy consumption is {energy_consumption1}")
    print(f"acc=1.0 step energy consumption is {energy_consumption2}")
    assert energy_consumption1 >= 0, "Energy consumption should be non-negative"
    assert energy_consumption2 >= 0, "Energy consumption should be non-negative"


def test_whole_env(mttoenv: MTTOEnv):
    check_env(mttoenv)
