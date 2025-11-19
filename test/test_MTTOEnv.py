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
from model.Track import TrackProfile

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
    # 读取危险速度域数据
    idp_points = np.load(file="data/rail/safeguard/idp_points.npy")
    with open("data/rail/safeguard/levi_curves_part.pkl", "rb") as f:
        levi_curves_part = pickle.load(f)
    with open("data/rail/safeguard/brake_curves_part.pkl", "rb") as f:
        brake_curves_part = pickle.load(f)

    guard = SafeGuardUtility(
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        idp_points=idp_points,
        levi_curves_part_list=levi_curves_part,
        brake_curves_part_list=brake_curves_part,
        gamma=0.95,
    )
    trackprofile = TrackProfile(
        slopes=slopes,
        slopeintervals=slope_intervals,
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
    )
    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    vehicle.starting_position = ly_zp
    vehicle.starting_velocity = 0.0
    vehicle.destination = pa_zp
    vehicle.schedule_time = 440.0
    maglevttoenv = MTTOEnv(
        vehicle=vehicle,
        trackprofile=trackprofile,
        safeguardutil=guard,
        ds=100,
        render_mode="human",
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
