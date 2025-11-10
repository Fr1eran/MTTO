import numpy as np
import json
import matplotlib.pyplot as plt
import pytest

from model.Vehicle import Vehicle
from model.Track import Track
from model.Task import Task
from model.ORS import ORS


@pytest.fixture(scope="module")
def ors():
    # 坡度，百分位
    with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
        slope_data = json.load(f)
        slopes = slope_data["slopes"]
        slope_intervals = slope_data["intervals"]

    # 区间限速
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        speedlimit_data = json.load(f)
        speed_limits = speedlimit_data["speed_limits"]
        speed_limits = np.asarray(speed_limits) / 3.6
        speed_limit_intervals = speedlimit_data["intervals"]

    # 车站
    with open("data/rail/raw/stations.json", "r", encoding="utf-8") as f:
        stations_data = json.load(f)
        ly_zp = stations_data["LY"]["zp"]
        pa_zp = stations_data["PA"]["zp"]

    track = Track(slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals)
    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    task = Task(
        starting_position=ly_zp,
        starting_velocity=0.0,
        destination=pa_zp,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error=120.0,
        max_stop_error=0.3,
    )
    return ORS(vehicle=vehicle, track=track, task=task, gamma=0.95)
