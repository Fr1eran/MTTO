import numpy as np
import pytest

from model.Vehicle import Vehicle
from model.Track import Track
from model.Task import Task
from model.ORS import ORS
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)


@pytest.fixture(scope="module")
def operation_reference_system():
    # 坡度，百分位
    slopes, slope_intervals = load_slopes()

    # 区间限速
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

    aps, dps = load_auxiliary_stopping_areas_ap_and_dp()

    # 车站
    ly_zp, pa_zp = load_stations_goal_positions()

    track = Track(
        slopes,
        slope_intervals,
        speed_limits.tolist(),
        speed_limit_intervals,
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
        max_arr_time_error=120.0,
        max_stop_error=0.3,
    )
    return ORS(vehicle=vehicle, track=track, factor=0.95)
