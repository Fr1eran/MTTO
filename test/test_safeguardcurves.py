import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardCurves
from model.Vehicle import Vehicle
from model.Track import Track, TrackProfile
from utils.data_loader import (
    load_auxiliary_parking_areas,
    load_slopes,
    load_speed_limits,
)


@pytest.fixture(scope="module")
def sgc_and_vehicle():
    # 坡度，百分位
    slopes, slope_intervals = load_slopes()

    # 区间限速
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

    aps, dps = load_auxiliary_parking_areas()

    track = Track(
        slopes,
        slope_intervals,
        speed_limits.tolist(),
        speed_limit_intervals,
        ASA_aps=aps,
        ASA_dps=dps,
    )
    trackprofile = TrackProfile(track=track)
    cal_SGC = SafeGuardCurves(trackprofile=trackprofile)
    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    return cal_SGC, vehicle


def test_cal_levi_curves(sgc_and_vehicle):
    cal_SGC, vehicle = sgc_and_vehicle
    aps, _ = load_auxiliary_parking_areas()
    curves = cal_SGC.CalcLeviCurves(aps, vehicle, ds=1)
    # 检查返回类型和内容
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    # 检查每个曲线的横纵坐标长度一致
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)  # 每条曲线横坐标数组
        assert isinstance(item[1, :], np.ndarray)  # 每条曲线纵坐标数组


def test_cal_brake_curves(sgc_and_vehicle):
    cal_SGC, vehicle = sgc_and_vehicle
    _, dps = load_auxiliary_parking_areas()
    curves = cal_SGC.CalcBrakeCurves(dps, vehicle, ds=1)
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)
        assert isinstance(item[1, :], np.ndarray)
