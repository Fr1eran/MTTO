import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardCurves
from model.Vehicle import Vehicle
from model.Track import TrackProfile


@pytest.fixture(scope="module")
def sgc_and_vehicle():
    # 读取数据
    with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
        slope_data = json.load(f)
        slopes = slope_data["slopes"]
        slope_intervals = slope_data["intervals"]
    with open("data/rail/raw/velocity_limits.json", "r", encoding="utf-8") as f:
        vl_data = json.load(f)
        v_limits = vl_data["velocity_limits"]
        v_intervals = vl_data["intervals"]
    trackprofile = TrackProfile(
        slopes=slopes,
        slopeintervals=slope_intervals,
        speed_limits=v_limits,
        speed_limit_intervals=v_intervals,
    )
    cal_SGC = SafeGuardCurves(trackprofile=trackprofile)
    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    return cal_SGC, vehicle


def test_cal_levi_curves(sgc_and_vehicle):
    cal_SGC, vehicle = sgc_and_vehicle
    with open("data/rail/raw/auxiliary_parking_areas.json", "r", encoding="utf-8") as f:
        apa_data = json.load(f)
        aps = apa_data["accessible_points"]
    curves = cal_SGC.CalLeviCurves(aps, vehicle, ds=1)
    # 检查返回类型和内容
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    # 检查每个曲线的横纵坐标长度一致
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)  # 每条曲线横坐标数组
        assert isinstance(item[1, :], np.ndarray)  # 每条曲线纵坐标数组


def test_cal_brake_curves(sgc_and_vehicle):
    cal_SGC, vehicle = sgc_and_vehicle
    with open("data/rail/raw/auxiliary_parking_areas.json", "r", encoding="utf-8") as f:
        apa_data = json.load(f)
        dps = apa_data["dangerous_points"]
    curves = cal_SGC.CalBrakeCurves(dps, vehicle, ds=1)
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)
        assert isinstance(item[1, :], np.ndarray)
