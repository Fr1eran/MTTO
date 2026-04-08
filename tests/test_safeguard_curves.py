import numpy as np
import pytest
from model.ocs import SafeGuardCurves
from model.vehicle import VehicleInfo
from model.track import TrackInfo, TrackProfile
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
)


@pytest.fixture(scope="module")
def safeguard_curves_and_vehicle():
    # 坡度，百分位
    slopes, slope_intervals = load_slopes()

    # 区间限速
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()

    track = TrackInfo(
        slopes,
        slope_intervals,
        speed_limits.tolist(),
        speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )
    trackprofile = TrackProfile(track=track)
    cal_SGC = SafeGuardCurves(trackprofile=trackprofile)
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    return cal_SGC, vehicle


def test_cal_levi_curves(safeguard_curves_and_vehicle):
    cal_SGC, vehicle = safeguard_curves_and_vehicle
    aps, _ = load_auxiliary_stopping_areas_ap_and_dp()
    curves = cal_SGC.calc_levi_curves(aps, vehicle, ds=1)
    # 检查返回类型和内容
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    # 检查每个曲线的横纵坐标长度一致
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)  # 每条曲线横坐标数组
        assert isinstance(item[1, :], np.ndarray)  # 每条曲线纵坐标数组


def test_cal_brake_curves(safeguard_curves_and_vehicle):
    cal_SGC, vehicle = safeguard_curves_and_vehicle
    _, dps = load_auxiliary_stopping_areas_ap_and_dp()
    curves = cal_SGC.calc_brake_curves(dps, vehicle, ds=1)
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)
        assert isinstance(item[1, :], np.ndarray)
