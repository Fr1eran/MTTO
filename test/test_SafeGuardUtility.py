import numpy as np
import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardUtility
from utils.data_loader import load_safeguard_curves, load_speed_limits


@pytest.fixture(scope="module")
def safeguardutil():
    s_limits, s_intervals = load_speed_limits(to_mps=True, dtype=np.float64)
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )
    return SafeGuardUtility(
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.95,
    )


def test_detect_danger(safeguardutil):
    pos = np.array(
        [
            725,
            1754,
            2116,
            2762,
            4113,
            5484,
            6794,
            7800,
            11109,
            13125,
            17419,
            20060,
            6189.5,
            8548,
        ],
        dtype=np.float64,
    )
    speed = (
        np.array(
            [42, 8, 16, 9.5, 10, 15, 61, 45, 66, 74, 92, 90, 378, 372], dtype=np.float64
        )
        / 3.6
    )
    expected_result = np.array(
        [
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
        ]
    )
    result = safeguardutil.DetectDanger(pos, speed)
    np.testing.assert_array_equal(result, expected_result)


def test_GetMinAndMaxSpeed_with_currentspisNone(safeguardutil):
    pos: list[float] = [
        200.0,
        530.0,
        800.0,
        300.0,
        1250.0,
        1200.0,
        1800.0,
        3500.0,
        4500.0,
        8800.0,
        17000.0,
        22000.0,
        26500.0,
        29000.0,
    ]
    speed: list[float] = [
        8.8 / 3.6,
        9.0 / 3.6,
        5.0 / 3.6,
        15.0 / 3.6,
        20.0 / 3.6,
        45.0 / 3.6,
        60.0 / 3.6,
        60.0 / 3.6,
        60.0 / 3.6,
        25.0 / 3.6,
        50.0 / 3.6,
        130.0 / 3.6,
        60.0 / 3.6,
        60.0 / 3.6,
    ]
    input_sp: list[int] = [-1, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8]
    expected_IsCurrentMinSpeedEqualToZero: list[bool] = [
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    expected_IsCurrentSpeedBiggerThanCurrentMaxSpeed: list[bool] = [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    result_CurrentMinSpeed: list[float] = []
    result_CurrentMaxSpeed: list[float] = []
    for i in range(len(pos)):
        current_min_speed, current_max_speed = safeguardutil.GetMinAndMaxSpeed(
            current_pos=pos[i], current_sp=input_sp[i]
        )
        result_CurrentMinSpeed.append(current_min_speed)
        result_CurrentMaxSpeed.append(current_max_speed)
    result_IsCurrentMinSpeedEqualToZero = np.isclose(result_CurrentMinSpeed, 0.0)
    result_IsCurrentSpeedBigger = np.asarray(speed) > np.asarray(result_CurrentMaxSpeed)
    np.testing.assert_array_equal(
        result_IsCurrentMinSpeedEqualToZero, expected_IsCurrentMinSpeedEqualToZero
    )
    np.testing.assert_array_equal(
        result_IsCurrentSpeedBigger,
        expected_IsCurrentSpeedBiggerThanCurrentMaxSpeed,
    )


def test_GetMinAndMaxSpeed_with_currentspisNotNone(safeguardutil):
    pos: list[float] = [
        200.0,
        530.0,
        800.0,
        300.0,
        1300.0,
        800.0,
        2000.0,
        2600.0,
        5230.0,
        28600.0,
        29312.0,
    ]
    speed: list[float] = [
        8.8 / 3.6,
        9.0 / 3.6,
        5.0 / 3.6,
        15.0 / 3.6,
        10.0 / 3.6,
        65.0 / 3.6,
        50.0 / 3.6,
        50.0 / 3.6,
        8.0 / 3.6,
        70.0 / 3.6,
        40.0 / 3.6,
    ]
    sp: list[int] = [
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        1,
        2,
        3,
        8,
        8,
    ]
    expected_IsCurrentMinSpeedEqualToZero: list[bool] = [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    expected_IsCurrentMaxSpeedBigger: list[bool] = [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]
    result_CurrentMinSpeed: list[float] = []
    result_CurrentMaxSpeed: list[float] = []
    for i in range(len(pos)):
        current_sp = sp[i]
        current_min_speed, current_max_speed = safeguardutil.GetMinAndMaxSpeed(
            current_pos=pos[i], current_sp=current_sp
        )
        result_CurrentMinSpeed.append(current_min_speed)
        result_CurrentMaxSpeed.append(current_max_speed)
    result_IsCurrentMinSpeedEqualToZero = np.isclose(result_CurrentMinSpeed, 0.0)
    result_IsCurrentMaxSpeedBiggerThanCurrentSpeed = np.asarray(
        result_CurrentMaxSpeed
    ) > np.asarray(speed)
    np.testing.assert_array_equal(
        result_IsCurrentMinSpeedEqualToZero, expected_IsCurrentMinSpeedEqualToZero
    )
    np.testing.assert_array_equal(
        result_IsCurrentMaxSpeedBiggerThanCurrentSpeed, expected_IsCurrentMaxSpeedBigger
    )


@pytest.fixture(scope="module")
def position_safeguardutil():
    safeguardutil = SafeGuardUtility.__new__(SafeGuardUtility)
    safeguardutil.speed_limits = np.array([10.0], dtype=np.float64)
    safeguardutil.speed_limit_intervals = np.array([0.0, 100.0], dtype=np.float64)
    safeguardutil.min_curves_list = [
        np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 0.0]], dtype=np.float64),
    ]
    safeguardutil.max_curves_list = [
        np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 4.0, 2.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 2.0, 3.0], [6.0, 5.0, 3.0, 0.0]], dtype=np.float64),
    ]
    safeguardutil._min_curve_pos_list = [
        curve[0, :] for curve in safeguardutil.min_curves_list
    ]
    safeguardutil._min_curve_speed_list = [
        curve[1, :] for curve in safeguardutil.min_curves_list
    ]
    safeguardutil._max_curve_pos_list = [
        curve[0, :] for curve in safeguardutil.max_curves_list
    ]
    safeguardutil._max_curve_speed_list = [
        curve[1, :] for curve in safeguardutil.max_curves_list
    ]
    return safeguardutil


def test_GetMinAndMaxPosition_with_currentsp(position_safeguardutil):
    min_pos, max_pos = (
        position_safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
            current_speed=1.5,
            current_sp=0,
        )
    )
    np.testing.assert_allclose(min_pos, 1.5)
    np.testing.assert_allclose(max_pos, 1.75)


def test_GetMinAndMaxPosition_with_currentsp_extrapolation(position_safeguardutil):
    min_pos, max_pos = (
        position_safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
            current_speed=4.5,
            current_sp=0,
        )
    )
    np.testing.assert_allclose(min_pos, -1.5)
    np.testing.assert_allclose(max_pos, -0.5)


def test_GetMinAndMaxPosition_with_currentsp_negative_one(position_safeguardutil):
    min_pos, max_pos = (
        position_safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
            current_speed=1.5,
            current_sp=-1,
        )
    )
    np.testing.assert_allclose(min_pos, 0.0)
    np.testing.assert_allclose(max_pos, 2.25)


def test_GetMinAndMaxPosition_real_data_smoke(safeguardutil):
    min_pos, max_pos = safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
        current_speed=2.0,
        current_sp=0,
    )
    assert np.isfinite(min_pos)
    assert np.isfinite(max_pos)
    assert min_pos <= max_pos
