import numpy as np
import pickle
import json
import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardUtility


@pytest.fixture(scope="module")
def safeguardutil():
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        sl_data = json.load(f)
        s_limits = sl_data["speed_limits"]
        s_limits = np.asarray(s_limits, dtype=np.float64) / 3.6
        s_intervals = sl_data["intervals"]
    with open("data/rail/safeguard/min_curves_list.pkl", "rb") as f:
        min_curves_list = pickle.load(f)
    with open("data/rail/safeguard/max_curves_list.pkl", "rb") as f:
        max_curves_list = pickle.load(f)
    return SafeGuardUtility(
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        gamma=0.95,
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
    expected_sp: list[int] = [-1, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8]
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
    result_Currentsp: list[int] = []
    for i in range(len(pos)):
        current_min_speed, current_max_speed, current_sp = (
            safeguardutil.GetMinAndMaxSpeed(
                current_pos=pos[i], current_speed=speed[i], current_sp=None
            )
        )
        result_CurrentMinSpeed.append(current_min_speed)
        result_CurrentMaxSpeed.append(current_max_speed)
        result_Currentsp.append(current_sp)
    result_IsCurrentMinSpeedEqualToZero = np.isclose(result_CurrentMinSpeed, 0.0)
    result_IsCurrentSpeedBigger = np.asarray(speed) > np.asarray(result_CurrentMaxSpeed)
    np.testing.assert_array_equal(
        result_IsCurrentMinSpeedEqualToZero, expected_IsCurrentMinSpeedEqualToZero
    )
    np.testing.assert_array_equal(result_Currentsp, expected_sp)
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
    expected_Currentsp: list[int] = [-1, -1, -1, 0, 0, 1, 2, 3, 3, 8, 8]
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
    result_Currentsp: list[int] = []
    for i in range(len(pos)):
        current_sp = sp[i]
        current_min_speed, current_max_speed, current_sp = (
            safeguardutil.GetMinAndMaxSpeed(
                current_pos=pos[i], current_speed=speed[i], current_sp=current_sp
            )
        )
        result_CurrentMinSpeed.append(current_min_speed)
        result_CurrentMaxSpeed.append(current_max_speed)
        result_Currentsp.append(current_sp)
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
    np.testing.assert_array_equal(result_Currentsp, expected_Currentsp)
