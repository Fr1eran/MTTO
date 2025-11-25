import numpy as np
import pickle
import json
import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardUtility


@pytest.fixture(scope="module")
def safeguard():
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        sl_data = json.load(f)
        s_limits = sl_data["speed_limits"]
        s_limits = np.asarray(s_limits, dtype=np.float64) / 3.6
        s_intervals = sl_data["intervals"]
    with open("data/rail/safeguard/levi_curves_list.pkl", "rb") as f:
        levi_curves_list = pickle.load(f)
    with open("data/rail/safeguard/brake_curves_list.pkl", "rb") as f:
        brake_curves_list = pickle.load(f)
    return SafeGuardUtility(
        speed_limits=s_limits,
        speed_limit_intervals=s_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        gamma=0.95,
    )


def test_detect_danger(safeguard):
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
    result = safeguard.DetectDanger(pos, speed)
    np.testing.assert_array_equal(result, expected_result)
