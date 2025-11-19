import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.Track import Track, TrackProfile


@pytest.fixture(scope="module")
def trackprofile():
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

    track = Track(slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals)
    return TrackProfile(track=track)


def test_GetSlope(trackprofile: TrackProfile):
    pos = np.array(
        [
            2600.0,
            2884.0,
            2900.0,
            15000.0,
            17000.0,
            17050.0,
            20000.0,
            21100.0,
            21200.0,
            22400.0,
            22600.0,
            22700.0,
            27000.0,
            28000.0,
            28150.0,
            28200.0,
            28700.0,
            28750.0,
            29000.0,
        ],
        dtype=np.float64,
    )
    expected_result = np.array(
        [
            0.0000,
            -0.0154,
            -0.0179,
            -0.0333,
            -0.0020,
            0.0913,
            0.1226,
            0.0337,
            -0.2309,
            -0.3198,
            -0.2573,
            -0.0625,
            0.0000,
            -0.0417,
            -1.0362,
            -1.0778,
            -1.0278,
            -0.0500,
            0.0000,
        ],
        dtype=np.float32,
    )
    result1 = trackprofile.GetSlope(pos=pos, interpolate=False, dtype=np.float32)
    result2 = trackprofile.GetSlope(pos=pos, interpolate=True)
    result3 = trackprofile.GetSlope(pos=2885.1417, interpolate=False)
    result4 = trackprofile.GetSlope(pos=2883.4972, interpolate=True)
    np.testing.assert_allclose(result1, expected_result)
    assert len(np.asarray(result2)) == len(pos)
    assert isinstance(result3, np.floating)
    np.testing.assert_allclose(result3, -0.0179)
    assert isinstance(result4, np.floating)


def test_GetSpeedlimit(trackprofile: TrackProfile):
    pos = np.array(
        [
            200.0,
            400.0,
            800.0,
            1500.0,
            3000.0,
            4000.0,
            6000.0,
            8000.0,
            11000.0,
            18000.0,
            21500.0,
            22000.0,
            25000.0,
            27000.0,
            27500.0,
            28500.0,
            28700.0,
            29700.0,
            29880.0,
        ],
        dtype=np.float64,
    )
    expected_result = (
        np.array(
            [
                60.0,
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                350.0,
                400.0,
                450.0,
                480.0,
                450.0,
                400.0,
                350.0,
                300.0,
                250.0,
                200.0,
                150.0,
                105.0,
                60.0,
            ],
            dtype=np.float64,
        )
        / 3.6
    )
    result1 = trackprofile.GetSpeedlimit(pos=pos, dtype=np.float32)
    result2 = trackprofile.GetSpeedlimit(pos=240.0)
    np.testing.assert_allclose(result1, expected_result)
    assert isinstance(result2, np.floating)
    np.testing.assert_allclose(result2, 100.0 / 3.6)


def test_GetNextSlopeAndDistance(trackprofile: TrackProfile):
    pos = np.array(
        [
            2600.0,
            2884.0,
            29000.0,
        ],
        dtype=np.float64,
    )
    expected_slope_ahead = np.array(
        [
            -0.0154,
            -0.0179,
            0.0,
        ],
        dtype=np.float32,
    )
    expected_distance_ahead = np.array(
        [
            283.4972,
            1.1417,
            1000.0,
        ],
        dtype=np.float32,
    )
    expected_dslope_rear = np.array(
        [
            0.0,
            0.0,
            -0.05,
        ],
        dtype=np.float32,
    )
    expected_distance_rear = np.array(
        [
            -2600.0,
            -0.5028,
            -230.9513,
        ],
        dtype=np.float32,
    )
    dslope_ahead, distance_ahead = trackprofile.GetNextSlopeAndDistance(
        pos=pos, direction=1, dtype=np.float32
    )
    dslope_rear, distance_rear = trackprofile.GetNextSlopeAndDistance(
        pos=pos, direction=-1, dtype=np.float32
    )
    # 32位浮点数精度较低
    print(distance_rear)
    np.testing.assert_allclose(dslope_ahead, expected_slope_ahead, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        distance_ahead, expected_distance_ahead, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(dslope_rear, expected_dslope_rear, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        distance_rear, expected_distance_rear, rtol=1e-4, atol=1e-6
    )
