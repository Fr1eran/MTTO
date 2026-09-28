import numpy as np

from mtto.domain.line import (
    Line,
    get_slope_array_numba,
    get_slope_scalar_numba,
    get_speed_limit_array_numba,
    get_speed_limit_scalar_numba,
)


def test_get_slope(paper_scenario):
    track: Line = paper_scenario.line
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
        dtype=np.float64,
    )
    result = get_slope_array_numba(pos, track.slopes, track.slope_intervals)
    scalar_result_1 = get_slope_scalar_numba(
        2885.1417, track.slopes, track.slope_intervals
    )
    scalar_result_2 = get_slope_scalar_numba(
        2883.4972, track.slopes, track.slope_intervals
    )
    np.testing.assert_allclose(result, expected_result)
    np.testing.assert_allclose(scalar_result_1, -0.0179)
    np.testing.assert_allclose(scalar_result_2, -0.0154)


def test_get_speed_limit(paper_scenario):
    track: Line = paper_scenario.line
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
    result = get_speed_limit_array_numba(
        pos, track.speed_limits, track.speed_limit_intervals
    )
    scalar_result = get_speed_limit_scalar_numba(
        240.0, track.speed_limits, track.speed_limit_intervals
    )
    np.testing.assert_allclose(result, expected_result)
    np.testing.assert_allclose(scalar_result, 100.0 / 3.6)
