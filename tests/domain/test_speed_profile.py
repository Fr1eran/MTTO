import warnings

import numpy as np
import pytest

from mtto.domain.speed_profile import SpeedProfile


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_m", []),
        ("speed_mps", [0.0]),
        ("time_s", [0.0]),
        ("propulsion_energy_kj", [0.0]),
        ("levitation_energy_kj", [0.0]),
        ("position_m", [[0.0, 1.0]]),
        ("speed_mps", [[0.0, 1.0]]),
        ("time_s", [[0.0, 1.0]]),
        ("propulsion_energy_kj", [[0.0, 1.0]]),
        ("levitation_energy_kj", [[0.0, 1.0]]),
        ("position_m", [0.0, np.nan]),
        ("speed_mps", [0.0, np.inf]),
        ("time_s", [0.0, np.nan]),
        ("propulsion_energy_kj", [0.0, np.inf]),
        ("levitation_energy_kj", [0.0, np.nan]),
        ("speed_mps", [0.0, -1.0]),
        ("position_m", [1.0, 0.0]),
        ("time_s", [0.0, -1.0]),
        ("propulsion_energy_kj", [1.0, 0.0]),
        ("levitation_energy_kj", [1.0, 0.0]),
        ("time_s", [1.0, 2.0]),
        ("propulsion_energy_kj", [1.0, 2.0]),
        ("levitation_energy_kj", [1.0, 2.0]),
    ],
)
def test_rejects_invalid_arrays(field, value):
    arrays = {
        "position_m": [0.0, 1.0],
        "speed_mps": [0.0, 1.0],
        "time_s": [0.0, 1.0],
        "propulsion_energy_kj": [0.0, 1.0],
        "levitation_energy_kj": [0.0, 1.0],
    }
    arrays[field] = value
    with pytest.raises(ValueError, match=field):
        SpeedProfile.from_arrays(**arrays)


@pytest.mark.parametrize(
    ("position", "speed", "time", "expected"),
    [
        ([0, 1], [0, 2], [0, 4], [0.5]),
        ([0, 0], [0, 2], [0, 0], [0.0]),
        ([0, 0], [0, 2], [0, 4], [0.5]),
    ],
)
def test_segment_acceleration(position, speed, time, expected):
    profile = SpeedProfile.from_arrays(position, speed, time, [0, 1], [0, 2])
    np.testing.assert_array_equal(profile.segment_acceleration_mps2, expected)
    np.testing.assert_array_equal(profile.total_energy_kj, [0, 3])


def test_single_node_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = SpeedProfile.from_arrays([0], [0], [0], [0], [0])
    assert profile.segment_acceleration_mps2.size == 0


@pytest.mark.parametrize("acceleration", [[], [[0.0]], [np.nan]])
def test_rejects_invalid_segment_acceleration(acceleration):
    with pytest.raises(ValueError, match="segment_acceleration_mps2"):
        SpeedProfile(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array(acceleration),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )


def test_arrays_are_read_only_and_copied():
    inputs = [np.array([0.0, 1.0]) for _ in range(5)]
    inputs[1][0] = 0.0
    inputs[2][0] = 0.0
    inputs[3][0] = 0.0
    inputs[4][0] = 0.0
    profile = SpeedProfile.from_arrays(*inputs)
    for field in (
        "position_m",
        "speed_mps",
        "time_s",
        "segment_acceleration_mps2",
        "propulsion_energy_kj",
        "levitation_energy_kj",
    ):
        with pytest.raises(ValueError):
            getattr(profile, field)[0] = 10.0
    inputs[0][0] = 99.0
    assert profile.position_m[0] == 0.0
