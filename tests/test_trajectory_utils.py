import numpy as np
import pytest

from utils.trajectory import recover_time_axis_from_trajectory, smooth_trajectory


def test_smooth_trajectory_returns_expected_length_and_endpoints() -> None:
    pos = np.asarray([0.0, 100.0, 200.0], dtype=np.float64)
    speed = np.asarray([0.0, 20.0, 0.0], dtype=np.float64)

    smooth_pos, smooth_speed = smooth_trajectory(
        pos,
        speed,
        samples_per_segment=5,
    )

    assert smooth_pos.dtype == np.float64
    assert smooth_speed.dtype == np.float64
    assert smooth_pos.size == smooth_speed.size
    assert smooth_pos.size == 9
    assert smooth_pos[0] == pytest.approx(pos[0])
    assert smooth_pos[-1] == pytest.approx(pos[-1])
    assert smooth_speed[0] == pytest.approx(speed[0])
    assert smooth_speed[-1] == pytest.approx(speed[-1])


def test_smooth_trajectory_deduplicates_consecutive_positions() -> None:
    pos = np.asarray([0.0, 10.0, 10.0, 20.0], dtype=np.float64)
    speed = np.asarray([0.0, 5.0, 6.0, 0.0], dtype=np.float64)

    smooth_pos, smooth_speed = smooth_trajectory(
        pos,
        speed,
        samples_per_segment=4,
        remove_duplicate_pos=True,
    )

    assert smooth_pos.size == 7
    assert smooth_pos[0] == pytest.approx(0.0)
    assert smooth_pos[-1] == pytest.approx(20.0)
    assert np.all(np.isfinite(smooth_speed))


def test_smooth_trajectory_returns_copy_for_short_input() -> None:
    pos = np.asarray([12.0], dtype=np.float64)
    speed = np.asarray([3.0], dtype=np.float64)

    smooth_pos, smooth_speed = smooth_trajectory(pos, speed)

    assert smooth_pos.size == 1
    assert smooth_speed.size == 1
    assert smooth_pos[0] == pytest.approx(12.0)
    assert smooth_speed[0] == pytest.approx(3.0)
    assert smooth_pos is not pos
    assert smooth_speed is not speed


@pytest.mark.parametrize(
    "pos,speed,error_msg",
    [
        (np.asarray([[0.0, 1.0]]), np.asarray([0.0, 1.0]), "1-D"),
        (np.asarray([0.0, 1.0]), np.asarray([[0.0, 1.0]]), "1-D"),
        (np.asarray([0.0, 1.0]), np.asarray([0.0]), "equal length"),
    ],
)
def test_smooth_trajectory_validates_shapes(
    pos: np.ndarray,
    speed: np.ndarray,
    error_msg: str,
) -> None:
    with pytest.raises(ValueError, match=error_msg):
        smooth_trajectory(pos, speed)


def test_smooth_trajectory_validates_options() -> None:
    pos = np.asarray([0.0, 1.0], dtype=np.float64)
    speed = np.asarray([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match=">= 2"):
        smooth_trajectory(pos, speed, samples_per_segment=1)

    with pytest.raises(ValueError, match=">= 0"):
        smooth_trajectory(pos, speed, duplicate_tolerance=-1.0)

    with pytest.raises(ValueError, match="Unknown method"):
        smooth_trajectory(pos, speed, method="pchip")


def test_recover_time_axis_from_trajectory_matches_uniform_motion() -> None:
    pos = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
    speed = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    time_arr = recover_time_axis_from_trajectory(pos, speed)

    np.testing.assert_allclose(time_arr, np.asarray([0.0, 1.0, 3.0]))


def test_recover_time_axis_from_trajectory_handles_repeated_position() -> None:
    pos = np.asarray([0.0, 0.0, 5.0], dtype=np.float64)
    speed = np.asarray([0.0, 0.0, 2.0], dtype=np.float64)

    time_arr = recover_time_axis_from_trajectory(pos, speed)

    assert time_arr[0] == pytest.approx(0.0)
    assert time_arr[1] == pytest.approx(0.0)
    assert time_arr[2] > time_arr[1]


@pytest.mark.parametrize(
    "pos,speed,error_msg",
    [
        (np.asarray([[0.0, 1.0]]), np.asarray([0.0, 1.0]), "1-D"),
        (np.asarray([0.0, 1.0]), np.asarray([[0.0, 1.0]]), "1-D"),
        (np.asarray([0.0, 1.0]), np.asarray([0.0]), "equal length"),
    ],
)
def test_recover_time_axis_from_trajectory_validates_shapes(
    pos: np.ndarray,
    speed: np.ndarray,
    error_msg: str,
) -> None:
    with pytest.raises(ValueError, match=error_msg):
        recover_time_axis_from_trajectory(pos, speed)


def test_recover_time_axis_from_trajectory_validates_tolerances() -> None:
    pos = np.asarray([0.0, 1.0], dtype=np.float64)
    speed = np.asarray([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match=">= 0"):
        recover_time_axis_from_trajectory(pos, speed, position_tolerance=-1.0)

    with pytest.raises(ValueError, match=">= 0"):
        recover_time_axis_from_trajectory(pos, speed, speed_tolerance=-1.0)
