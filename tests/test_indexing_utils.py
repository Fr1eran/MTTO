import numpy as np

from utils.indexing_utils import find_speed_rise_entry_and_fall


def test_find_speed_rise_entry_and_fall_default_and_speed_factor():
    speed_limits = np.array([10.0, 10.0, 20.0, 20.0, 15.0, 15.0, 25.0])
    interval_points = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    rise_entries, fall_exits = find_speed_rise_entry_and_fall(
        speed_limits=speed_limits,
        interval_points=interval_points,
        speed_factor=2.5,
    )

    assert [entry.next_interval for entry in rise_entries] == [2, 6]
    assert [entry.boundary_pos for entry in rise_entries] == [2.0, 6.0]
    np.testing.assert_allclose(
        [entry.left_speed_scaled for entry in rise_entries],
        [25.0, 37.5],
    )

    assert [entry.prev_interval for entry in fall_exits] == [3]
    assert [entry.boundary_pos for entry in fall_exits] == [4.0]
    np.testing.assert_allclose(
        [entry.right_speed_scaled for entry in fall_exits],
        [37.5],
    )


def test_find_speed_rise_entry_and_fall_sub_range():
    speed_limits = np.array([10.0, 10.0, 20.0, 20.0, 15.0, 15.0, 25.0])
    interval_points = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    rise_entries, fall_exits = find_speed_rise_entry_and_fall(
        speed_limits=speed_limits,
        interval_points=interval_points,
        start_idx=2,
        end_idx=6,
        speed_factor=1.0,
    )

    assert [entry.next_interval for entry in rise_entries] == [6]
    assert [entry.prev_interval for entry in fall_exits] == [3]
    assert all(2 <= entry.next_interval - 1 < 6 for entry in rise_entries)
    assert all(2 <= entry.prev_interval < 6 for entry in fall_exits)


def test_find_speed_rise_entry_and_fall_empty_when_range_invalid():
    speed_limits = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    interval_points = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    rise_entries, fall_exits = find_speed_rise_entry_and_fall(
        speed_limits=speed_limits,
        interval_points=interval_points,
        start_idx=3,
        end_idx=1,
    )
    assert rise_entries == []
    assert fall_exits == []
