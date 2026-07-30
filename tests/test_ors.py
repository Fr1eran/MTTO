import numpy as np
import pytest

from model.vehicle import VehicleInfo
from model.track import TrackInfo
from model.ocs import TrainService
from model.common import ORS
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)
from utils.indexing_utils import find_speed_rise_entry_and_fall


@pytest.fixture(scope="module")
def operation_reference_system():
    # 坡度，百分位
    slopes, slope_intervals = load_slopes()

    # 区间限速
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

    aps, dps = load_auxiliary_stopping_areas_ap_and_dp()

    # 车站
    ly_zp, pa_zp = load_stations_goal_positions()

    track = TrackInfo(
        slopes,
        slope_intervals,
        speed_limits.tolist(),
        speed_limit_intervals,
        ASA_aps=aps,
        ASA_dps=dps,
    )
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    train_service = TrainService(
        start_position=ly_zp,
        start_speed=0.0,
        target_position=pa_zp,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=120.0,
        max_stop_error=0.3,
    )
    return ORS(vehicle=vehicle, track=track, factor=0.95)


def _sum_operation_time(operations) -> float:
    return float(sum(float(time) for _, time in operations))


def test_find_speed_rise_entry_and_fall_is_stable_under_repeated_calls(
    operation_reference_system: ORS,
):
    ors = operation_reference_system
    n_limits = len(ors.track.speed_limits)
    start_idx = max(0, n_limits // 8)
    end_idx = min(n_limits - 1, start_idx + max(2, n_limits // 5))

    first = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )
    second = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Interleave another range query; same-range result should remain unchanged.
    _ = ors._find_speed_rise_entry_and_fall(
        start_idx=max(0, start_idx - 2),
        end_idx=min(n_limits - 1, end_idx + 2),
    )
    third = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )

    assert first == second
    assert second == third


def test_find_speed_rise_entry_and_fall_matches_indexing_utils(
    operation_reference_system: ORS,
):
    ors = operation_reference_system
    n_limits = len(ors.track.speed_limits)
    start_idx = max(0, n_limits // 10)
    end_idx = min(n_limits - 1, start_idx + max(3, n_limits // 4))

    rise_entries, fall_exits = find_speed_rise_entry_and_fall(
        speed_limits=ors.track.speed_limits,
        interval_points=ors.track.speed_limit_intervals,
        start_idx=start_idx,
        end_idx=end_idx,
        speed_factor=ors.gamma,
    )
    expected_rise = [
        (entry.boundary_pos, entry.left_speed_scaled, entry.next_interval)
        for entry in rise_entries
    ]
    expected_fall = [
        (entry.boundary_pos, entry.right_speed_scaled, entry.prev_interval)
        for entry in fall_exits
    ]

    actual_rise_points, actual_fall_points = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )
    actual_rise = [
        (point.begin_pos, point.begin_speed, point.begin_interval)
        for point in actual_rise_points
    ]
    actual_fall = [
        (point.end_pos, point.end_speed, point.end_interval)
        for point in actual_fall_points
    ]

    assert actual_rise == expected_rise
    assert actual_fall == expected_fall


def test_calc_min_operation_time_matches_runtime_operation_sum(
    operation_reference_system: ORS,
):
    ors = operation_reference_system
    train_start = float(ors.track.speed_limit_intervals[0])
    train_end = float(ors.track.speed_limit_intervals[-1])
    target_end = float(
        max(train_start, min(train_end, ors.track.speed_limit_intervals[-1] - 1.0))
    )

    rng = np.random.default_rng(123)
    for _ in range(16):
        begin_pos = float(rng.uniform(train_start, target_end))
        begin_speed = float(rng.uniform(0.0, 80.0 / 3.6))
        end_pos = float(rng.uniform(begin_pos, train_end))
        end_speed = 0.0

        runtime_time = ors.calc_min_operation_time(
            begin_pos=begin_pos,
            begin_speed=begin_speed,
            end_pos=end_pos,
            end_speed=end_speed,
        )
        operations = ors._calc_min_runtime_operation(
            current_pos=begin_pos,
            current_speed=begin_speed,
            end_pos=end_pos,
            end_speed=end_speed,
        )
        operations_time = _sum_operation_time(operations)
        np.testing.assert_allclose(runtime_time, operations_time, rtol=0.0, atol=1e-9)


def test_ors_does_not_own_dp_trajectory_loading() -> None:
    assert not hasattr(ORS, "load_or_build_ref_redundant_operation_time_from_dp")

