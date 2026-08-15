from collections.abc import Sequence

import numpy as np
import pytest

from model.common import (
    ECC,
    max_energy_and_min_operation_time,
    min_operation_time,
    min_operation_time_curve,
    min_operation_time_numba,
    min_runtime_operations_numba,
)
from model.common.operation_reference_system import (
    _calc_min_runtime_operation_reference,
    _find_speed_rise_entry_and_fall_reference,
)
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)
from utils.indexing_utils import find_speed_rise_entry_and_fall


@pytest.fixture(scope="module")
def reference_context():
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    aps, dps = load_auxiliary_stopping_areas_ap_and_dp()
    _ = load_stations_goal_positions()

    track = TrackInfo(
        slopes,
        slope_intervals,
        speed_limits.tolist(),
        speed_limit_intervals,
        ASA_aps=aps,
        ASA_dps=dps,
    )
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    return vehicle, track, 0.95


def _sum_operation_time(operations: Sequence[tuple[object, float]]) -> float:
    return float(sum(float(time) for _, time in operations))


def _min_runtime_operations_jitted(
    vehicle: VehicleInfo,
    track: TrackInfo,
    gamma: float,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    return min_runtime_operations_numba(
        begin_pos,
        begin_speed,
        end_pos,
        end_speed,
        track.speed_limits,
        track.speed_limit_intervals,
        float(gamma),
        float(vehicle.max_acc),
        float(vehicle.max_dec),
        float(vehicle.max_dec_abs),
        1e-9,
    )


def _min_operation_time_jitted(
    vehicle: VehicleInfo,
    track: TrackInfo,
    gamma: float,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> float:
    return min_operation_time_numba(
        begin_pos,
        begin_speed,
        end_pos,
        end_speed,
        track.speed_limits,
        track.speed_limit_intervals,
        float(gamma),
        float(vehicle.max_acc),
        float(vehicle.max_dec),
        float(vehicle.max_dec_abs),
        1e-9,
    )


def test_find_speed_rise_entry_and_fall_reference_matches_indexing_utils(
    reference_context,
):
    vehicle, track, gamma = reference_context
    _ = vehicle
    n_limits = track.speed_limits.size
    start_idx = max(0, n_limits // 10)
    end_idx = min(n_limits - 1, start_idx + max(3, n_limits // 4))

    expected_rise, expected_fall = find_speed_rise_entry_and_fall(
        speed_limits=track.speed_limits,
        interval_points=track.speed_limit_intervals,
        start_idx=start_idx,
        end_idx=end_idx,
        speed_factor=gamma,
    )
    actual_rise, actual_fall = _find_speed_rise_entry_and_fall_reference(
        track.speed_limits,
        track.speed_limit_intervals,
        factor=gamma,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    assert [
        (point.begin_pos, point.begin_speed, point.begin_interval)
        for point in actual_rise
    ] == [
        (entry.boundary_pos, entry.left_speed_scaled, entry.next_interval)
        for entry in expected_rise
    ]
    assert [
        (point.end_pos, point.end_speed, point.end_interval) for point in actual_fall
    ] == [
        (entry.boundary_pos, entry.right_speed_scaled, entry.prev_interval)
        for entry in expected_fall
    ]


def test_min_operation_time_matches_reference_sum(reference_context):
    vehicle, track, gamma = reference_context
    train_start = float(track.speed_limit_intervals[0])
    train_end = float(track.speed_limit_intervals[-1])

    rng = np.random.default_rng(123)
    for _ in range(16):
        begin_pos = float(rng.uniform(train_start, train_end))
        begin_speed = float(rng.uniform(0.0, 80.0 / 3.6))
        end_pos = float(rng.uniform(begin_pos, train_end))

        actual = min_operation_time(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
        )
        expected = _sum_operation_time(
            _calc_min_runtime_operation_reference(
                vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)


def test_min_runtime_operations_numba_matches_reference(reference_context):
    vehicle, track, gamma = reference_context
    train_start = float(track.speed_limit_intervals[0])
    train_end = float(track.speed_limit_intervals[-1])

    rng = np.random.default_rng(2024)
    cases = []
    for _ in range(40):
        begin_pos = float(rng.uniform(train_start, train_end))
        begin_speed = float(rng.uniform(0.0, 140.0))
        end_pos = float(rng.uniform(begin_pos, train_end))
        cases.append((begin_pos, begin_speed, end_pos, 0.0))
    cases.extend(
        [
            (train_start, 0.0, train_end, 0.0),
            (train_start, 140.0, train_end, 0.0),
            (train_end - 1.0, 0.0, train_end, 0.0),
        ]
    )

    for begin_pos, begin_speed, end_pos, end_speed in cases:
        expected = _calc_min_runtime_operation_reference(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, end_speed
        )
        acc_arr, time_arr = _min_runtime_operations_jitted(
            vehicle,
            track,
            gamma,
            begin_pos,
            begin_speed,
            end_pos,
            end_speed,
        )
        assert len(acc_arr) == len(expected)
        np.testing.assert_allclose(
            acc_arr, [op.acc for op in expected], rtol=0.0, atol=1e-9
        )
        np.testing.assert_allclose(
            time_arr, [op.operation_time for op in expected], rtol=0.0, atol=1e-9
        )


def test_min_operation_time_numba_matches_reference(reference_context):
    vehicle, track, gamma = reference_context
    train_start = float(track.speed_limit_intervals[0])
    train_end = float(track.speed_limit_intervals[-1])

    rng = np.random.default_rng(2025)
    for _ in range(40):
        begin_pos = float(rng.uniform(train_start, train_end))
        begin_speed = float(rng.uniform(0.0, 140.0))
        end_pos = float(rng.uniform(begin_pos, train_end))

        expected = _sum_operation_time(
            _calc_min_runtime_operation_reference(
                vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
            )
        )
        actual = _min_operation_time_jitted(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)


def test_min_operation_time_curve_matches_jitted_operations(reference_context):
    vehicle, track, gamma = reference_context
    begin_pos = float(track.speed_limit_intervals[0])
    end_pos = float(track.speed_limit_intervals[-1])
    begin_speed = 0.0

    acc_arr, time_arr = _min_runtime_operations_jitted(
        vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
    )
    expected_positions = np.array([begin_pos], dtype=np.float64)
    expected_speeds = np.array([begin_speed], dtype=np.float64)
    for acc, operation_time in zip(acc_arr, time_arr, strict=True):
        acc_value = float(acc)
        operation_time_value = float(operation_time)
        if operation_time_value <= 0:
            continue
        dt = 0.1
        n_steps = max(int(np.floor(operation_time_value / dt)), 2)
        t_samples = np.linspace(
            0.0, operation_time_value, n_steps, endpoint=True, dtype=np.float64
        )
        speeds = begin_speed + acc_value * t_samples
        positions = begin_pos + begin_speed * t_samples + 0.5 * acc_value * t_samples**2
        expected_positions = np.concatenate((expected_positions[:-1], positions))
        expected_speeds = np.concatenate((expected_speeds[:-1], speeds))
        begin_pos = float(expected_positions[-1])
        begin_speed = float(expected_speeds[-1])

    if expected_positions.size > 1:
        keep_mask = np.empty(expected_positions.size, dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = np.diff(expected_positions) != 0.0
        expected_positions = expected_positions[keep_mask]
        expected_speeds = expected_speeds[keep_mask]

    actual_positions, actual_speeds = min_operation_time_curve(
        vehicle,
        track,
        gamma,
        float(track.speed_limit_intervals[0]),
        0.0,
        end_pos,
        0.0,
    )
    np.testing.assert_allclose(
        actual_positions, expected_positions, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(actual_speeds, expected_speeds, rtol=0.0, atol=1e-9)


def test_max_energy_and_min_operation_time_consistent(reference_context):
    vehicle, track, gamma = reference_context
    ecc = ECC(
        R_m=0.2796,
        L_d=0.0002,
        R_k=50.0,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.8,
    )
    begin_pos = float(track.speed_limit_intervals[0])
    end_pos = float(track.speed_limit_intervals[-1])
    distance = end_pos - begin_pos

    mec, lec, total_time = max_energy_and_min_operation_time(
        vehicle,
        track,
        gamma,
        ecc,
        begin_pos,
        0.0,
        end_pos,
        0.0,
        distance,
    )
    min_time = min_operation_time(vehicle, track, gamma, begin_pos, 0.0, end_pos, 0.0)

    assert mec >= 0.0
    assert lec >= 0.0
    np.testing.assert_allclose(total_time, min_time, rtol=0.0, atol=1e-9)

    _, _, partial_time = max_energy_and_min_operation_time(
        vehicle,
        track,
        gamma,
        ecc,
        begin_pos,
        0.0,
        end_pos,
        0.0,
        0.6 * distance,
    )
    assert 0.0 < partial_time < total_time
