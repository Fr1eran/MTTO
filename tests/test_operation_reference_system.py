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
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)


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


def test_min_operation_time_matches_sum_of_operations(reference_context):
    vehicle, track, gamma = reference_context
    train_start = float(track.speed_limit_intervals[0])
    train_end = float(track.speed_limit_intervals[-1])

    rng = np.random.default_rng(123)
    for _ in range(20):
        begin_pos = float(rng.uniform(train_start, train_end))
        begin_speed = float(rng.uniform(0.0, 80.0 / 3.6))
        end_pos = float(rng.uniform(begin_pos, train_end))

        acc_arr, time_arr = _min_runtime_operations_jitted(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
        )
        total_time_sum = float(np.sum(time_arr))
        time_numba = _min_operation_time_jitted(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
        )
        time_python = min_operation_time(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, 0.0
        )

        np.testing.assert_allclose(time_numba, total_time_sum, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(time_python, total_time_sum, rtol=0.0, atol=1e-9)


def test_min_runtime_operations_kinematic_consistency(reference_context):
    vehicle, track, gamma = reference_context
    train_start = float(track.speed_limit_intervals[0])
    train_end = float(track.speed_limit_intervals[-1])

    rng = np.random.default_rng(2024)
    cases = []
    for _ in range(30):
        begin_pos = float(rng.uniform(train_start, train_end))
        begin_speed = float(rng.uniform(0.0, 100.0 / 3.6))
        end_pos = float(rng.uniform(begin_pos, train_end))
        cases.append((begin_pos, begin_speed, end_pos, 0.0))
    cases.extend(
        [
            (train_start, 0.0, train_end, 0.0),
            (train_start, 50.0 / 3.6, train_end, 0.0),
            (train_end - 100.0, 10.0, train_end, 0.0),
        ]
    )

    for begin_pos, begin_speed, end_pos, end_speed in cases:
        acc_arr, time_arr = _min_runtime_operations_jitted(
            vehicle, track, gamma, begin_pos, begin_speed, end_pos, end_speed
        )
        # 验证所有工况加速度均在合理物理边界内
        for a in acc_arr:
            assert np.isclose(a, vehicle.max_acc, atol=1e-6) or \
                   np.isclose(a, vehicle.max_dec, atol=1e-6) or \
                   np.isclose(a, 0.0, atol=1e-6)

        # 积分运动学还原位移与速度
        cur_p = begin_pos
        cur_v = begin_speed
        for a, t in zip(acc_arr, time_arr, strict=True):
            assert t >= 0.0
            cur_p += cur_v * t + 0.5 * a * t**2
            cur_v += a * t

        if abs(end_pos - begin_pos) > 1e-3:
            np.testing.assert_allclose(cur_p, end_pos, rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(cur_v, end_speed, rtol=1e-4, atol=1e-3)


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
