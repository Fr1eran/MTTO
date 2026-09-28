import numpy as np
import pytest

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import EnergyParams
from mtto.domain.line import Line
from mtto.domain.srtsp import (
    build_srtsp_curve,
    build_srtsp_lookup,
    interp_upper_speed,
    lookup_upper_speed,
    lookup_upper_speed_or_zero,
    max_energy_and_min_operation_time,
    min_operation_time,
    min_operation_time_curve,
    min_operation_time_numba,
    min_runtime_operations_numba,
)


@pytest.fixture(scope="module")
def reference_context(paper_scenario):
    track: Line = paper_scenario.line
    vehicle = Vehicle(
        mass=317.5,
        numoftrainsets=5,
        length=128.5,
        max_speed=500.0 / 3.6,
        max_acc=1.0,
        max_dec=-1.0,
        max_slope_capacity=4.0,
        levi_power_per_mass=1.7,
    )
    return vehicle, track, 0.95


def _min_runtime_operations_jitted(
    vehicle: Vehicle,
    track: Line,
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
    vehicle: Vehicle,
    track: Line,
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
            assert (
                np.isclose(a, vehicle.max_acc, atol=1e-6)
                or np.isclose(a, vehicle.max_dec, atol=1e-6)
                or np.isclose(a, 0.0, atol=1e-6)
            )

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
    energy = EnergyParams(
        R_m=0.2796,
        L_d=0.00292,
        R_k=0.0736,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.5,
        Phi_1=0.1049,
        Phi_2=1.006,
    )
    begin_pos = float(track.speed_limit_intervals[0])
    end_pos = float(track.speed_limit_intervals[-1])
    distance = end_pos - begin_pos

    mec, lec, total_time = max_energy_and_min_operation_time(
        vehicle,
        track,
        gamma,
        energy,
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
        energy,
        begin_pos,
        0.0,
        end_pos,
        0.0,
        0.6 * distance,
    )
    assert 0.0 < partial_time < total_time


@pytest.mark.parametrize(
    ("position_m", "expected_clamped", "expected_or_zero"),
    [
        (5.0, 9.0, 9.0),
        (-1.0, 4.0, 0.0),
        (25.0, 24.0, 0.0),
        (10.0, 14.0, 14.0),
    ],
)
def test_srtsp_lookup_queries(
    position_m: float, expected_clamped: float, expected_or_zero: float
) -> None:
    lookup = build_srtsp_lookup(np.asarray([0.0, 20.0]), np.asarray([4.0, 24.0]))

    assert lookup.pos_min_m == 0.0
    assert lookup.step_m == 10.0
    assert lookup.speed_mps.dtype == np.float32
    assert not lookup.speed_mps.flags.writeable
    assert lookup_upper_speed(lookup, position_m) == pytest.approx(expected_clamped)
    assert lookup_upper_speed_or_zero(lookup, position_m) == pytest.approx(
        expected_or_zero
    )


@pytest.mark.parametrize(
    ("pos_input", "speed_input", "query_pos", "expected_speeds"),
    [
        (
            [0.0, 10.0, 20.0],
            [5.0, 15.0, 25.0],
            [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            [5.0, 5.0, 10.0, 15.0, 20.0, 25.0, 25.0],
        ),
        (
            [20.0, 10.0, 0.0],
            [25.0, 15.0, 5.0],
            [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            [5.0, 5.0, 10.0, 15.0, 20.0, 25.0, 25.0],
        ),
        (
            [0.0, 10.0, 20.0],
            [-5.0, 15.0, -2.0],
            [0.0, 5.0, 10.0, 15.0, 20.0],
            [0.0, 7.5, 15.0, 7.5, 0.0],
        ),
    ],
)
def test_srtsp_curve_interpolation(
    pos_input: list[float],
    speed_input: list[float],
    query_pos: list[float],
    expected_speeds: list[float],
) -> None:
    curve = build_srtsp_curve(pos_input, speed_input)
    actual = interp_upper_speed(curve, np.asarray(query_pos, dtype=np.float64))
    np.testing.assert_allclose(actual, expected_speeds)
    assert actual.dtype == np.float64

    for q, exp in zip(query_pos, expected_speeds, strict=True):
        val = interp_upper_speed(curve, q)
        assert float(val) == pytest.approx(exp)
        interp_exp = np.interp(q, curve.position_m, curve.speed_mps)
        assert float(val) == pytest.approx(interp_exp)


def test_srtsp_curve_properties_and_errors() -> None:
    curve = build_srtsp_curve([0.0, 10.0], [5.0, 15.0])
    assert not curve.position_m.flags.writeable
    assert not curve.speed_mps.flags.writeable
    assert curve.position_m.dtype == np.float64
    assert curve.speed_mps.dtype == np.float64

    with pytest.raises(ValueError, match="strictly increasing"):
        build_srtsp_curve([0.0, 5.0, 5.0, 10.0], [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        build_srtsp_curve([0.0, 5.0, 3.0, 10.0], [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="invalid"):
        build_srtsp_curve([], [])

    with pytest.raises(ValueError, match="invalid"):
        build_srtsp_curve([0.0, 10.0], [5.0])
