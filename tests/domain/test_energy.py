import math

import numpy as np
import pytest

from mtto.domain.dynamics import Vehicle, calc_longitudinal_force_scalar_numba
from mtto.domain.energy import EnergyParams, segment_energy
from mtto.domain.line import Line, get_slope_array_numba, get_slope_scalar_numba


@pytest.fixture(scope="module")
def energy_consumption_calculator_case():
    track = Line(
        slopes=np.asarray([0.0, 0.8, -0.4], dtype=np.float64),
        slope_intervals=np.asarray([0.0, 500.0, 1000.0, 20000.0], dtype=np.float64),
        speed_limits=np.asarray([120.0 / 3.6], dtype=np.float64),
        speed_limit_intervals=np.asarray([0.0, 20000.0], dtype=np.float64),
        accessible_points_m=(),
        danger_points_m=(),
    )
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
    energy_params = EnergyParams(
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
    return energy_params, vehicle, track


def _calc_energy_constant_acc_reference(
    energy_params: EnergyParams,
    vehicle: Vehicle,
    track: Line,
    *,
    begin_pos: float,
    begin_speed: float,
    acc: float,
    distance: float,
    direction: int,
    operation_time: float | None,
) -> tuple[float, float]:
    h = math.pi * energy_params.Psi_fd / energy_params.Tau
    if np.abs(distance) < 1e-6:
        slope = get_slope_scalar_numba(begin_pos, track.slopes, track.slope_intervals)
        f_longitudinal = calc_longitudinal_force_scalar_numba(
            begin_speed, slope, acc, vehicle.mass, vehicle.numoftrainsets
        )
        mechanic_energy_consumption = np.abs(f_longitudinal * distance)
        motor_energy_consumption = 0.0
    else:
        n_samples = max(10, int(np.abs(distance) / 1.0))
        d_nodes = np.linspace(0.0, distance, n_samples + 1)
        delta_d = np.diff(d_nodes)
        p_nodes = begin_pos + d_nodes * direction

        speed_nodes = np.empty_like(d_nodes)
        speed_nodes[0] = begin_speed
        for i in range(n_samples):
            next_speed_squared = speed_nodes[i] ** 2 + 2.0 * acc * delta_d[i]
            speed_nodes[i + 1] = np.sqrt(np.maximum(next_speed_squared, 0.0))

        t_nodes = np.zeros_like(d_nodes)
        for i in range(n_samples):
            avg_speed = np.maximum(
                (speed_nodes[i] + speed_nodes[i + 1]) / 2.0,
                1e-6,
            )
            t_nodes[i + 1] = t_nodes[i] + np.abs(delta_d[i]) / avg_speed

        slope_nodes = get_slope_array_numba(
            p_nodes, track.slopes, track.slope_intervals
        )
        f_longitudinal = np.asarray(
            [
                calc_longitudinal_force_scalar_numba(
                    speed_nodes[i],
                    slope_nodes[i],
                    acc,
                    vehicle.mass,
                    vehicle.numoftrainsets,
                )
                for i in range(n_samples + 1)
            ],
            dtype=np.float64,
        )
        mechanic_energy_consumption = np.sum(
            0.5
            * (np.abs(f_longitudinal[:-1]) + np.abs(f_longitudinal[1:]))
            * np.abs(delta_d)
        )
        motor_energy_consumption = np.trapezoid(
            y=(2 * f_longitudinal**2 / (3 * h**2))
            * (
                energy_params.R_m
                + energy_params.k_c**2 * energy_params.R_k
                + (1 - energy_params.k_c) ** 2 * energy_params.R_k
            ),
            x=t_nodes,
        ) + np.trapezoid(
            y=(np.abs(f_longitudinal) * 2 / (3 * h**2))
            * (
                energy_params.L_d
                + energy_params.k_c**2 * energy_params.L_k
                + (1 - energy_params.k_c) ** 2 * energy_params.L_k
            ),
            x=np.abs(f_longitudinal),
        )

    if operation_time is None:
        if np.abs(acc) < 1e-9:
            time = distance / np.maximum(begin_speed, 1e-6)
        else:
            next_speed_squared = begin_speed**2 + 2 * acc * distance
            next_speed = np.sqrt(np.maximum(next_speed_squared, 0))
            time = (next_speed - begin_speed) / acc
    else:
        time = operation_time

    propulsion_energy_consumption = (
        mechanic_energy_consumption + motor_energy_consumption
    )
    leviation_energy_consumption = (
        energy_params.Phi_1 * distance + energy_params.Phi_2 * vehicle.mass * time
    )
    return float(propulsion_energy_consumption), float(leviation_energy_consumption)


@pytest.mark.parametrize(
    ("begin_pos", "begin_speed", "acc", "distance", "direction", "operation_time"),
    [
        (100.0, 10.0, 0.35, 200.0, 1, None),
        (1200.0, 18.0, -0.2, 350.0, -1, None),
        (500.0, 12.0, 0.25, 100.0, 1, 12.34),
        (200.0, 7.5, 0.2, 1e-8, 1, None),
    ],
)
def test_calc_energy_constant_acc_matches_reference(
    energy_consumption_calculator_case: tuple[EnergyParams, Vehicle, Line],
    begin_pos: float,
    begin_speed: float,
    acc: float,
    distance: float,
    direction: int,
    operation_time: float | None,
):
    energy_params, vehicle, track = energy_consumption_calculator_case
    expected_pec, expected_lec = _calc_energy_constant_acc_reference(
        energy_params,
        vehicle,
        track,
        begin_pos=begin_pos,
        begin_speed=begin_speed,
        acc=acc,
        distance=distance,
        direction=direction,
        operation_time=operation_time,
    )
    pec, lec = segment_energy(
        energy_params,
        vehicle,
        track,
        begin_pos=begin_pos,
        begin_speed=begin_speed,
        acc=acc,
        distance=distance,
        direction=direction,
        operation_time=(math.nan if operation_time is None else float(operation_time)),
    )

    assert np.isfinite(pec)
    assert np.isfinite(lec)
    assert pec == pytest.approx(expected_pec, rel=1e-10, abs=1e-10)
    assert lec == pytest.approx(expected_lec, rel=1e-10, abs=1e-10)
