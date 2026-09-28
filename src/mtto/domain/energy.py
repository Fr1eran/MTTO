from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from mtto.domain.dynamics import Vehicle, calc_longitudinal_force_scalar_numba
from mtto.domain.line import Line, get_slope_scalar_numba

__all__ = [
    "EnergyParams",
    "_calc_energy_constant_acc_numba",
    "segment_energy",
    "profile_energy",
]


@dataclass(frozen=True, slots=True)
class EnergyParams:
    """ECC parameters: 7 constructor arguments plus Phi_1 and Phi_2."""

    R_m: float
    L_d: float
    R_k: float
    L_k: float
    Tau: float
    Psi_fd: float
    k_c: float
    Phi_1: float
    Phi_2: float

    def __post_init__(self) -> None:
        values = (
            self.R_m,
            self.L_d,
            self.R_k,
            self.L_k,
            self.Tau,
            self.Psi_fd,
            self.k_c,
            self.Phi_1,
            self.Phi_2,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("All EnergyParams values must be finite")


@njit(cache=True)
def _calc_energy_constant_acc_numba(
    begin_pos: float,
    begin_speed: float,
    acc: float,
    distance: float,
    direction: int,
    operation_time_value: float,
    mass: float,
    numoftrainsets: float,
    slopes: NDArray[np.float64],
    slope_intervals: NDArray[np.float64],
    r_m: float,
    l_d: float,
    r_k: float,
    l_k: float,
    k_c: float,
    h: float,
    phi_1: float,
    phi_2: float,
) -> tuple[float, float]:
    mechanic_energy_consumption = 0.0
    motor_energy_consumption = 0.0
    abs_distance = np.abs(distance)

    if abs_distance < 1e-6:
        slope = get_slope_scalar_numba(begin_pos, slopes, slope_intervals)
        f_longitudinal = calc_longitudinal_force_scalar_numba(
            begin_speed, slope, acc, mass, numoftrainsets
        )
        mechanic_energy_consumption = np.abs(f_longitudinal * distance)
    else:
        n_samples = int(abs_distance / 1.0)
        if n_samples < 10:
            n_samples = 10

        delta_d = distance / n_samples
        abs_delta_d = np.abs(delta_d)
        motor_r_coeff = (
            2.0 / (3.0 * h**2) * (r_m + k_c**2 * r_k + (1.0 - k_c) ** 2 * r_k)
        )
        motor_l_coeff = (
            2.0 / (3.0 * h**2) * (l_d + k_c**2 * l_k + (1.0 - k_c) ** 2 * l_k)
        )

        speed = begin_speed
        time_current = 0.0
        slope = get_slope_scalar_numba(begin_pos, slopes, slope_intervals)
        f_current = calc_longitudinal_force_scalar_numba(
            speed, slope, acc, mass, numoftrainsets
        )
        abs_f_current = np.abs(f_current)
        motor_r_current = f_current**2 * motor_r_coeff
        motor_l_current = abs_f_current * motor_l_coeff

        for i in range(n_samples):
            next_speed_squared = speed**2 + 2.0 * acc * delta_d
            if next_speed_squared < 0.0:
                next_speed_squared = 0.0
            speed_next = np.sqrt(next_speed_squared)

            avg_speed = (speed + speed_next) / 2.0
            if avg_speed < 1e-6:
                avg_speed = 1e-6
            time_next = time_current + abs_delta_d / avg_speed

            d_next = delta_d * (i + 1)
            pos_next = begin_pos + d_next * direction
            slope_next = get_slope_scalar_numba(pos_next, slopes, slope_intervals)
            f_next = calc_longitudinal_force_scalar_numba(
                speed_next,
                slope_next,
                acc,
                mass,
                numoftrainsets,
            )
            abs_f_next = np.abs(f_next)
            motor_r_next = f_next**2 * motor_r_coeff
            motor_l_next = abs_f_next * motor_l_coeff

            mechanic_energy_consumption += (
                0.5 * (abs_f_current + abs_f_next) * abs_delta_d
            )
            motor_energy_consumption += (
                0.5 * (motor_r_current + motor_r_next) * (time_next - time_current)
            )
            motor_energy_consumption += (
                0.5 * (motor_l_current + motor_l_next) * (abs_f_next - abs_f_current)
            )

            speed = speed_next
            time_current = time_next
            f_current = f_next
            abs_f_current = abs_f_next
            motor_r_current = motor_r_next
            motor_l_current = motor_l_next

    if np.isnan(operation_time_value):
        if np.abs(acc) < 1e-9:
            speed_denom = begin_speed
            if speed_denom < 1e-6:
                speed_denom = 1e-6
            time = distance / speed_denom
        else:
            next_speed_squared = begin_speed**2 + 2.0 * acc * distance
            if next_speed_squared < 0.0:
                next_speed_squared = 0.0
            next_speed = np.sqrt(next_speed_squared)
            time = (next_speed - begin_speed) / acc
    else:
        time = operation_time_value

    propulsion_energy_consumption = (
        mechanic_energy_consumption + motor_energy_consumption
    )
    leviation_energy_consumption = phi_1 * distance + phi_2 * mass * time

    return propulsion_energy_consumption, leviation_energy_consumption


def segment_energy(
    params: EnergyParams,
    vehicle: Vehicle,
    line: Line,
    *,
    begin_pos: float,
    begin_speed: float,
    acc: float,
    distance: float,
    direction: int,
    operation_time: float,
) -> tuple[float, float]:
    """计算单个分段常加速度运动的牵引与悬浮能耗 (kJ)。"""
    h = math.pi * params.Psi_fd / params.Tau
    return _calc_energy_constant_acc_numba(
        begin_pos,
        begin_speed,
        acc,
        distance,
        direction,
        operation_time,
        vehicle.mass,
        vehicle.numoftrainsets,
        line.slopes,
        line.slope_intervals,
        params.R_m,
        params.L_d,
        params.R_k,
        params.L_k,
        params.k_c,
        h,
        params.Phi_1,
        params.Phi_2,
    )


def profile_energy(
    params: EnergyParams,
    vehicle: Vehicle,
    line: Line,
    position_m: NDArray[np.float64],
    speed_mps: NDArray[np.float64],
    time_s: NDArray[np.float64],
    segment_acceleration_mps2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate propulsion and levitation energy over every segment."""
    propulsion = np.zeros(position_m.size, dtype=np.float64)
    levitation = np.zeros(position_m.size, dtype=np.float64)
    for index, acceleration in enumerate(segment_acceleration_mps2):
        prop_delta, lev_delta = segment_energy(
            params,
            vehicle,
            line,
            begin_pos=float(position_m[index]),
            begin_speed=float(speed_mps[index]),
            acc=float(acceleration),
            distance=float(position_m[index + 1] - position_m[index]),
            direction=1,
            operation_time=float(time_s[index + 1] - time_s[index]),
        )
        propulsion[index + 1] = propulsion[index] + prop_delta
        levitation[index + 1] = levitation[index] + lev_delta
    return propulsion, levitation
