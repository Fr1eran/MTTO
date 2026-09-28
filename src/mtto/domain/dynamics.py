from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

__all__ = [
    "Vehicle",
    "air_resis_force_numba",
    "calc_brake_deceleration_scalar_numba",
    "calc_levi_deceleration_scalar_numba",
    "calc_longitudinal_force_scalar_numba",
    "guideway_vortex_resis_force_numba",
    "linear_generator_resis_force_numba",
    "sledge_frictional_brake_force_numba",
    "slope_resis_force_numba",
    "vortex_brake_force_numba",
    "wear_plate_frictional_brake_force_numba",
]


@dataclass(frozen=True, slots=True)
class Vehicle:
    mass: float  # 单位: T
    numoftrainsets: int
    length: float  # 单位: m
    max_speed: float  # 单位: m/s
    max_acc: float  # 单位: m/s^2
    max_dec: float  # 单位: m/s^2，物理语义为负值
    max_slope_capacity: float  # 百分位
    levi_power_per_mass: float  # 单位 kW/T

    def __post_init__(self) -> None:
        if self.max_acc <= 0.0:
            raise ValueError("max_acc must be positive")
        if self.max_dec >= 0.0:
            raise ValueError("max_dec must be negative under physical sign semantics")

    @property
    def max_dec_abs(self) -> float:
        return -self.max_dec


@njit(cache=True)
def sledge_frictional_brake_force_numba(speed: float, mass: float, slope: float):
    speed_km = 3.6 * speed
    if speed_km > 10.0:
        return 0.0
    u = -0.003 * speed_km + 0.27
    return 0.1 * u * mass * 100.0 / np.sqrt(100.0**2 + slope**2) * 9.8


@njit(cache=True)
def vortex_brake_force_numba(speed: float, numoftrainsets: float, level: int):
    speed_km = 3.6 * speed
    if speed_km <= 10.0:
        return 0.0
    x = speed_km / 200.0
    sqrt_x = np.sqrt(x)
    return (
        (7 - level)
        / 7.0
        * 2.0
        * numoftrainsets
        * 147.8
        * sqrt_x
        / (x + (1.0 + sqrt_x) ** 2)
    )


@njit(cache=True)
def wear_plate_frictional_brake_force_numba(speed: float, numoftrainsets: float):
    speed_km = 3.6 * speed
    if speed_km <= 10.0 or speed_km > 150.0:
        return 0.0

    if speed_km <= 20.0:
        mu = -0.003 * speed_km + 0.28
    elif speed_km <= 30.0:
        mu = -0.002 * speed_km + 0.26
    elif speed_km <= 50.0:
        mu = -0.001 * speed_km + 0.23
    elif speed_km <= 100.0:
        mu = -0.0008 * speed_km + 0.22
    elif speed_km <= 200.0:
        mu = -0.0002 * speed_km + 0.16
    else:
        mu = 0.3

    a = 580.32
    b = 312384.47
    c = 3.0816
    d = 227.727
    e = 42.0
    root_term = b - c * (speed_km - d) ** 2
    if root_term < 0.0:
        return 0.0
    return mu * (2.0 * numoftrainsets * (a - np.sqrt(root_term)) - e)


@njit(cache=True)
def air_resis_force_numba(speed: float, numoftrainsets: float):
    return 2.8 * (0.53 * numoftrainsets / 2.0 + 0.3) * speed**2 / 1000.0


@njit(cache=True)
def guideway_vortex_resis_force_numba(speed: float, numoftrainsets: float):
    speed_km = 3.6 * speed
    return numoftrainsets * (0.1 * speed_km**0.5 + 0.02 * speed_km**0.7)


@njit(cache=True)
def linear_generator_resis_force_numba(speed: float, numoftrainsets: float):
    speed_km = 3.6 * speed
    if speed_km < 20.0:
        return 0.0
    if speed_km < 70.0:
        return 7.3 * numoftrainsets
    if speed_km < 600.0:
        return 146.0 * 3.6 * numoftrainsets / speed_km - 0.2
    return 0.0


@njit(cache=True)
def slope_resis_force_numba(mass: float, slope: float):
    return 9.8 * mass * slope / 100.0


@njit(cache=True)
def calc_levi_deceleration_scalar_numba(
    speed: float,
    slope: float,
    mass: float,
    numoftrainsets: float,
) -> float:
    f_total = (
        sledge_frictional_brake_force_numba(speed, mass, slope)
        + air_resis_force_numba(speed, numoftrainsets)
        + guideway_vortex_resis_force_numba(speed, numoftrainsets)
        + linear_generator_resis_force_numba(speed, numoftrainsets)
        + slope_resis_force_numba(mass, slope)
    )
    return f_total / mass


@njit(cache=True)
def calc_brake_deceleration_scalar_numba(
    speed: float,
    slope: float,
    mass: float,
    numoftrainsets: float,
    level: int,
) -> float:
    f_total = (
        vortex_brake_force_numba(speed, numoftrainsets, level)
        + wear_plate_frictional_brake_force_numba(speed, numoftrainsets)
        + sledge_frictional_brake_force_numba(speed, mass, slope)
        + air_resis_force_numba(speed, numoftrainsets)
        + guideway_vortex_resis_force_numba(speed, numoftrainsets)
        + linear_generator_resis_force_numba(speed, numoftrainsets)
        + slope_resis_force_numba(mass, slope)
    )
    return f_total / mass


@njit(cache=True)
def calc_longitudinal_force_scalar_numba(
    speed: float,
    slope: float,
    acc: float,
    mass: float,
    numoftrainsets: float,
) -> float:
    f_resis = (
        air_resis_force_numba(speed, numoftrainsets)
        + guideway_vortex_resis_force_numba(speed, numoftrainsets)
        + linear_generator_resis_force_numba(speed, numoftrainsets)
        + slope_resis_force_numba(mass, slope)
    )
    return mass * acc + f_resis
