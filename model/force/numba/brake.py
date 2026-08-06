import numpy as np
from numba import (
    njit,
)


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


__all__ = [
    "sledge_frictional_brake_force_numba",
    "vortex_brake_force_numba",
    "wear_plate_frictional_brake_force_numba",
]
