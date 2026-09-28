from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    "Motion",
    "accel_between",
    "run_distance",
    "run_time",
    "segment_acceleration",
]


class Motion(NamedTuple):
    v1_mps: float
    distance_m: float
    duration_s: float


def segment_acceleration(
    speed_mps: NDArray[np.float64], time_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return each segment's speed change divided by elapsed time."""
    with np.errstate(invalid="ignore"):
        dv = np.diff(speed_mps.ravel())
        dt = np.diff(time_s.ravel())
        return np.divide(dv, dt, out=np.zeros_like(dv), where=dt > 0)


@njit(cache=True)
def run_distance(v0: float, a: float, ds: float) -> tuple[float, float, float]:
    """Advance one RL-style constant-acceleration distance step.

    Returns ``(next_speed_mps, actual_distance_m, duration_s)``.  A braking
    step that would pass below zero speed is shortened to its stopping point.
    """
    acc_tolerance = 1e-6
    speed_tolerance = 1e-6

    if abs(a) < acc_tolerance:
        next_speed_mps = v0
        if next_speed_mps < speed_tolerance:
            return 0.0, 0.0, 0.0
        return (
            next_speed_mps,
            ds,
            ds / next_speed_mps,
        )

    next_speed_squared = v0 * v0 + 2.0 * a * ds
    actual_distance_m = ds
    if next_speed_squared < speed_tolerance:
        next_speed_mps = 0.0
        actual_distance_m = -(v0 * v0) / (2.0 * a)
    else:
        next_speed_mps = math.sqrt(next_speed_squared)

    duration_s = (next_speed_mps - v0) / a
    return next_speed_mps, actual_distance_m, duration_s


@njit(cache=True)
def accel_between(v0: float, v1: float, ds: float) -> tuple[float, float]:
    """Infer constant acceleration and duration from a DP-style edge.

    Callers must validate non-zero displacement and a non-zero speed sum before
    calling this function.
    """
    acceleration_mps2 = (v1 * v1 - v0 * v0) / (2.0 * ds)

    if abs(acceleration_mps2) < 1e-9:
        duration_s = abs(ds) / v0
    else:
        duration_s = (v1 - v0) / acceleration_mps2

    return acceleration_mps2, duration_s


@njit(cache=True)
def run_time(v0: float, a: float, dt: float) -> tuple[float, float, float]:
    """以速度 v0、加速度 a 运行时间 dt，返回 (末速度, 距离, 时长)。"""
    acc_tolerance = 1e-6
    speed_tolerance = 1e-6

    if abs(a) < acc_tolerance:
        if v0 < speed_tolerance:
            return 0.0, 0.0, 0.0
        return v0, v0 * dt, dt

    if a < 0.0:
        if v0 < speed_tolerance:
            return 0.0, 0.0, 0.0
        t_stop = v0 / (-a)
        if dt >= t_stop:
            actual_distance = 0.5 * (v0 * v0) / (-a)
            return 0.0, actual_distance, t_stop

    v1 = v0 + a * dt
    actual_distance = v0 * dt + 0.5 * a * dt * dt
    return v1, actual_distance, dt
