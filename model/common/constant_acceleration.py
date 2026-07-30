"""Scalar constant-acceleration kinematics shared by DP and RL."""

from __future__ import annotations

import math

from numba import njit

__all__ = [
    "calc_transition_from_acc_scalar_numba",
    "calc_transition_to_speed_scalar_numba",
]


@njit(cache=True)
def calc_transition_from_acc_scalar_numba(
    begin_speed_mps: float,
    acceleration_mps2: float,
    requested_distance_m: float,
) -> tuple[float, float, float]:
    """Advance one RL-style constant-acceleration distance step.

    Returns ``(next_speed_mps, actual_distance_m, duration_s)``.  A braking
    step that would pass below zero speed is shortened to its stopping point.
    """
    acc_tolerance = 1e-6
    speed_tolerance = 1e-6

    if abs(acceleration_mps2) < acc_tolerance:
        next_speed_mps = begin_speed_mps
        if next_speed_mps < speed_tolerance:
            return 0.0, 0.0, 0.0
        return (
            next_speed_mps,
            requested_distance_m,
            requested_distance_m / next_speed_mps,
        )

    next_speed_squared = (
        begin_speed_mps * begin_speed_mps
        + 2.0 * acceleration_mps2 * requested_distance_m
    )
    actual_distance_m = requested_distance_m
    if next_speed_squared < speed_tolerance:
        next_speed_mps = 0.0
        actual_distance_m = -(
            begin_speed_mps * begin_speed_mps
        ) / (2.0 * acceleration_mps2)
    else:
        next_speed_mps = math.sqrt(next_speed_squared)

    duration_s = (next_speed_mps - begin_speed_mps) / acceleration_mps2
    return next_speed_mps, actual_distance_m, duration_s


@njit(cache=True)
def calc_transition_to_speed_scalar_numba(
    begin_speed_mps: float,
    end_speed_mps: float,
    displacement_m: float,
) -> tuple[float, float]:
    """Infer constant acceleration and duration from a DP-style edge.

    Callers must validate non-zero displacement and a non-zero speed sum before
    calling this function.
    """
    acceleration_mps2 = (
        end_speed_mps * end_speed_mps - begin_speed_mps * begin_speed_mps
    ) / (2.0 * displacement_m)

    if abs(acceleration_mps2) < 1e-9:
        duration_s = abs(displacement_m) / begin_speed_mps
    else:
        duration_s = (end_speed_mps - begin_speed_mps) / acceleration_mps2

    return acceleration_mps2, duration_s
