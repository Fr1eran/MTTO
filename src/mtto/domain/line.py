from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from mtto.domain._numerics import get_interval_index_scalar_numba

__all__ = [
    "Line",
    "get_slope_array_numba",
    "get_slope_scalar_numba",
    "get_speed_limit_array_numba",
    "get_speed_limit_scalar_numba",
]


@dataclass(frozen=True, slots=True, eq=False)
class Line:
    slopes: NDArray[np.float64]
    slope_intervals: NDArray[np.float64]
    speed_limits: NDArray[np.float64]  # m/s
    speed_limit_intervals: NDArray[np.float64]
    accessible_points_m: tuple[float, ...]
    danger_points_m: tuple[float, ...]


@njit(cache=True)
def get_slope_scalar_numba(
    pos: float,
    slopes: NDArray[np.float64],
    slope_intervals: NDArray[np.float64],
) -> float:
    idx = get_interval_index_scalar_numba(pos, slope_intervals)
    if idx < 0:
        idx = 0
    elif idx >= slopes.size:
        idx = slopes.size - 1
    return slopes[idx]


@njit(cache=True)
def get_slope_array_numba(
    pos_arr: NDArray[np.float64],
    slopes: NDArray[np.float64],
    slope_intervals: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.empty(pos_arr.size, dtype=np.float64)
    for i in range(pos_arr.size):
        out[i] = get_slope_scalar_numba(pos_arr[i], slopes, slope_intervals)
    return out


@njit(cache=True)
def get_speed_limit_scalar_numba(
    pos: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
) -> float:
    idx = get_interval_index_scalar_numba(pos, speed_limit_intervals)
    if idx < 0:
        idx = 0
    elif idx >= speed_limits.size:
        idx = speed_limits.size - 1
    return speed_limits[idx]


@njit(cache=True)
def get_speed_limit_array_numba(
    pos_arr: NDArray[np.float64],
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.empty(pos_arr.size, dtype=np.float64)
    for i in range(pos_arr.size):
        out[i] = get_speed_limit_scalar_numba(
            pos_arr[i],
            speed_limits,
            speed_limit_intervals,
        )
    return out
