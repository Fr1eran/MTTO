from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, DTypeLike, NDArray

from utils.indexing_utils import get_interval_index, get_interval_index_scalar_numba
from utils.type_utils import ScalarNumeric, restore_output_type


@dataclass
class TrackInfo:
    slopes: NDArray[np.float64]  # 单位: %
    slope_intervals: NDArray[np.float64]  # 单位: m
    speed_limits: NDArray[np.float64]  # 单位: m/s
    speed_limit_intervals: NDArray[np.float64]  # 单位: m
    ASA_aps: list[float] = field(default_factory=list)  # 辅助停车区可达点
    ASA_dps: list[float] = field(default_factory=list)  # 辅助停车区危险点

    def __post_init__(self):
        self.slopes = np.asarray(self.slopes, dtype=np.float64)
        self.slope_intervals = np.asarray(self.slope_intervals, dtype=np.float64)
        self.speed_limits = np.asarray(self.speed_limits, dtype=np.float64)
        self.speed_limit_intervals = np.asarray(
            self.speed_limit_intervals, dtype=np.float64
        )


# 标量化 numba 版本


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
            pos_arr[i], speed_limits, speed_limit_intervals
        )
    return out


# 向量化 numpy 版本


@overload
def get_slope(
    pos: ScalarNumeric,
    slopes: ArrayLike,
    slope_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> np.floating: ...


@overload
def get_slope(
    pos: ArrayLike,
    slopes: ArrayLike,
    slope_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> NDArray[np.floating]: ...


def get_slope(
    pos: ScalarNumeric | ArrayLike,
    slopes: ArrayLike,
    slope_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> np.floating | NDArray[np.floating]:
    """
    根据当前位置计算相应的分段常值坡度。
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    slopes_arr = np.asarray(slopes, dtype=np.float64)
    intervals_arr = np.asarray(slope_intervals, dtype=np.float64)
    slope = slopes_arr[
        np.clip(get_interval_index(pos_arr, intervals_arr), 0, len(slopes_arr) - 1)
    ]
    return restore_output_type(slope.astype(dtype=dtype))


@overload
def get_next_slope_and_distance(
    pos: ScalarNumeric,
    direction: int,
    slopes: ArrayLike,
    slope_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> tuple[np.floating, np.floating]: ...


@overload
def get_next_slope_and_distance(
    pos: ArrayLike,
    direction: int,
    slopes: ArrayLike,
    slope_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...


def get_next_slope_and_distance(
    pos: ScalarNumeric | ArrayLike,
    direction: int,
    slopes: ArrayLike,
    slope_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> (
    tuple[np.floating, np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]
):
    """
    根据当前位置和运动方向返回下个坡度区间坡度和当前位置与下一坡度区间的距离。
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    slopes_arr = np.asarray(slopes, dtype=np.float64)
    intervals_arr = np.asarray(slope_intervals, dtype=np.float64)
    current_interval_index = get_interval_index(pos_arr, intervals_arr)
    next_interval_index = np.clip(
        current_interval_index + direction, 0, len(slopes_arr) - 1
    )

    next_slope = slopes_arr[next_interval_index]
    distance = (
        intervals_arr[
            np.clip(
                current_interval_index + (direction + 1) // 2,
                0,
                len(intervals_arr) - 1,
            )
        ]
        - pos_arr
    )

    return restore_output_type(next_slope.astype(dtype=dtype)), restore_output_type(
        distance.astype(dtype=dtype)
    )


@overload
def get_speed_limit(
    pos: ScalarNumeric,
    speed_limits: ArrayLike,
    speed_limit_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> np.floating: ...


@overload
def get_speed_limit(
    pos: ArrayLike,
    speed_limits: ArrayLike,
    speed_limit_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> NDArray[np.floating]: ...


def get_speed_limit(
    pos: ScalarNumeric | ArrayLike,
    speed_limits: ArrayLike,
    speed_limit_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> np.floating | NDArray[np.floating]:
    """
    根据当前位置返回对应的限速值。
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    limits_arr = np.asarray(speed_limits, dtype=np.float64)
    intervals_arr = np.asarray(speed_limit_intervals, dtype=np.float64)
    speed_limit = limits_arr[
        np.clip(get_interval_index(pos_arr, intervals_arr), 0, len(limits_arr) - 1)
    ]
    return restore_output_type(speed_limit.astype(dtype=dtype))


@overload
def get_next_speed_limit_and_distance(
    pos: ScalarNumeric,
    direction: int,
    speed_limits: ArrayLike,
    speed_limit_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> tuple[np.floating, np.floating]: ...


@overload
def get_next_speed_limit_and_distance(
    pos: ArrayLike,
    direction: int,
    speed_limits: ArrayLike,
    speed_limit_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...


def get_next_speed_limit_and_distance(
    pos: ScalarNumeric | ArrayLike,
    direction: int,
    speed_limits: ArrayLike,
    speed_limit_intervals: ArrayLike,
    *,
    dtype: DTypeLike = np.float32,
) -> (
    tuple[np.floating, np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]
):
    """
    根据当前位置和运动方向返回下个限速区间限速和当前位置与下一限速区间的距离。
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    limits_arr = np.asarray(speed_limits, dtype=np.float64)
    intervals_arr = np.asarray(speed_limit_intervals, dtype=np.float64)
    current_interval_index = get_interval_index(pos_arr, intervals_arr)
    next_interval_index = np.clip(
        current_interval_index + direction, 0, len(limits_arr) - 1
    )

    next_speed_limit = limits_arr[next_interval_index]
    distance = (
        intervals_arr[
            np.clip(
                current_interval_index + (direction + 1) // 2,
                0,
                len(intervals_arr) - 1,
            )
        ]
        - pos_arr
    )

    return restore_output_type(
        next_speed_limit.astype(dtype=dtype)
    ), restore_output_type(distance.astype(dtype=dtype))
