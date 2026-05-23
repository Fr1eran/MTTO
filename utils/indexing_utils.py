from __future__ import annotations

from typing import NamedTuple, overload

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from utils.type_utils import ScalarNumeric, restore_output_type


@overload
def get_interval_index(
    pos: ScalarNumeric, interval_points: NDArray[np.floating]
) -> np.intp: ...


@overload
def get_interval_index(
    pos: ArrayLike, interval_points: NDArray[np.floating]
) -> NDArray[np.intp]: ...


def get_interval_index(
    pos: ScalarNumeric | ArrayLike,
    interval_points: NDArray[np.floating],
    side_right: bool = True,
) -> np.intp | NDArray[np.intp]:
    """返回包含给定位置的区间索引(numpy 向量化版本)"""
    pos = np.asarray(pos, dtype=np.float64)
    interval_points = np.asarray(interval_points, dtype=np.float64)
    idx = (
        np.searchsorted(interval_points, pos, side="right" if side_right else "left")
        - 1
    )
    return restore_output_type(idx)


@njit(cache=True)
def get_interval_index_scalar_numba(
    pos: float,
    interval_points: NDArray[np.float64],
    side_right: bool = True,
) -> int:
    """返回包含给定位置的区间索引(numba 标量化版本)"""
    left = 0
    right = interval_points.size
    while left < right:
        mid = (left + right) // 2
        if side_right:
            move_left = pos < interval_points[mid]
        else:
            move_left = pos <= interval_points[mid]
        if move_left:
            right = mid
        else:
            left = mid + 1
    return left - 1


class SpeedRiseEntry(NamedTuple):
    boundary_pos: float
    left_speed_scaled: float
    next_interval: int


class SpeedFallExit(NamedTuple):
    boundary_pos: float
    right_speed_scaled: float
    prev_interval: int


def find_speed_rise_entry_and_fall(
    speed_limits: ArrayLike,
    interval_points: ArrayLike,
    start_idx: int | None = None,
    end_idx: int | None = None,
    speed_factor: float = 1.0,
) -> tuple[list[SpeedRiseEntry], list[SpeedFallExit]]:
    """返回在[start_idx, end_idx)区间内的速度上升沿和速度下降沿。"""
    sl = np.asarray(speed_limits, dtype=np.float64)
    pts = np.asarray(interval_points, dtype=np.float64)

    if sl.ndim != 1 or pts.ndim != 1:
        raise ValueError("speed_limits and interval_points must be 1D arrays")
    if pts.size != sl.size + 1:
        raise ValueError("interval_points size must equal speed_limits size + 1")

    n = sl.size
    if n < 2:
        return [], []

    resolved_start = 0 if start_idx is None else int(start_idx)
    resolved_end = n - 1 if end_idx is None else int(end_idx)
    resolved_start = max(0, resolved_start)
    resolved_end = min(n - 1, resolved_end)

    if resolved_start >= resolved_end:
        return [], []

    diff = np.diff(sl)
    rise_indices = np.where(diff > 0)[0]
    fall_indices = np.where(diff < 0)[0]

    rise_entries = [
        SpeedRiseEntry(
            boundary_pos=float(pts[i + 1]),
            left_speed_scaled=float(sl[i] * speed_factor),
            next_interval=int(i + 1),
        )
        for i in rise_indices
        if resolved_start <= int(i) < resolved_end
    ]

    fall_exits = [
        SpeedFallExit(
            boundary_pos=float(pts[j + 1]),
            right_speed_scaled=float(sl[j + 1] * speed_factor),
            prev_interval=int(j),
        )
        for j in fall_indices
        if resolved_start <= int(j) < resolved_end
    ]

    return rise_entries, fall_exits
