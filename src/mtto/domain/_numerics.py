from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ["get_interval_index_scalar_numba", "get_interval_index_array"]


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


def get_interval_index_array(
    pos: NDArray[np.floating],
    interval_points: NDArray[np.floating],
) -> NDArray[np.intp]:
    """返回包含给定位置的区间索引(numpy 数组版本)"""
    return np.searchsorted(interval_points, pos, side="right") - 1
