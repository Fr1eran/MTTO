from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mtto.domain._numerics import get_interval_index_array
from mtto.domain.safeguard.geometry import cal_regions, pad_2curve_lists

if TYPE_CHECKING:
    from mtto.domain.safeguard.curves import Safeguard

__all__ = [
    "StaticRegion",
    "build_static_region",
    "intersecting_danger_points",
    "detect_danger",
    "detect_any_danger",
]


@dataclass(frozen=True, slots=True, eq=False)
class StaticRegion:
    idp_points_x: NDArray[np.float64]
    min_curves_part_x_padded: tuple[NDArray[np.float64], ...]
    min_curves_part_y_padded: tuple[NDArray[np.float64], ...]
    max_curves_part_x_padded: tuple[NDArray[np.float64], ...]
    max_curves_part_y_padded: tuple[NDArray[np.float64], ...]
    num_regions: int


def build_static_region(
    min_curves: Sequence[NDArray[np.float64]],
    max_curves: Sequence[NDArray[np.float64]],
) -> StaticRegion:
    below = (
        list(max_curves)[:-1]
        if len(max_curves) == len(min_curves) + 1
        else list(max_curves)
    )
    idp_points, min_parts, max_parts = cal_regions(list(min_curves), below)
    idp_points_x = np.asarray(idp_points[0, :], dtype=np.float64)
    min_parts_arr = [np.asarray(c, dtype=np.float64) for c in min_parts]
    max_parts_arr = [np.asarray(c, dtype=np.float64) for c in max_parts]

    min_padded, max_padded = pad_2curve_lists(min_parts_arr, max_parts_arr)
    min_x = [np.asarray(c[0, :], dtype=np.float64) for c in min_padded]
    min_y = [np.asarray(c[1, :], dtype=np.float64) for c in min_padded]
    max_x = [np.asarray(c[0, :], dtype=np.float64) for c in max_padded]
    max_y = [np.asarray(c[1, :], dtype=np.float64) for c in max_padded]

    idp_points_x.flags.writeable = False
    for arr in min_x:
        arr.flags.writeable = False
    for arr in min_y:
        arr.flags.writeable = False
    for arr in max_x:
        arr.flags.writeable = False
    for arr in max_y:
        arr.flags.writeable = False

    return StaticRegion(
        idp_points_x=idp_points_x,
        min_curves_part_x_padded=tuple(min_x),
        min_curves_part_y_padded=tuple(min_y),
        max_curves_part_x_padded=tuple(max_x),
        max_curves_part_y_padded=tuple(max_y),
        num_regions=int(idp_points.shape[1]),
    )


def intersecting_danger_points(safeguard: Safeguard) -> NDArray[np.float64]:
    return safeguard.static_region.idp_points_x


def detect_danger(
    safeguard: Safeguard,
    pos: float | ArrayLike,
    speed: float | ArrayLike,
) -> bool | NDArray[np.bool_]:
    """检查速度是否超出限速或落入危险速度域"""
    pos_arr, speed_arr = np.broadcast_arrays(
        np.asarray(pos, dtype=np.float64),
        np.asarray(speed, dtype=np.float64),
    )
    result1 = _detect_speed_exceed(safeguard, pos_arr, speed_arr)
    result2 = _detect_dangerous_region_enter(safeguard, pos_arr, speed_arr)
    result = result1 | result2
    if result.ndim == 0:
        return bool(result)
    return result


def detect_any_danger(
    safeguard: Safeguard,
    pos: float | ArrayLike,
    speed: float | ArrayLike,
) -> bool:
    """检查输入序列中是否存在任一危险状态(早停语义)。"""
    pos_arr, speed_arr = np.broadcast_arrays(
        np.asarray(pos, dtype=np.float64),
        np.asarray(speed, dtype=np.float64),
    )
    if _detect_speed_exceed_any(safeguard, pos_arr, speed_arr):
        return True
    return _detect_dangerous_region_enter_any(safeguard, pos_arr, speed_arr)


def _detect_speed_exceed(
    safeguard: Safeguard,
    pos: NDArray[np.floating],
    speed: NDArray[np.floating],
) -> NDArray[np.bool_]:
    speed_limit = safeguard.speed_limits[
        np.clip(
            get_interval_index_array(pos, safeguard.speed_limit_intervals),
            0,
            len(safeguard.speed_limits) - 1,
        )
    ]
    return speed >= speed_limit * safeguard.params.factor


def _detect_speed_exceed_any(
    safeguard: Safeguard,
    pos: NDArray[np.floating],
    speed: NDArray[np.floating],
) -> bool:
    if pos.ndim == 0:
        idx = int(get_interval_index_array(pos, safeguard.speed_limit_intervals))
        idx = min(max(idx, 0), len(safeguard.speed_limits) - 1)
        threshold = safeguard.speed_limits[idx] * safeguard.params.factor
        return bool(float(speed) >= threshold)

    speed_limit = safeguard.speed_limits[
        np.clip(
            np.searchsorted(safeguard.speed_limit_intervals, pos, side="right") - 1,
            0,
            len(safeguard.speed_limits) - 1,
        )
    ]
    return bool(np.any(speed >= speed_limit * safeguard.params.factor))


def _detect_dangerous_region_enter(
    safeguard: Safeguard,
    pos: NDArray[np.floating],
    speed: NDArray[np.floating],
) -> bool | NDArray[np.bool_]:
    region = safeguard.static_region
    if pos.ndim == 0:
        pos_value = float(pos)
        speed_value = float(speed)
        for i in range(region.num_regions):
            if not (
                pos_value > region.idp_points_x[i]
                and pos_value < region.min_curves_part_x_padded[i][-1]
            ):
                continue
            above_v = float(
                np.interp(
                    pos_value,
                    region.min_curves_part_x_padded[i],
                    region.min_curves_part_y_padded[i],
                )
            )
            below_v = float(
                np.interp(
                    pos_value,
                    region.max_curves_part_x_padded[i],
                    region.max_curves_part_y_padded[i],
                )
            )
            return bool(speed_value <= above_v and speed_value >= below_v)
        return False

    result = np.zeros_like(pos, dtype=bool)
    for i in range(region.num_regions):
        mask = (pos > region.idp_points_x[i]) & (
            pos < region.min_curves_part_x_padded[i][-1]
        )
        if not np.any(mask):
            continue

        pos_masked = pos[mask]
        speed_masked = speed[mask]
        above_v = np.interp(
            pos_masked,
            region.min_curves_part_x_padded[i],
            region.min_curves_part_y_padded[i],
        )
        below_v = np.interp(
            pos_masked,
            region.max_curves_part_x_padded[i],
            region.max_curves_part_y_padded[i],
        )
        result[mask] |= (speed_masked <= above_v) & (speed_masked >= below_v)
    return result


def _detect_dangerous_region_enter_any(
    safeguard: Safeguard,
    pos: NDArray[np.floating],
    speed: NDArray[np.floating],
) -> bool:
    region = safeguard.static_region
    if pos.ndim == 0:
        return bool(_detect_dangerous_region_enter(safeguard, pos, speed))

    for i in range(region.num_regions):
        mask = (pos > region.idp_points_x[i]) & (
            pos < region.min_curves_part_x_padded[i][-1]
        )
        if not np.any(mask):
            continue

        pos_masked = pos[mask]
        speed_masked = speed[mask]
        above_v = np.interp(
            pos_masked,
            region.min_curves_part_x_padded[i],
            region.min_curves_part_y_padded[i],
        )
        below_v = np.interp(
            pos_masked,
            region.max_curves_part_x_padded[i],
            region.max_curves_part_y_padded[i],
        )
        if np.any((speed_masked <= above_v) & (speed_masked >= below_v)):
            return True
    return False
