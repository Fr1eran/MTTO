from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from utils.curve_geometry import concatenate_curves_list


def concatenate_curves_with_NaN(
    curves_set: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not curves_set:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    total_len = sum(curve.shape[1] + 1 for curve in curves_set)
    out = np.full((2, total_len), np.nan, dtype=np.float64)

    curr = 0
    for curve in curves_set:
        n = curve.shape[1]
        out[:, curr : curr + n] = curve
        curr += n + 1

    return out[0], out[1]


def draw_regions(
    ax: Axes,
    above_curves_list: list[NDArray[np.float64]],
    below_curves_list: list[NDArray[np.float64]],
    label: str,
    color: str,
    alpha: float,
) -> None:
    if not above_curves_list or not below_curves_list:
        return

    above_curves_x_con, above_curves_y_con = concatenate_curves_list(above_curves_list)
    _below_curves_x_con, below_curves_y_con = concatenate_curves_list(below_curves_list)

    above_curves_y_kmh = above_curves_y_con * 3.6
    below_curves_y_kmh = below_curves_y_con * 3.6

    _ = ax.fill_between(
        above_curves_x_con,
        above_curves_y_kmh,
        below_curves_y_kmh,
        where=(above_curves_y_kmh > below_curves_y_kmh),
        interpolate=False,
        step="pre",
        label=label,
        color=color,
        alpha=alpha,
    )
