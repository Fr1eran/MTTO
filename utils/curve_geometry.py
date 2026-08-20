from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def find_2curves_crosspoint(
    curve1: NDArray[np.float64], curve2: NDArray[np.float64]
) -> tuple[float, float]:
    x1 = curve1[0, :]
    y1 = curve1[1, :]
    x2 = curve2[0, :]
    y2 = curve2[1, :]

    start_x = max(float(x1[0]), float(x2[0]))
    end_x = min(float(x1[-1]), float(x2[-1]))
    if start_x >= end_x:
        raise ValueError("Curves do not have an overlapping x-domain")

    x_common = np.linspace(start_x, end_x, num=1000, dtype=np.float64)
    y1_interp = np.interp(x_common, x1, y1)
    y2_interp = np.interp(x_common, x2, y2)
    diff = y1_interp - y2_interp

    idx = np.where(np.diff(np.sign(diff)))[0]
    if idx.size == 0:
        raise ValueError("Curves do not intersect in the overlapping region")

    x0_, x1_ = float(x_common[idx[0]]), float(x_common[idx[0] + 1])
    y0_, y1_ = float(diff[idx[0]]), float(diff[idx[0] + 1])
    denom = y1_ - y0_
    if abs(denom) < 1e-12:
        x_cross = 0.5 * (x0_ + x1_)
    else:
        x_cross = x0_ - y0_ * (x1_ - x0_) / denom
    y_cross = float(np.interp(x_cross, x1, y1))

    return float(x_cross), float(y_cross)


def cut_curve_by_crosspoint(
    s_list: NDArray[np.float64], v_list: NDArray[np.float64], x_cross: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    idx = int(np.searchsorted(s_list, x_cross, side="left"))
    return s_list[idx:], v_list[idx:]


def cal_regions(
    above_curves_list: list[NDArray[np.float64]],
    below_curves_list: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    if len(above_curves_list) != len(below_curves_list):
        raise ValueError("Two curve sets must have the same size")
    num_regions = len(above_curves_list)
    cross_points = np.empty((2, num_regions), dtype=np.float64)
    above_curves_part: list[NDArray[np.float64]] = []
    below_curves_part: list[NDArray[np.float64]] = []
    for i in range(num_regions):
        above_curve = above_curves_list[i]
        below_curve = below_curves_list[i]
        x_cross, y_cross = find_2curves_crosspoint(above_curve, below_curve)
        cross_points[0, i] = x_cross
        cross_points[1, i] = y_cross
        x_cut_a, y_cut_a = cut_curve_by_crosspoint(
            above_curve[0, :], above_curve[1, :], x_cross
        )
        x_cut_b, y_cut_b = cut_curve_by_crosspoint(
            below_curve[0, :], below_curve[1, :], x_cross
        )
        above_curves_part.append(np.stack([x_cut_a, y_cut_a], axis=0))
        below_curves_part.append(np.stack([x_cut_b, y_cut_b], axis=0))

    return cross_points, above_curves_part, below_curves_part


def pad_2curve_lists(
    curves_list1: list[NDArray[np.float64]],
    curves_list2: list[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    if len(curves_list1) != len(curves_list2):
        raise ValueError("Two curve lists must have the same size")
    curves_list1_padded: list[NDArray[np.float64]] = []
    curves_list2_padded: list[NDArray[np.float64]] = []
    for i in range(len(curves_list1)):
        curve1_x, curve1_y = curves_list1[i][0, :], curves_list1[i][1, :]
        curve2_x, curve2_y = curves_list2[i][0, :], curves_list2[i][1, :]
        c1_x, c1_y, c2_x, c2_y = pad_2curves(curve1_x, curve1_y, curve2_x, curve2_y)
        curves_list1_padded.append(np.stack([c1_x, c1_y], axis=0))
        curves_list2_padded.append(np.stack([c2_x, c2_y], axis=0))
    return curves_list1_padded, curves_list2_padded


def pad_2curves(
    curve1_x: NDArray[np.float64],
    curve1_y: NDArray[np.float64],
    curve2_x: NDArray[np.float64],
    curve2_y: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    pad_width = len(curve1_x) - len(curve2_x)
    if pad_width > 0:
        curve2_y = np.pad(curve2_y, (0, pad_width), mode="edge")
        curve2_x = np.pad(
            curve2_x,
            (0, pad_width),
            mode="linear_ramp",
            end_values=float(curve1_x[-1]),
        )
    elif pad_width < 0:
        curve1_y = np.pad(curve1_y, (0, -pad_width), mode="edge")
        curve1_x = np.pad(
            curve1_x,
            (0, -pad_width),
            mode="linear_ramp",
            end_values=float(curve2_x[-1]),
        )
    return curve1_x, curve1_y, curve2_x, curve2_y


def concatenate_curves_list(
    curves: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not curves:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    combined = np.concatenate(curves, axis=1)
    return combined[0], combined[1]
