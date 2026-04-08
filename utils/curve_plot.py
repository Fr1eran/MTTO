import numpy as np
from matplotlib.axes import Axes

from utils.curve_geometry import concatenate_curves_list


def concatenate_curves_with_NaN(curves_set: list):
    curves_x = [data[0, :] for data in curves_set]
    curves_y = [data[1, :] for data in curves_set]

    curves_x_con_with_nan = np.concatenate(
        [np.concatenate([seg, [np.nan]]) for seg in curves_x]
    )
    curves_y_con_with_nan = np.concatenate(
        [np.concatenate([seg, [np.nan]]) for seg in curves_y]
    )

    return curves_x_con_with_nan, curves_y_con_with_nan


def draw_regions(ax: Axes, above_curves_list, below_curves_list, label, color, alpha):
    above_curves_x_con, above_curves_y_con = concatenate_curves_list(above_curves_list)
    below_curves_x_con, below_curves_y_con = concatenate_curves_list(below_curves_list)

    above_curves_y_con *= 3.6
    below_curves_y_con *= 3.6

    ax.fill_between(
        above_curves_x_con,
        above_curves_y_con,
        below_curves_y_con,
        where=(above_curves_y_con > below_curves_y_con).tolist(),
        interpolate=False,
        step="pre",
        label=label,
        color=color,
        alpha=alpha,
    )
