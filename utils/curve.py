import numpy as np
from matplotlib.axes import Axes


def Find2CurvesCrossPoint(curve1, curve2):
    x1 = curve1[0, :]
    y1 = curve1[1, :]
    x2 = curve2[0, :]
    y2 = curve2[1, :]
    x_common = np.linspace(max(x1[0], x2[0]), min(x1[-1], x2[-1]), num=1000)
    y1_interp = np.interp(x_common, x1, y1)
    y2_interp = np.interp(x_common, x2, y2)
    diff = y1_interp - y2_interp

    idx = np.where(np.diff(np.sign(diff)))[0]  # 搜索符号变化点的索引
    x0_, x1_ = x_common[idx], x_common[idx + 1]
    y0_, y1_ = diff[idx], diff[idx + 1]
    x_cross = x0_ - y0_ * (x1_ - x0_) / (y1_ - y0_)
    y_cross = np.interp(x_cross, x1, y1)

    return (x_cross[0], y_cross[0])


def CutCurveByCrossPoint(s_list, v_list, x_cross):
    idx = np.searchsorted(s_list, x_cross, side="left")
    return s_list[idx:], v_list[idx:]


def CalRegions(above_curves_list, below_curves_list):
    if len(above_curves_list) != len(below_curves_list):
        raise Exception("两个曲线集合的曲线个数不相等")
    cross_points = np.empty((2, len(above_curves_list)))
    above_curves_part = []
    below_curves_part = []
    for i in range(len(above_curves_list)):
        x_arr_a, y_arr_a = above_curves_list[i][0, :], above_curves_list[i][1, :]
        x_arr_b, y_arr_b = below_curves_list[i][0, :], below_curves_list[i][1, :]
        x_cross, y_cross = Find2CurvesCrossPoint(
            above_curves_list[i], below_curves_list[i]
        )
        cross_points[0, i] = x_cross
        cross_points[1, i] = y_cross
        x_cur_a, y_cut_a = CutCurveByCrossPoint(x_arr_a, y_arr_a, x_cross)
        x_cut_b, y_cut_b = CutCurveByCrossPoint(x_arr_b, y_arr_b, x_cross)
        above_curves_part.append(np.stack([x_cur_a, y_cut_a], axis=0))
        below_curves_part.append(np.stack([x_cut_b, y_cut_b], axis=0))

    return cross_points, above_curves_part, below_curves_part


def Padding2CurvesList(curves_list1, curves_list2):
    """
    将曲线对的数据点个数对齐

    Args:
        curves_set1: 第一个曲线列表
        curves_set2: 第二个曲线列表

    Returns:
        返回数据点个数对齐后的两个曲线列表

    Raises:
        Exception: 如果两个曲线列表的大小不一致
    """
    if len(curves_list1) != len(curves_list2):
        raise Exception("两个曲线列表的大小不一致")
    curves_list1_padded = []
    curves_list2_padded = []
    for i in range(len(curves_list1)):
        curve1_x, curve1_y = curves_list1[i][0, :], curves_list1[i][1, :]
        curve2_x, curve2_y = curves_list2[i][0, :], curves_list2[i][1, :]
        curve1_x, curve1_y, curve2_x, curve2_y = Padding2Curves(
            curve1_x, curve1_y, curve2_x, curve2_y
        )
        curves_list1_padded.append(np.stack([curve1_x, curve1_y], axis=0))
        curves_list2_padded.append(np.stack([curve2_x, curve2_y], axis=0))
    return curves_list1_padded, curves_list2_padded


def Padding2Curves(curve1_x, curve1_y, curve2_x, curve2_y):
    pad_width = len(curve1_x) - len(curve2_x)
    if pad_width > 0:
        curve2_y = np.pad(curve2_y, (0, pad_width), mode="edge")
        curve2_x = np.pad(
            curve2_x, (0, pad_width), mode="linear_ramp", end_values=curve1_x[-1]
        )
    elif pad_width < 0:
        curve1_y = np.pad(curve1_y, (0, -pad_width), mode="edge")
        curve1_x = np.pad(
            curve1_x, (0, -pad_width), mode="linear_ramp", end_values=curve2_x[-1]
        )
    else:
        pass
    return curve1_x, curve1_y, curve2_x, curve2_y


def ConcatenateCurvesWithNaN(curves_set: list):
    """
    将一个曲线集合拆分并拼接为两个坐标值数组，
    并在曲线段之间插入np.nan

    Args:
        curves: 曲线列表
    Returns:
        返回沿两个坐标轴拼接后的坐标值np.ndarray数组
    """
    curves_x = [data[0, :] for data in curves_set]
    curves_y = [data[1, :] for data in curves_set]

    curves_x_con_with_nan = np.concatenate(
        [np.concatenate([seg, [np.nan]]) for seg in curves_x]
    )
    curves_y_con_with_nan = np.concatenate(
        [np.concatenate([seg, [np.nan]]) for seg in curves_y]
    )

    return curves_x_con_with_nan, curves_y_con_with_nan


def DrawRegions(ax: Axes, above_curves_list, below_curves_list, label, color, alpha):
    """
    绘制由最小速度曲线、最大速度曲线和横坐标轴围成的图形

    Args:
        ax: 绘图轴
        above_curves_list: 总位于上方的曲线列表
        below_curves_list: 总位于下方的曲线列表
        color: 颜色
        alpha: 透明度
    """

    above_curves_x_con, above_curves_y_con = ConcatenateCurves(above_curves_list)
    below_curves_x_con, below_curves_y_con = ConcatenateCurves(below_curves_list)

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


def ConcatenateCurves(curves: list):
    curves_x = [data[0, :] for data in curves]
    curves_y = [data[1, :] for data in curves]

    curves_x_con = np.concatenate(curves_x, dtype=np.float64)
    curves_y_con = np.concatenate(curves_y, dtype=np.float64)

    return curves_x_con, curves_y_con
