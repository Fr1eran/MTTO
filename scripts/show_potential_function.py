import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from utils.data_loader import load_safeguard_curves
from utils.plot_utils import set_global_plot_style


def _potential_safety_speed(pos, speed, min_speed, max_speed, target_pos):
    """
    向量化安全势能函数

    速度带中间位置势能最大
    """
    distance_to_target = abs(target_pos - pos)
    center_speed = (max_speed + min_speed) / 2.0
    safe_margin = np.maximum((max_speed - min_speed) / 2.0, 0.5)

    # 基础偏离惩罚(四次方项, 引导列车走中间)
    norm_speed_diff = (speed - center_speed) / safe_margin
    speed_log_arg = 1.01 - norm_speed_diff**2
    in_speed_band = (speed >= min_speed) & (speed <= max_speed)
    valid_mask_speed = in_speed_band & (speed_log_arg > 0.0)
    phi_base = np.full_like(norm_speed_diff, np.nan, dtype=np.float64)
    phi_base[valid_mask_speed] = 2.0 * np.log(speed_log_arg[valid_mask_speed])

    # 靠近目标位置时，适当增大惩罚力度
    scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

    return scale * phi_base


def _potential_safety_position(pos, min_pos, max_pos, target_pos):
    distance_to_target = np.abs(target_pos - pos)
    center_pos = (max_pos + min_pos) / 2.0
    safe_margin = (max_pos - min_pos) / 2.0

    norm_pos_diff = (pos - center_pos) / safe_margin
    spatial_log_arg = 1.1 - norm_pos_diff**2
    in_pos_band = (pos >= min_pos) & (pos <= max_pos)
    valid_mask_spatial = in_pos_band & (spatial_log_arg > 0.0)
    phi_base = np.full_like(norm_pos_diff, np.nan, dtype=np.float64)
    phi_base[valid_mask_spatial] = 2.0 * np.log(spatial_log_arg[valid_mask_spatial])

    scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

    return scale * phi_base


def _potential_safety_speed_adaptive(pos, speed, min_speed, max_speed, target_pos):
    distance_to_target = np.abs(target_pos - pos)
    scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

    v_star = np.where(min_speed > 0.0, (max_speed + min_speed) / 2.0, 0.0)
    L = np.where(
        min_speed > 0,
        np.clip((max_speed - min_speed) / 2.0, 1.0, None),
        np.clip(max_speed, 1.0, None),
    )

    norm_speed_diff = (speed - v_star) / L
    speed_log_arg = 1.01 - norm_speed_diff**2

    in_speed_band = (speed >= min_speed) & (speed <= max_speed)
    valid_mask_speed = in_speed_band & (speed_log_arg > 0.0)

    phi_base = np.full_like(norm_speed_diff, np.nan, dtype=np.float64)
    phi_base[valid_mask_speed] = 2.0 * np.log(speed_log_arg[valid_mask_speed])

    return scale * phi_base


def _potential_safety_speed_asymmetric_v1(
    pos,
    speed,
    min_speed,
    max_speed,
    target_pos,
) -> float:
    distance_to_target = np.abs(target_pos - pos)

    # 设定一个危险缓冲距离 (m/s)，仅当距离边界小于该值时才触发惩罚
    upper_bound = 8.0
    lower_bound = 5.0

    # 1. 上限惩罚 (始终激活)
    margin_max = max_speed - speed
    phi_max = np.where(
        margin_max < upper_bound,
        2.0 * np.log(1.01 - (1.0 - np.maximum(margin_max, 0.0) / upper_bound) ** 2),
        0.0,
    )

    # 2. 下限惩罚 (条件激活：仅当存在实质性的最小速度约束时才惩罚)
    margin_min = speed - min_speed
    # 当 min_speed 极小 (例如接近 0) 时，说明当前允许停车，直接关闭下限惩罚
    activate_min = (min_speed > 0.0) & (margin_min < lower_bound)
    phi_min = np.where(
        activate_min,
        2.0 * np.log(1.01 - (1.0 - np.maximum(margin_min, 0.0) / lower_bound) ** 2),
        0.0,
    )

    # 距离缩放系数
    scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

    # 最终势能为两侧惩罚之和
    return scale * (phi_max + phi_min)


def _potential_safety_speed_asymmetric_v2(
    pos,
    speed,
    min_speed,
    max_speed,
    target_pos,
) -> float:
    distance_to_target = np.abs(target_pos - pos)

    # 设定一个危险缓冲距离 (m/s)，仅当距离边界小于该值时才触发惩罚
    upper_bound = 8.0
    lower_bound = 5.0

    # 1. 上限惩罚 (始终激活)
    margin_max = max_speed - speed
    norm_margin_max = np.maximum(1.0 - margin_max / upper_bound, 0.0)
    phi_max = -(norm_margin_max**2)

    # 2. 下限惩罚 (条件激活：仅当存在实质性的最小速度约束时才惩罚)
    margin_min = speed - min_speed
    norm_margin_min = np.maximum(1.0 - margin_min / lower_bound, 0.0)
    # 当 min_speed 极小 (例如接近 0) 时，说明当前允许停车，直接关闭下限惩罚
    phi_min = np.where(
        min_speed > 0.0,
        -(norm_margin_min**2),
        0.0,
    )

    # 距离缩放系数
    scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

    # 最终势能为两侧惩罚之和
    return scale * (phi_max + phi_min) * 4.0


def _potential_stopping_v1(
    pos,
    speed,
    target_pos,
):
    """
    向量化停站势函数
    """
    # 基础参数
    d_scale = 3000.0
    speed_max = 500.0 / 3.6

    sigma_x_hat_weak = 1.0
    sigma_v_hat_weak = 1.0

    sigma_x_hat_strong = 0.1
    sigma_v_hat_strong = 0.2

    # 正则化项
    dist_error = np.abs(target_pos - pos)
    x_hat = dist_error / d_scale
    v_hat = np.abs(speed / speed_max)

    # 增益参数
    K_W = 2.0
    K_S = 30.0

    phi_weak = K_W * np.exp(-x_hat / sigma_x_hat_weak - v_hat / sigma_v_hat_weak)
    phi_strong = K_S * np.exp(-x_hat / sigma_x_hat_strong - v_hat / sigma_v_hat_strong)

    return phi_weak + phi_strong


def _potential_stopping_v2(
    pos,
    speed,
    target_pos,
):
    """
    向量化停站势函数
    """
    # 基础参数
    d_scale = 3000.0
    speed_max = 500.0 / 3.6

    sigma_x_hat = 0.3
    sigma_v_hat = 0.2

    # 正则化项
    dist_error = np.abs(target_pos - pos)
    x_hat = dist_error / d_scale
    v_hat = np.abs(speed / speed_max)

    # 增益参数
    K_G = 20.0

    phi_strong = K_G * np.exp(-x_hat / sigma_x_hat) * np.exp(-v_hat / sigma_v_hat)

    return phi_strong


def _potential_punctuality_v1(
    redundant_operation_time,
    schedule_time: float,
):
    time_redundancy_norm = redundant_operation_time / schedule_time

    return -4.0 * np.log1p(np.exp(-3.0 * time_redundancy_norm))


def _potential_punctuality_v2(redundant_operation_time, schedule_time):
    K_base = 0.5
    K_safe = 0.1
    K_late = 1.0

    time_redundancy_norm = redundant_operation_time / schedule_time

    return np.where(
        time_redundancy_norm >= 0,
        K_base + K_safe * time_redundancy_norm,
        K_base + K_safe * time_redundancy_norm - K_late * time_redundancy_norm**2,
    )


def _potential_punctuality_v3(redundant_operation_time, schedule_time):
    K_base = 1.0
    K_safe = 1.0
    K_late = 10.0
    alpha = 5.0

    time_redundancy_norm = redundant_operation_time / schedule_time

    return np.where(
        time_redundancy_norm >= 0,
        K_base + K_safe * time_redundancy_norm,
        K_base
        + K_safe * time_redundancy_norm
        - K_late
        / alpha
        * (np.exp(-alpha * time_redundancy_norm) + alpha * time_redundancy_norm - 1),
    )


def infer_position_from_speed(curve_pos, curve_speed, target_speed):
    """
    在最小速度曲线随位置单调递减时，根据目标速度反推对应位置。
    """

    # np.interp 要求自变量单调递增，因此将递减速度轴反转后执行反向插值。
    return np.interp(
        target_speed,
        curve_speed[::-1],
        curve_pos[::-1],
        left=float(curve_pos[-1]),
        right=float(curve_pos[0]),
    )


def interp_with_constant_fill(x, y, query, left_value, right_value):
    """
    使用 numpy.interp 做线性插值，并在区间外用常量填充。
    """

    return np.interp(
        query,
        x,
        y,
        left=left_value,
        right=right_value,
    )


def _apply_minimal_axis_style(ax) -> None:
    ax.grid(False)
    ax.set_axis_on()


def _apply_transparent_background(fig: Figure) -> None:
    """让图窗与所有坐标轴背景透明，便于嵌入论文架构图。"""
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for ax in fig.axes:
        ax.set_facecolor("none")
        if ax.patch is not None:
            ax.patch.set_alpha(0.0)

        # 3D 坐标轴 pane 默认非透明，需单独处理。
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis_obj = getattr(ax, axis_name, None)
            pane = getattr(axis_obj, "pane", None)
            if pane is not None:
                pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def plot_safety_potential_heatmap_speed(*, minimal: bool = False) -> Figure:
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )

    # 以第7个辅助停车区作为示例
    target_pos = 17828.0

    upper_speed = 200.0 / 3.6
    lower_speed = 0.0

    min_curve_pos = min_curves_list[6][0, :]
    min_curve_speed = min_curves_list[6][1, :]
    max_curve_pos = max_curves_list[7][0, :]
    max_curve_speed = max_curves_list[7][1, :]

    pos_left_bound = infer_position_from_speed(
        min_curve_pos, min_curve_speed, upper_speed
    )
    pos_right_bound = max_curves_list[7][0, -1]

    pos_array = np.linspace(pos_left_bound, pos_right_bound, 2000)
    speed_array_ms = np.linspace(lower_speed, upper_speed, 2000)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)

    speed_min_1d = interp_with_constant_fill(
        min_curve_pos,
        min_curve_speed,
        pos_array,
        left_value=lower_speed,
        right_value=lower_speed,
    )
    speed_max_1d = interp_with_constant_fill(
        max_curve_pos,
        max_curve_speed,
        pos_array,
        left_value=upper_speed,
        right_value=upper_speed,
    )

    speed_min_1d_masked = np.where(speed_min_1d >= lower_speed, speed_min_1d, np.nan)
    speed_max_1d_masked = np.where(speed_max_1d < upper_speed, speed_max_1d, np.nan)

    speed_min_1d = np.maximum(speed_min_1d, lower_speed)
    speed_max_1d = np.minimum(speed_max_1d, upper_speed)

    SPEED_MIN = np.tile(speed_min_1d, (speed_array_ms.size, 1))
    SPEED_MAX = np.tile(speed_max_1d, (speed_array_ms.size, 1))

    # 计算整个网络的势能值

    # 原始版本
    # POTENTIAL = calc_potential_safety_speed(
    #     POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    # )

    # 自适应安全目标势能场
    # POTENTIAL = calc_potential_safety_speed_adaptive(
    #     POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    # )

    # 非对称解耦势能场 v1
    # POTENTIAL = calc_potential_safety_speed_asymmetric_v1(
    #     POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    # )

    # 非对称解耦势能场 v2
    POTENTIAL = _potential_safety_speed_asymmetric_v2(
        POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    )

    # 生成 masking, 越界区域的值设为NAN, 使其在图上透明
    in_speed_band = (SPEED >= SPEED_MIN) & (SPEED <= SPEED_MAX)
    POTENTIAL_MASKED = np.where(in_speed_band, POTENTIAL, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.get_cmap("Spectral")

    c = ax.pcolormesh(
        POS,
        SPEED * 3.6,
        POTENTIAL_MASKED,
        cmap=cmap,
        vmin=-4.0,
        vmax=0.0,
    )

    ax.set_xlim(pos_left_bound + 5000, pos_right_bound + 1000)
    ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6 * 0.75)

    if minimal:
        ax.plot(
            pos_array,
            speed_max_1d_masked * 3.6,
            color="red",
            linewidth=1,
        )
        ax.plot(
            pos_array,
            speed_min_1d_masked * 3.6,
            color="blue",
            linewidth=1,
        )
        _apply_minimal_axis_style(ax)
    else:
        ax.plot(
            pos_array,
            speed_max_1d_masked * 3.6,
            color="red",
            linewidth=1,
            label=r"upper speed curve",
        )
        ax.plot(
            pos_array,
            speed_min_1d_masked * 3.6,
            color="blue",
            linewidth=1,
            label=r"lower speed curve",
        )

        # 添加色标
        fig.colorbar(c, ax=ax, extend="min")

        # 图表格式化
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.tick_params(axis="both", which="major")
        ax.legend(loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


def plot_safety_potential_heatmap_position(*, minimal: bool = False) -> Figure:
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )

    # 仍然以第7个辅助停车区为示例。
    target_pos = 17828.0

    upper_speed = 150.0 / 3.6
    lower_speed = 0.0

    min_curve_pos = min_curves_list[6][0, :]
    min_curve_speed = min_curves_list[6][1, :]
    max_curve_pos = max_curves_list[7][0, :]
    max_curve_speed = max_curves_list[7][1, :]

    # 这里将速度视为自变量，按速度从 0 到 200 km/h 反算位置边界。
    speed_array_ms = np.linspace(lower_speed, upper_speed, 2000)
    pos_from_min_curve = infer_position_from_speed(
        min_curve_pos, min_curve_speed, speed_array_ms
    )
    pos_from_max_curve = infer_position_from_speed(
        max_curve_pos, max_curve_speed, speed_array_ms
    )

    safe_center_pos_array = (pos_from_min_curve + pos_from_max_curve) / 2.0

    pos_lower_1d = np.minimum(pos_from_min_curve, pos_from_max_curve)
    pos_upper_1d = np.maximum(pos_from_min_curve, pos_from_max_curve)

    pos_left_bound = float(np.min(pos_lower_1d))
    pos_right_bound = float(np.max(pos_upper_1d))

    pos_array = np.linspace(pos_left_bound, pos_right_bound, 2000)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)
    POS_LOWER = np.tile(pos_lower_1d[:, None], (1, pos_array.size))
    POS_UPPER = np.tile(pos_upper_1d[:, None], (1, pos_array.size))

    POTENTIAL = _potential_safety_position(POS, POS_LOWER, POS_UPPER, target_pos)
    in_pos_band = (POS >= POS_LOWER) & (POS <= POS_UPPER)
    POTENTIAL_MASKED = np.where(in_pos_band, POTENTIAL, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.get_cmap("Spectral")

    c = ax.pcolormesh(
        POS,
        SPEED * 3.6,
        POTENTIAL_MASKED,
        cmap=cmap,
        shading="auto",
        vmin=-8.0,
        vmax=0.0,
    )

    ax.set_xlim(pos_left_bound - 1000, pos_right_bound + 1000)
    ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6)

    if minimal:
        ax.plot(
            pos_from_max_curve,
            speed_array_ms * 3.6,
            color="red",
            linewidth=1,
        )
        ax.plot(
            pos_from_min_curve,
            speed_array_ms * 3.6,
            color="blue",
            linewidth=1,
        )
        _apply_minimal_axis_style(ax)
    else:
        ax.plot(
            pos_from_max_curve,
            speed_array_ms * 3.6,
            color="red",
            linewidth=1,
            label="upper speed curve",
        )
        ax.plot(
            pos_from_min_curve,
            speed_array_ms * 3.6,
            color="blue",
            linewidth=1,
            label="lower speed curve",
        )
        ax.plot(
            safe_center_pos_array,
            speed_array_ms * 3.6,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="safe center",
        )

        fig.colorbar(c, ax=ax, extend="min")

        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.legend(loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


def plot_stopping_potential_heatmap(
    view_mode: str = "3d",
    *,
    minimal: bool = False,
) -> Figure:
    """
    按停站势函数公式绘制势场图。

    Args:
        view_mode: "2d" 绘制热力图, "3d" 绘制三维曲面图。
    """

    target_pos = 10000.0

    K_G = 30.0

    # 扩大位置与速度展示范围
    pos_array = np.linspace(target_pos - 1000.0, target_pos + 1000.0, 1200)
    speed_array_ms = np.linspace(-200.0 / 3.6, 200.0 / 3.6, 1000)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)

    # version 1
    POTENTIAL = _potential_stopping_v1(
        pos=POS,
        speed=SPEED,
        target_pos=target_pos,
    )

    # version 2
    # POTENTIAL = calc_potential_stopping_v2(
    #     pos=POS,
    #     speed=SPEED,
    #     target_pos=target_pos,
    # )

    mode = str(view_mode).lower().strip()
    speed_array_kmh = speed_array_ms * 3.6
    SPEED_KMH = SPEED * 3.6

    if mode == "2d":
        fig, ax = plt.subplots(figsize=(12, 6))

        cmap = plt.get_cmap("YlOrRd")
        c = ax.pcolormesh(
            POS,
            SPEED_KMH,
            POTENTIAL,
            cmap=cmap,
            shading="auto",
            vmin=0.0,
            vmax=K_G,
        )

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])

        if minimal:
            _apply_minimal_axis_style(ax)
        else:
            fig.colorbar(c, ax=ax)
            ax.set_xlabel("Position (m)")
            ax.set_ylabel("Velocity (km/h)")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle=":")

        _apply_transparent_background(fig)
        plt.tight_layout()
        return fig

    if mode == "3d":
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")

        cmap = plt.get_cmap("YlOrRd")
        ax.plot_surface(
            POS,
            SPEED_KMH,
            POTENTIAL,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            vmin=0.0,
            vmax=K_G,
        )

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])
        ax.set_zlim(0, K_G * 1.02)
        ax.view_init(elev=28, azim=-130)

        if minimal:
            _apply_minimal_axis_style(ax)
        else:
            ax.set_xlabel("Position (m)")
            ax.set_ylabel("Velocity (km/h)")
            ax.set_zlabel("Stopping Potential")

        _apply_transparent_background(fig)
        plt.tight_layout()
        return fig

    raise ValueError("view_mode 仅支持 '2d' 或 '3d'")


def plot_stopping_potential_slices(*, minimal: bool = False) -> Figure:
    """
    绘制停站势函数在距离维与速度维上的一维切片，便于调参。
    """

    target_pos = 29270.046

    scale_pos = 500.0  # m
    scale_speed = 10.0  # m/s

    distance_error_array = np.linspace(-600.0, 600.0, 1200)
    pos_array = target_pos + distance_error_array
    speed_array_ms = np.linspace(0.0, 120.0 / 3.6, 1200)

    potential_vs_distance = _potential_stopping_v1(
        pos=pos_array,
        speed=0.0,
        target_pos=target_pos,
    )
    potential_vs_speed = _potential_stopping_v1(
        pos=target_pos,
        speed=speed_array_ms,
        target_pos=target_pos,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(distance_error_array, potential_vs_distance, color="tab:orange")

    axes[1].plot(speed_array_ms * 3.6, potential_vs_speed, color="tab:red")

    if minimal:
        _apply_minimal_axis_style(axes[0])
        _apply_minimal_axis_style(axes[1])
    else:
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        axes[0].axvline(scale_pos, color="gray", linestyle=":", linewidth=1.2)
        axes[0].axvline(-scale_pos, color="gray", linestyle=":", linewidth=1.2)
        axes[0].set_xlabel("stopping error (m)")
        axes[0].set_ylabel(r"$\Phi_D$")
        axes[0].set_title("v = 0")
        axes[0].grid(True, alpha=0.3, linestyle=":")

        axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        axes[1].axvline(scale_speed * 3.6, color="gray", linestyle=":", linewidth=1.2)
        axes[1].set_xlabel("speed (km/h)")
        axes[1].set_ylabel(r"$\Phi_D$")
        axes[1].set_title("d = 0")
        axes[1].grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            r"$\Phi_D$ slice",
            fontsize=13,
        )
    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


def plot_punctuality_potential_curve(
    schedule_time: float = 430.0,
    redundant_time_upper: float = 80.0,
    redundant_time_lower: float = -100.0,
    num_points: int = 1200,
    *,
    minimal: bool = False,
) -> Figure:
    """
    绘制准点势函数关于冗余运行时间的一维曲线。

    Args:
        schedule_time: 规划运行时间(s)。
        redundant_time_upper: 冗余运行时间上界(s)。
        redundant_time_lower: 冗余运行时间下界(s)。
        num_points: 采样点数。
    """

    num_points = max(int(num_points), 2)
    redundant_operation_time_array = np.linspace(
        redundant_time_upper,
        redundant_time_lower,
        num_points,
        dtype=np.float64,
    )

    # version 1
    # potential_array = calc_potential_punctuality_v1(
    #     redundant_operation_time=redundant_operation_time_array,
    #     schedule_time=schedule_time,
    # )

    # version 2
    # potential_array = calc_potential_punctuality_v2(
    #     redundant_operation_time=redundant_operation_time_array,
    #     schedule_time=schedule_time,
    # )

    # version 3
    potential_array = _potential_punctuality_v3(
        redundant_operation_time=redundant_operation_time_array,
        schedule_time=schedule_time,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        redundant_operation_time_array,
        potential_array,
        color="tab:green",
        linewidth=2,
    )

    ax.set_xlim(redundant_time_upper, redundant_time_lower)

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        ax.axvline(
            0.0, color="black", linestyle="--", linewidth=1.2, label=r"$\rho =0$"
        )
        ax.set_xlabel("Redunctant Operation Time (s)")
        ax.set_ylabel("Punctuality Potential")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


PLOT_TYPE_CHOICES: tuple[str, ...] = (
    "safety-speed",
    "safety-position",
    "stopping-heatmap",
    "stopping-slices",
    "punctuality-curve",
)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="展示并可选保存势函数图。")
    parser.add_argument(
        "--plot-type",
        choices=PLOT_TYPE_CHOICES,
        default="stopping-heatmap",
        help="选择展示哪种势函数图。",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="启用图像保存。启用后必须传 --output-file。",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出图像路径（如 output/potential.png）。",
    )
    parser.add_argument(
        "--minimal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="极简图形模式：仅保留核心数据图元，移除文字与辅助标注。",
    )
    return parser


def _validate_cli_args(cli_args: argparse.Namespace) -> None:
    if cli_args.output_file is not None and not cli_args.output_file.strip():
        raise ValueError("--output-file must not be empty")
    if cli_args.save and cli_args.output_file is None:
        raise ValueError("--output-file is required when --save is set")


def _resolve_plotter(plot_type: str, *, minimal: bool) -> Callable[[], Figure]:
    plotters: dict[str, Callable[[], Figure]] = {
        "safety-speed": lambda: plot_safety_potential_heatmap_speed(minimal=minimal),
        "safety-position": lambda: plot_safety_potential_heatmap_position(
            minimal=minimal
        ),
        "stopping-heatmap": lambda: plot_stopping_potential_heatmap(
            view_mode="3d",
            minimal=minimal,
        ),
        "stopping-slices": lambda: plot_stopping_potential_slices(minimal=minimal),
        "punctuality-curve": lambda: plot_punctuality_potential_curve(minimal=minimal),
    }
    return plotters[plot_type]


def _apply_plot_style() -> None:
    set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=12.0,
        axis_label_font_size=12.0,
        tick_font_size=12.0,
        legend_font_size=12.0,
        figure_dpi=100.0,
        savefig_dpi=300.0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    cli_args = parser.parse_args(argv)

    try:
        _validate_cli_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    _apply_plot_style()
    figure = _resolve_plotter(cli_args.plot_type, minimal=cli_args.minimal)()

    if cli_args.save:
        output_path = Path(cli_args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            transparent=True,
            facecolor="none",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"图像已保存: {output_path}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
