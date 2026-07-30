import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
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


def _smooth_softplus_risk(z, alpha):
    x = alpha * z
    return np.where(
        x > 20.0,
        z,
        np.log1p(np.exp(np.minimum(x, 20.0))) / alpha,
    )


def _potential_safety_speed_asymmetric_v3(
    pos,
    speed,
    min_speed,
    max_speed,
    target_pos,
) -> float:
    del pos, target_pos

    K_Safety = 0.25
    speed_band = max_speed - min_speed + 0.1
    upper_bound = np.clip(speed_band * 0.15, 2.0, 8.0)
    lower_bound = np.clip(speed_band * 0.15, 2.0, 5.0)
    alpha = 3.0

    margin_max = max_speed - speed
    z_max = 1.0 - margin_max / upper_bound
    smooth_z_max = _smooth_softplus_risk(z_max, alpha)
    phi_max = -(smooth_z_max**2)

    margin_min = speed - min_speed
    z_min = 1.0 - margin_min / lower_bound
    smooth_z_min = _smooth_softplus_risk(z_min, alpha)
    phi_min = np.where(
        min_speed > 0.0,
        -(smooth_z_min**2),
        0.0,
    )

    return K_Safety * (phi_max + phi_min)


def _potential_stopping_v1(
    pos,
    speed,
    target_pos,
):
    """
    向量化停站势函数
    """
    K_weak = 5.0
    K_strong = 50.0
    P_weak = 3000.0
    P_strong = 300.0
    V_weak = 0.7 * 500.0 / 3.6
    V_strong = 0.1 * 500.0 / 3.6

    dist_error_abs = np.abs(pos - target_pos)
    speed_abs = np.abs(speed)

    phi_weak = K_weak * np.exp(-dist_error_abs / P_weak - speed_abs / V_weak)
    phi_strong = K_strong * np.exp(-dist_error_abs / P_strong - speed_abs / V_strong)

    return phi_weak + phi_strong


def _potential_stopping_v2(
    pos,
    speed,
    target_pos,
):
    stop_error_abs = np.abs(target_pos - pos)

    x_hat = stop_error_abs / 3000.0
    v_hat = np.abs(speed * 3.6 / 500.0)

    phi_far = 2.0 * np.exp(-x_hat)

    phi_mid = 8.0 * np.exp(-x_hat / 0.1 - v_hat / 0.2)

    phi_near = 20.0 * np.exp(-x_hat / 0.01 - v_hat / 0.01)

    return phi_far + phi_mid + phi_near


def _potential_stopping_v3(pos, speed, target_pos) -> float:
    K_S = 1.0
    sigma_d = 300.0
    sigma_v = 0.2 * 500.0 / 3.6

    dist_error = pos - target_pos

    gaussian_exp = -((dist_error / sigma_d) ** 2) / 2.0 - ((speed / sigma_v) ** 2) / 2.0

    return K_S * np.exp(gaussian_exp)


def infer_position_from_speed(curve_pos, curve_speed, target_speed):
    """在速度曲线单调递减时，根据目标速度反推对应位置。"""
    return np.interp(
        target_speed,
        curve_speed[::-1],
        curve_pos[::-1],
        left=float(curve_pos[-1]),
        right=float(curve_pos[0]),
    )


def interp_with_constant_fill(x, y, query, left_value, right_value):
    """使用 numpy.interp 做线性插值，并在区间外用常量填充。"""
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
    ax.axison = True


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

    # 非对称解耦平滑自适应势能场 v3
    POTENTIAL = _potential_safety_speed_asymmetric_v3(
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
        vmin=-0.3,
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
            label=r"maximum speed curve",
        )
        ax.plot(
            pos_array,
            speed_min_1d_masked * 3.6,
            color="blue",
            linewidth=1,
            label=r"minimum speed curve",
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
            label="maximum speed curve",
        )
        ax.plot(
            pos_from_min_curve,
            speed_array_ms * 3.6,
            color="blue",
            linewidth=1,
            label="minimum speed curve",
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

    K_S = 1.0

    # 扩大位置与速度展示范围
    pos_array = np.linspace(target_pos - 1000.0, target_pos + 1000.0, 1200)
    speed_array_ms = np.linspace(-500.0 / 3.6, 500.0 / 3.6, 1000)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)

    # version 1
    # POTENTIAL = _potential_stopping_v1(
    #     pos=POS,
    #     speed=SPEED,
    #     target_pos=target_pos,
    # )

    # version 2
    # POTENTIAL = _potential_stopping_v2(
    #     pos=POS,
    #     speed=SPEED,
    #     target_pos=target_pos,
    # )

    # version 3
    POTENTIAL = _potential_stopping_v3(
        pos=POS,
        speed=SPEED,
        target_pos=target_pos,
    )

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
            vmax=K_S,
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
        surface_step = 4
        ax.plot_surface(
            POS[::surface_step, ::surface_step],
            SPEED_KMH[::surface_step, ::surface_step],
            POTENTIAL[::surface_step, ::surface_step],
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            vmin=0.0,
            vmax=K_S,
        )

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])
        ax.set_zlim(0, K_S * 1.02)
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


PLOT_TYPE_CHOICES: tuple[str, ...] = (
    "safety-speed",
    "safety-position",
    "stopping-heatmap",
    "stopping-slices",
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
        "--output-file",
        type=Path,
        default=None,
        help="输出紧凑版图像路径（如 output/potential.png）。不传时仅展示图像。",
    )
    parser.add_argument(
        "--minimal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="极简图形模式：仅保留核心数据图元，移除文字与辅助标注。",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="保存图像但不打开交互式展示窗口。",
    )
    return parser


def _validate_cli_args(cli_args: argparse.Namespace) -> None:
    if cli_args.output_file is not None and str(cli_args.output_file).strip() == "":
        raise ValueError("--output-file must not be empty")


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
    }
    return plotters[plot_type]


def _apply_plot_style() -> None:
    set_global_plot_style(
        font_preset="sci",
        preferred_font="Times New Roman",
        title_font_size=12.0,
        axis_label_font_size=12.0,
        tick_font_size=12.0,
        legend_font_size=12.0,
        figure_dpi=100.0,
        savefig_dpi=300.0,
    )


def _save_compact_figure(figure: Figure, output_file: Path) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_file,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    cli_args = parser.parse_args(argv)

    try:
        _validate_cli_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    _apply_plot_style()
    figure = _resolve_plotter(cli_args.plot_type, minimal=cli_args.minimal)()

    if cli_args.output_file is not None:
        output_path = _save_compact_figure(figure, cli_args.output_file)
        print(f"图像已保存: {output_path}")

    if not cli_args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
