import numpy as np
import matplotlib.pyplot as plt
from typing import Any, cast

from utils.data_loader import load_safeguard_curves
from utils.plot_utils import set_global_plot_style


def calc_potential_safety_speed(pos, speed, min_speed, max_speed, target_pos):
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


def calc_potential_safety_position(pos, min_pos, max_pos, target_pos):
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


def calc_potential_safety_speed_adaptive(pos, speed, min_speed, max_speed, target_pos):
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


def calc_potential_docking(
    pos,
    speed,
    target_pos,
):
    """
    向量化停站势函数
    """
    # 基础参数
    d_scale = 30000.0
    speed_max = 500.0 / 3.6

    sigma_x_hat = 0.2
    sigma_v_hat = 0.1

    # 正则化项
    dist_error = np.abs(target_pos - pos)
    x_hat = dist_error / d_scale
    v_hat = speed / speed_max

    # 增益参数
    K_L = 2.0
    K_G = 20.0

    # 势能
    # phi_linear = -K_L * np.sqrt(x_hat**2 + v_hat**2)
    phi_linear = -K_L * np.sqrt(x_hat**2)

    phi_strong = K_G * np.exp(
        -np.abs(x_hat) / sigma_x_hat - np.abs(v_hat) / sigma_v_hat
    )

    return phi_linear + phi_strong


def calc_potential_punctuality(
    redundant_operation_time,
    schedule_time: float,
):
    # 计算时间冗余度
    time_redundancy_norm = redundant_operation_time / schedule_time

    return -10.0 * np.log1p(np.exp(-12.0 * time_redundancy_norm))


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


def plot_safety_potential_heatmap_speed():
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
    POTENTIAL = calc_potential_safety_speed(
        POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    )

    # 自适应安全目标势能场
    # POTENTIAL = calc_potential_safety_speed_adaptive(
    #     POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    # )

    # 生成 masking, 越界区域的值设为NAN, 使其在图上透明
    in_speed_band = (SPEED >= SPEED_MIN) & (SPEED <= SPEED_MAX)
    POTENTIAL_MASKED = np.where(in_speed_band, POTENTIAL, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

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

    ax.plot(
        pos_array,
        speed_max_1d_masked * 3.6,
        color="red",
        linewidth=1,
        label=r"Upper Speed Curve",
    )
    ax.plot(
        pos_array,
        speed_min_1d_masked * 3.6,
        color="blue",
        linewidth=1,
        label=r"Lower Speed Curve",
    )

    # 添加色标
    fig.colorbar(c, ax=ax, extend="min")

    # 图表格式化
    ax.set_xlim(pos_left_bound + 5000, pos_right_bound + 1000)
    ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6 * 0.75)
    ax.set_xlabel("Position ($m$)")
    ax.set_ylabel("Speed ($km/h$)")
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.show()


def plot_safety_potential_heatmap_position():
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

    POTENTIAL = calc_potential_safety_position(POS, POS_LOWER, POS_UPPER, target_pos)
    in_pos_band = (POS >= POS_LOWER) & (POS <= POS_UPPER)
    POTENTIAL_MASKED = np.where(in_pos_band, POTENTIAL, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

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

    ax.plot(
        pos_from_max_curve,
        speed_array_ms * 3.6,
        color="red",
        linewidth=1,
        label="Upper Speed Curve",
    )
    ax.plot(
        pos_from_min_curve,
        speed_array_ms * 3.6,
        color="blue",
        linewidth=1,
        label="Lower Speed Curve",
    )
    ax.plot(
        safe_center_pos_array,
        speed_array_ms * 3.6,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Safe Center",
    )

    fig.colorbar(c, ax=ax, extend="min")

    ax.set_xlim(pos_left_bound - 1000, pos_right_bound + 1000)
    ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.show()


def plot_docking_potential_heatmap(view_mode="3d"):
    """
    按停站势函数公式绘制势场图。

    Args:
        view_mode: "2d" 绘制热力图, "3d" 绘制三维曲面图。
    """

    target_pos = 29270.046

    K_L = 2.0
    K_G = 20.0

    # 扩大位置与速度展示范围
    pos_array = np.linspace(target_pos - 10000.0, target_pos + 500.0, 1200)
    speed_array_ms = np.linspace(0.0, 480.0 / 3.6, 1000)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)

    POTENTIAL = calc_potential_docking(
        pos=POS,
        speed=SPEED,
        target_pos=target_pos,
    )

    mode = str(view_mode).lower().strip()
    speed_array_kmh = speed_array_ms * 3.6
    SPEED_KMH = SPEED * 3.6
    target_speed_ms = 0.0
    target_speed_kmh = target_speed_ms * 3.6
    target_potential = float(
        calc_potential_docking(
            pos=target_pos,
            speed=target_speed_ms,
            target_pos=target_pos,
        )
    )

    if mode == "2d":
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

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

        ax.scatter(
            target_pos,
            target_speed_kmh,
            color="black",
            s=55,
            marker="o",
            edgecolor="white",
            linewidth=0.8,
            label="目标停车点",
        )

        fig.colorbar(c, ax=ax)

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()
        plt.show()
        return

    if mode == "3d":
        fig = plt.figure(figsize=(12, 6), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        cmap = plt.get_cmap("YlOrRd")
        surface = ax.plot_surface(
            POS,
            SPEED_KMH,
            POTENTIAL,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            vmin=0.0,
            vmax=K_G,
        )

        ax_3d = cast(Any, ax)
        ax_3d.scatter(
            np.array([target_pos]),
            np.array([target_speed_kmh]),
            np.array([target_potential]),
            color="black",
            s=55,
            marker="o",
            depthshade=False,
            label="Target Point",
        )

        fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.08)

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])
        ax.set_zlim(-K_L * 0.8, K_G * 1.02)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_zlabel(r"$\Phi_D$")
        ax.view_init(elev=28, azim=-130)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
        return

    raise ValueError("view_mode 仅支持 '2d' 或 '3d'")


def plot_docking_potential_slices():
    """
    绘制停站势函数在距离维与速度维上的一维切片，便于调参。
    """

    target_pos = 29270.046

    scale_pos = 500.0  # m
    scale_speed = 10.0  # m/s

    distance_error_array = np.linspace(-600.0, 600.0, 1200)
    pos_array = target_pos + distance_error_array
    speed_array_ms = np.linspace(0.0, 120.0 / 3.6, 1200)

    potential_vs_distance = calc_potential_docking(
        pos=pos_array,
        speed=0.0,
        target_pos=target_pos,
    )
    potential_vs_speed = calc_potential_docking(
        pos=target_pos,
        speed=speed_array_ms,
        target_pos=target_pos,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    axes[0].plot(distance_error_array, potential_vs_distance, color="tab:orange")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].axvline(scale_pos, color="gray", linestyle=":", linewidth=1.2)
    axes[0].axvline(-scale_pos, color="gray", linestyle=":", linewidth=1.2)
    axes[0].set_xlabel("停站距离误差 d (m)")
    axes[0].set_ylabel(r"势能值 $\Phi_P$")
    axes[0].set_title("v = 0 时的距离维切片 (|d| 指数衰减)")
    axes[0].grid(True, alpha=0.3, linestyle=":")

    axes[1].plot(speed_array_ms * 3.6, potential_vs_speed, color="tab:red")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(scale_speed * 3.6, color="gray", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("速度 v (km/h)")
    axes[1].set_ylabel(r"势能值 $\Phi_P$")
    axes[1].set_title("d = 0 时的速度维切片 (速度指数衰减)")
    axes[1].grid(True, alpha=0.3, linestyle=":")

    fig.suptitle(
        "停站势函数一维切片",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


def plot_punctuality_potential_curve(
    schedule_time: float = 440.0,
    redundant_time_upper: float = 150.0,
    redundant_time_lower: float = -200.0,
    num_points: int = 1200,
):
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
    potential_array = calc_potential_punctuality(
        redundant_operation_time=redundant_operation_time_array,
        schedule_time=float(schedule_time),
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    ax.plot(
        redundant_operation_time_array,
        potential_array,
        color="tab:green",
        linewidth=2,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="零冗余")

    ax.set_xlim(redundant_time_upper, redundant_time_lower)
    ax.set_xlabel("Redunctant Operation Time (s)")
    ax.set_ylabel(r"$\Phi_{T}$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=300.0,
        savefig_dpi=600.0,
    )
    # plot_safety_potential_heatmap_speed()
    plot_safety_potential_heatmap_position()
    # plot_docking_potential_heatmap(view_mode="3d")
    # plot_docking_potential_heatmap(view_mode="2d")
    # plot_docking_potential_slices()
    # plot_punctuality_potential_curve(
    #     schedule_time=440.0,
    #     redundant_time_upper=150.0,
    #     redundant_time_lower=-200.0,
    # )
