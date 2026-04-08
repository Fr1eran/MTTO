import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Any, cast

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_safeguard_curves
from utils.misc import SetChineseFont

SetChineseFont()
# SimHei 缺少 U+2212 负号字形，关闭 Unicode 负号可避免告警。
plt.rcParams["axes.unicode_minus"] = False


def calc_potential_safety_speed(pos, speed, min_speed, max_speed, target_pos):
    """
    向量化安全势能函数

    速度带中间位置势能最大
    """
    distanceToTarget = abs(target_pos - pos)
    center_speed = (max_speed + min_speed) / 2.0
    safe_margin = np.maximum((max_speed - min_speed) / 2.0, 0.5)

    # 基础偏离惩罚(二次方项, 引导列车走中间)
    norm_speed_diff = (speed - center_speed) / safe_margin
    phi_base = -5.0 * (norm_speed_diff**2)

    # 边界壁垒

    # 靠近目标位置时，适当增大惩罚力度
    scale = 1.0 + 1.0 * np.exp(-0.001 * distanceToTarget)

    return scale * phi_base


def calc_potential_safety_spatial(pos, min_pos, max_pos, target_pos):
    distanceToTarget = np.abs(target_pos - pos)
    center_pos = (max_pos + min_pos) / 2.0
    safe_margin = (max_pos - min_pos) / 2.0

    norm_pos_diff = (pos - center_pos) / safe_margin
    phi_base = -5.0 * (norm_pos_diff**2)

    scale = 1.0 + 1.0 * np.exp(-0.001 * distanceToTarget)

    return scale * phi_base


def calc_potential_docking(
    pos,
    speed,
    target_pos,
):
    """
    向量化停站势函数（对应图中公式）

    Phi_P(s) = K_P * exp(-d^2 / (2 * sigma_d^2)) * exp(-v^2 / (2 * sigma_v^2))
    其中 d = target_pos - pos, v = speed。
    """
    dist_error = target_pos - pos

    # 广域引导: 在距离目标5km时就知道终点在哪
    phi_wide = (
        1.0
        * np.exp(-(dist_error**2) / (2.0 * 2000.0**2))
        * np.exp(-(speed**2) / (2.0 * 50.0**2))
    )

    # 精确引导
    phi_tight = (
        4.0
        * np.exp(-(dist_error**2) / (2.0 * 100.0**2))
        * np.exp(-(speed**2) / (2.0 * 5.0**2))
    )

    return phi_wide + phi_tight


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

    pos_array = np.linspace(pos_left_bound, pos_right_bound, 800)
    speed_array_ms = np.linspace(lower_speed, upper_speed, 1000)

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
    POTENTIAL = calc_potential_safety_speed(
        POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos
    )

    # 生成 masking, 越界区域的值设为NAN, 使其在图上透明
    POTENTIAL_MASKED = np.where(
        (SPEED >= SPEED_MIN) & (SPEED <= SPEED_MAX), POTENTIAL, np.nan
    )

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
        label=r"速度上限($V_{max}$)",
    )
    ax.plot(
        pos_array,
        speed_min_1d_masked * 3.6,
        color="blue",
        linewidth=1,
        label=r"速度下限($V_{min}$)",
    )

    # 添加色标
    cbar = fig.colorbar(c, ax=ax, extend="min")
    cbar.set_label(r"安全势能值 $\Phi_{safety}$", fontsize=12)

    # 图表格式化
    ax.set_xlim(pos_left_bound - 1000, pos_right_bound + 1000)
    ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6)
    ax.set_xlabel("位置 (m)")
    ax.set_ylabel("速度 (km/h)")
    ax.set_title("磁浮列车运行安全势能场空间分布热力图", fontsize=14, pad=15)
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
    speed_array_ms = np.linspace(lower_speed, upper_speed, 1000)
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

    pos_array = np.linspace(pos_left_bound, pos_right_bound, 800)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)
    POS_LOWER = np.tile(pos_lower_1d[:, None], (1, pos_array.size))
    POS_UPPER = np.tile(pos_upper_1d[:, None], (1, pos_array.size))

    POTENTIAL = calc_potential_safety_spatial(POS, POS_LOWER, POS_UPPER, target_pos)
    POTENTIAL_MASKED = np.where(
        (POS >= POS_LOWER) & (POS <= POS_UPPER), POTENTIAL, np.nan
    )

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
        label=r"速度上限($V_{max}$)",
    )
    ax.plot(
        pos_from_min_curve,
        speed_array_ms * 3.6,
        color="blue",
        linewidth=1,
        label=r"速度下限($V_{min}$)",
    )
    ax.plot(
        safe_center_pos_array,
        speed_array_ms * 3.6,
        color="black",
        linestyle="--",
        linewidth=2,
        label=r"安全中心带",
    )

    cbar = fig.colorbar(c, ax=ax, extend="min")
    cbar.set_label(r"安全势能值 $\Phi_{safety}$", fontsize=12)

    ax.set_xlim(pos_left_bound - 1000, pos_right_bound + 1000)
    ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6)
    ax.set_xlabel("位置 (m)")
    ax.set_ylabel("速度 (km/h)")
    ax.set_title("磁浮列车运行位置安全势能场空间分布热力图", fontsize=14, pad=15)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.show()


def plot_stop_potential_heatmap(view_mode="3d"):
    """
    按停站势函数公式绘制势场图。

    Args:
        view_mode: "2d" 绘制热力图, "3d" 绘制三维曲面图。
    """

    target_pos = 17828.0

    k_p = 5.0

    # 扩大位置与速度展示范围
    pos_array = np.linspace(target_pos - 10000.0, target_pos + 10000.0, 1200)
    speed_array_ms = np.linspace(0.0, 360.0 / 3.6, 1000)

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
            vmax=k_p,
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

        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label(r"停站势能值 $\Phi_P$", fontsize=12)

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])
        ax.set_xlabel("位置 (m)")
        ax.set_ylabel("速度 (km/h)")
        ax.set_title(
            r"停站势函数热力图: $\Phi_P = K_P e^{-d^2/(2\sigma_d^2)} e^{-v^2/(2\sigma_v^2)}$",
            fontsize=13,
            pad=12,
        )
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
            vmax=k_p,
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
            label="目标停车点",
        )

        cbar = fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.08)
        cbar.set_label(r"停站势能值 $\Phi_P$", fontsize=12)

        ax.set_xlim(pos_array[0], pos_array[-1])
        ax.set_ylim(speed_array_kmh[0], speed_array_kmh[-1])
        ax.set_zlim(0.0, k_p * 1.02)
        ax.set_xlabel("位置 (m)")
        ax.set_ylabel("速度 (km/h)")
        ax.set_zlabel(r"势能值 $\Phi_P$")
        ax.set_title(
            r"停站势函数三维曲面图: $\Phi_P = K_P e^{-d^2/(2\sigma_d^2)} e^{-v^2/(2\sigma_v^2)}$",
            fontsize=13,
            pad=12,
        )
        ax.view_init(elev=28, azim=-130)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
        return

    raise ValueError("view_mode 仅支持 '2d' 或 '3d'")


def plot_stop_potential_slices():
    """
    绘制停站势函数在距离维与速度维上的一维切片，便于调参。
    """

    target_pos = 17828.0

    sigma_d = 120.0  # m
    sigma_v = 20.0 / 3.6  # m/s

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
    axes[0].axvline(sigma_d, color="gray", linestyle=":", linewidth=1.2)
    axes[0].axvline(-sigma_d, color="gray", linestyle=":", linewidth=1.2)
    axes[0].set_xlabel("停站距离误差 d (m)")
    axes[0].set_ylabel(r"势能值 $\Phi_P$")
    axes[0].set_title("v = 0 时的距离维切片")
    axes[0].grid(True, alpha=0.3, linestyle=":")

    axes[1].plot(speed_array_ms * 3.6, potential_vs_speed, color="tab:red")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(sigma_v * 3.6, color="gray", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("速度 v (km/h)")
    axes[1].set_ylabel(r"势能值 $\Phi_P$")
    axes[1].set_title("d = 0 时的速度维切片")
    axes[1].grid(True, alpha=0.3, linestyle=":")

    fig.suptitle("停站势函数一维切片", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_safety_potential_heatmap_speed()
    # plot_safety_potential_heatmap_position()
    plot_stop_potential_heatmap(view_mode="3d")
    # plot_stop_potential_heatmap(view_mode="2d")
    # plot_stop_potential_slices()
