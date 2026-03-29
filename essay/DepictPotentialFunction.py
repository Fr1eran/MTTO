import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from matplotlib.colors import LinearSegmentedColormap

from utils.misc import SetChineseFont

SetChineseFont()


def calc_potential_safety(pos, speed, min_speed, max_speed, target_pos):
    """
    向量化安全势能函数
    """
    distanceToTarget = np.abs(target_pos - pos)
    center_speed = (max_speed + min_speed) / 2.0
    safe_margin = (max_speed - min_speed) / 2.0

    # 防止除零错误
    safe_margin = np.maximum(safe_margin, 1e-3)

    norm_speed = (speed - center_speed) / safe_margin

    # 距离敏感系数
    scale = 1.0 + 1.0 * np.exp(-0.002 * distanceToTarget)

    return -4.0 * scale * (norm_speed**2)


def infer_position_from_speed(curve_xy, target_speed):
    """
    在最小速度曲线随位置单调递减时，根据目标速度反推对应位置。
    """
    pos = curve_xy[0, :]
    speed = curve_xy[1, :]

    # np.interp 要求自变量单调递增，因此将递减速度轴反转后执行反向插值。
    return float(np.interp(target_speed, speed[::-1], pos[::-1]))


def plot_safety_potential_heatmap():
    with open("data/rail/safeguard/min_curves_list.pkl", "rb") as f:
        min_curves_list = pickle.load(f)
    with open("data/rail/safeguard/max_curves_list.pkl", "rb") as f:
        max_curves_list = pickle.load(f)

    # 以第7个辅助停车区作为示例
    target_pos = 17828.0

    upper_speed = 200.0 / 3.6
    lower_speed = 0.0

    pos_left_bound = infer_position_from_speed(min_curves_list[6], upper_speed)
    pos_right_bound = max_curves_list[7][0, -1]

    pos_array = np.linspace(pos_left_bound, pos_right_bound, 800)
    speed_array_ms = np.linspace(lower_speed, upper_speed, 500)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)

    speed_min_1d = np.interp(
        pos_array, min_curves_list[6][0, :], min_curves_list[6][1, :]
    )

    speed_max_1d = np.interp(
        pos_array, max_curves_list[7][0, :], max_curves_list[7][1, :]
    )

    speed_min_1d_masked = np.where(speed_min_1d >= lower_speed, speed_min_1d, np.nan)
    speed_max_1d_masked = np.where(speed_max_1d <= upper_speed, speed_max_1d, np.nan)

    speed_min_1d = np.maximum(speed_min_1d, lower_speed)
    speed_max_1d = np.minimum(speed_max_1d, upper_speed)

    SPEED_MIN = np.tile(speed_min_1d, (speed_array_ms.size, 1))
    SPEED_MAX = np.tile(speed_max_1d, (speed_array_ms.size, 1))

    # 计算整个网络的势能值
    POTENTIAL = calc_potential_safety(POS, SPEED, SPEED_MIN, SPEED_MAX, target_pos)

    # 生成 masking, 越界区域的值设为NAN, 使其在图上透明
    POTENTIAL_MASKED = np.where(
        (SPEED >= SPEED_MIN) & (SPEED <= SPEED_MAX), POTENTIAL, np.nan
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    cmap = plt.cm.get_cmap("Spectral")

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
        linestyle="--",
        linewidth=2,
        label=r"速度上限($V_{max}$)",
    )
    ax.plot(
        pos_array,
        speed_min_1d_masked * 3.6,
        color="green",
        linestyle="--",
        linewidth=2,
        label=r"速度下限($V_{min}$)",
    )

    # 添加色标
    cbar = fig.colorbar(c, ax=ax, extend="min")
    cbar.set_label(r"安全势能值 $\Phi_{safety}$", fontsize=12)

    # 图表格式化
    ax.set_xlim(12000, 19000)
    ax.set_ylim(0, 130)
    ax.set_xlabel("位置 (m)")
    ax.set_ylabel("速度 (km/h")
    ax.set_title("磁浮列车运行安全势能场空间分布热力图", fontsize=14, pad=15)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_safety_potential_heatmap()
