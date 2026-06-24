import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from dp.experiment_utils import DP_DEFAULT_SEARCH_DIR
from model.common import ORS
from utils.data_loader import load_safeguard_curves
from utils.plot_utils import set_global_plot_style
from utils.scenario import build_scenario


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


# 一元函数
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


def _potential_punctuality_v4(redundant_operation_time, schedule_time):
    K_peak = 4.0
    K_early = 4.0
    K_late = 20.0
    alpha_late = 8.0

    time_redundancy_norm = np.clip(
        redundant_operation_time / schedule_time,
        -1.0,
        1.0,
    )
    late_error_ratio = -time_redundancy_norm

    return np.where(
        time_redundancy_norm >= 0,
        K_peak - K_early * time_redundancy_norm**2,
        K_peak - K_late / alpha_late * (np.exp(alpha_late * late_error_ratio) - 1),
    )


def _potential_punctuality_v10(redundant_operation_time, schedule_time):
    K_T = 10.0
    gamma = 10.0
    omega = 100.0

    e_redundancy = np.where(
        redundant_operation_time > 0,
        gamma * (redundant_operation_time / schedule_time) ** 2,
        -omega * (redundant_operation_time / schedule_time) ** 2,
    )

    return K_T * e_redundancy


def _potential_punctuality_v11(redundant_operation_time, operation_time, schedule_time):
    """优化版二维准点势函数"""
    K_T = 15.0
    gamma = 1.0
    omega = 15.0
    alpha = 4.0
    margin = 2.0

    ratio = (redundant_operation_time - margin) / schedule_time

    e_redundancy = np.where(
        ratio > 0.0,
        gamma * (ratio**2),
        -omega * (ratio**2) * (1.0 + alpha * ((operation_time / schedule_time) ** 2)),
    )

    return K_T * e_redundancy


def _potential_punctuality_v12(redundant_operation_time, operation_time, schedule_time):
    K_T = 8.0
    min_stage_weight = 0.2
    sigma_early = 0.14
    sigma_late = 0.06
    tail_smooth = 1.0e-6
    T_ref = 100.0

    progress = np.clip(operation_time / schedule_time, 0.0, 1.0)
    rho = redundant_operation_time / T_ref

    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    stage_weight = min_stage_weight + (1.0 - min_stage_weight) * smooth_progress

    sigma = np.where(rho >= 0.0, sigma_early, sigma_late)
    normalized_error = rho / sigma
    pseudo_huber_error = np.sqrt(normalized_error**2 + tail_smooth) - np.sqrt(
        tail_smooth
    )
    punctuality_peak = np.exp(-pseudo_huber_error)

    return K_T * stage_weight * punctuality_peak


def _potential_punctuality_v13(redundant_operation_time, operation_time, schedule_time):
    """
    特征：全域恒正防刷分、C1连续可导抗振荡、多项式长尾防摆烂、时光减损自驱动
    """
    K_T = 15.0
    alpha = 0.3  # 时光绝对流逝成本系数 (0.3确保全域恒正底线)
    gamma_pot = 0.2  # 安全裕量区二次项非线性增益
    omega = 80.0  # 晚点危险区有理屏崖壁阻垒系数

    redundant_operation_time = np.asarray(redundant_operation_time, dtype=np.float64)
    operation_time = np.asarray(operation_time, dtype=np.float64)
    schedule_time = np.asarray(schedule_time, dtype=np.float64)

    # 3. 时空状态变量归一化解耦
    progress = operation_time / schedule_time
    ratio = redundant_operation_time / schedule_time

    # 4. 计算时光无情流逝带来的基础线性势能衰减基底 (缓慢下降驱动源)
    time_consumption_base = 1.0 - alpha * progress

    # 5. 高效向量化解算分段协同流形 (通过数学形态确保C1连续)
    e_redundancy = np.where(
        ratio > 0.0,
        1.0 + gamma_pot * (ratio**2),  # 安全区：抛物线平滑减损向脊线靠拢
        1.0 / (1.0 + omega * (ratio**2)),  # 危险区：逆多项式急速下坠，长尾保持梯度
    )

    # 6. 合成最终全域安全、数值极度稳定的高大梯度势能输出
    potential = K_T * time_consumption_base * e_redundancy

    # 2. 物理可行域安全性约束强校验 (t + r <= T)
    feasible_mask = operation_time + redundant_operation_time <= schedule_time + 1e-3
    return np.where(feasible_mask, potential, 0.0)


def _potential_punctuality_v14(redundant_operation_time, operation_time, schedule_time):
    K_T = 10.0
    omega_pos = 5.0
    omega_neg = 2.0
    T_ref = 100.0
    margin = 2.0

    progress = operation_time / schedule_time
    ratio = (redundant_operation_time - margin) / T_ref

    e_redundancy = np.where(
        ratio > 0.0,
        omega_pos * (ratio**2) * (1.0 - progress),
        -omega_neg * np.log1p(ratio**2) * (1.0 + progress),
    )

    return K_T * e_redundancy


def _potential_punctuality_v16(redundant_operation_time, operation_time, schedule_time):
    omega_pos = 10.0
    omega_neg = 6.0
    T_ref = 100.0
    margin = 2.0
    C_shaping = 10.0

    progress = operation_time / schedule_time
    ratio = (redundant_operation_time - margin) / T_ref

    e_redundancy = np.where(
        ratio > 0.0,
        omega_pos * (ratio**2) * (1.0 - progress),
        -omega_neg * np.arcsinh(-ratio) * (1.0 + 0.6 * progress),
    )

    return e_redundancy + C_shaping


def _potential_punctuality_v18(
    redundant_operation_time,
    ref_redundant_operation_time,
):
    K_T = 1.0
    T_scale = 10.0
    delta = 1.5

    bias = redundant_operation_time - ref_redundant_operation_time
    ratio = bias / T_scale

    # Pseudo-Huber
    phi = -K_T * (delta**2 * (np.sqrt(1.0 + (ratio / delta) ** 2) - 1.0))

    return phi


def _potential_punctuality_v19(redundant_operation_time, ref_redundant_operation_time):
    K_T = 0.8
    T_tol = 6.0

    bias = redundant_operation_time - ref_redundant_operation_time
    ratio = bias / T_tol

    # ln(cosh(x)) 的数值稳定版本
    abs_ratio = abs(ratio)
    phi = -K_T * (abs_ratio + np.log1p(np.exp(-2.0 * abs_ratio)) - np.log(2.0))

    return phi


# 二元函数
def _potential_punctuality_v5(operation_time, redundant_operation_time, schedule_time):
    K_progress = 5.0  # 时间推进系数
    lambda_val = 0.01  # 控制时间推进梯度
    K_redundant = 2.0  # 冗余度奖励系数
    K_late = 5.0  # 晚点衰减系数
    alpha = 5.0  # 晚点惩罚敏感系数
    K_overtime = 15.0  # 超时惩罚系数
    beta = 0.5  # 超时惩罚敏感系数

    time_redundancy_norm = np.clip(redundant_operation_time / schedule_time, -1.0, 1.0)
    remaining_operation_time = schedule_time - operation_time

    progress = K_progress * np.exp(
        -lambda_val * np.maximum(remaining_operation_time, 0.0)
    )

    phi_safe = progress + K_redundant * time_redundancy_norm

    late_penalty = K_late * (np.exp(-alpha * time_redundancy_norm) - 1.0)
    phi_late = progress - late_penalty

    phi = np.where(time_redundancy_norm >= 0.0, phi_safe, phi_late)

    overtime_seconds = np.clip(-remaining_operation_time, 0.0, 60.0)
    cliff_penalty = K_overtime * (np.exp(beta * overtime_seconds) - 1.0)

    phi -= cliff_penalty

    return phi


def _potential_punctuality_v6(operation_time, redundant_operation_time, schedule_time):
    K_T = 20.0
    sigma_tau_early = 300.0
    sigma_tau_late = 180.0
    sigma_rho_early = 240.0
    sigma_rho_late = 60.0

    remaining_schedule_time = schedule_time - operation_time

    e_time = np.where(
        remaining_schedule_time > 0.0,
        np.exp(-((remaining_schedule_time / sigma_tau_early) ** 2)),
        np.exp(-((remaining_schedule_time / sigma_tau_late) ** 2)),
    )

    e_redundancy = np.where(
        redundant_operation_time >= 0.0,
        np.exp(-((redundant_operation_time / sigma_rho_early) ** 2)),
        np.exp(-((redundant_operation_time / sigma_rho_late) ** 2)),
    )

    return K_T * (e_time * e_redundancy)


def _potential_punctuality_v7(operation_time, redundant_operation_time, schedule_time):
    K_T = 10.0
    sigma_tau = 10.0
    sigma_rho = 0.1

    remaining_schedule_time = schedule_time - operation_time
    time_redundancy = redundant_operation_time / schedule_time

    e_time = np.where(
        remaining_schedule_time > 0.0,
        1.0,
        np.exp(-((remaining_schedule_time / sigma_tau) ** 2)),
    )

    e_redundancy = np.where(
        time_redundancy >= 0.0, 1.0, np.exp(-((time_redundancy / sigma_rho) ** 2))
    )

    return K_T * (e_time * e_redundancy)


def _potential_punctuality_v8(operation_time, redundant_operation_time, schedule_time):
    K_T = 15.0
    early_time_lambda = 0.6
    early_redundancy_lambda = 0.6
    late_time_sigma = 60.0
    late_redundancy_sigma = 40.0

    e_time = np.where(
        operation_time <= schedule_time,
        (1 - np.exp(-early_time_lambda * (operation_time / schedule_time)))
        / (1 - np.exp(-early_time_lambda)),
        np.exp(-(((operation_time - schedule_time) / late_time_sigma) ** 2)),
    )

    e_redundancy = np.where(
        redundant_operation_time >= 0.0,
        (
            1
            - np.exp(
                -early_redundancy_lambda
                * (schedule_time - redundant_operation_time)
                / schedule_time
            )
        )
        / (1 - np.exp(-early_redundancy_lambda)),
        np.exp(-((redundant_operation_time / late_redundancy_sigma) ** 2)),
    )

    return K_T * (e_time * e_redundancy)


def _potential_punctuality_v9(operation_time, redundant_operation_time, schedule_time):
    K_T = 5.0
    gamma_t = 0.1
    gamma_r = 0.1
    sigma_t = 100.0
    sigma_r = 80.0

    e_time = np.where(
        operation_time < schedule_time,
        1.0 + gamma_t * (1.0 - operation_time / schedule_time) ** 2,
        np.exp(-(((operation_time - schedule_time) / sigma_t) ** 2)),
    )

    e_redundancy = np.where(
        redundant_operation_time > 0.0,
        1.0 + gamma_r * (redundant_operation_time / schedule_time) ** 2,
        np.exp(-((redundant_operation_time / sigma_r) ** 2)),
    )

    return K_T * e_time * e_redundancy


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


def plot_punctuality_potential_curve(
    schedule_time: float = 430.0,
    operation_time_lower: float = 0.0,
    operation_time_upper: float = 530.0,
    redundant_time_upper: float = 52.0,
    redundant_time_lower: float = -120.0,
    num_points: int = 1200,
    *,
    minimal: bool = False,
) -> Figure:
    """
    绘制二维准点势函数关于运行时间的一维切片曲线。

    Args:
        schedule_time: 规划运行时间(s)。
        operation_time_lower: 运行时间下界(s)。
        operation_time_upper: 运行时间上界(s)。
        redundant_time_upper: 冗余运行时间上界(s)。
        redundant_time_lower: 冗余运行时间下界(s)。
        num_points: 采样点数。
    """
    if schedule_time <= 0.0:
        raise ValueError("schedule_time must be positive")
    if operation_time_lower >= operation_time_upper:
        raise ValueError("operation_time_lower must be less than operation_time_upper")
    if redundant_time_lower >= redundant_time_upper:
        raise ValueError("redundant_time_lower must be less than redundant_time_upper")

    num_points = max(int(num_points), 2)
    operation_time_array = np.linspace(
        operation_time_lower,
        operation_time_upper,
        num_points,
        dtype=np.float64,
    )

    redundant_time_slices = np.array(
        [
            redundant_time_lower,
            0.0,
            redundant_time_upper,
        ],
        dtype=np.float64,
    )
    redundant_time_slices = redundant_time_slices[
        (redundant_time_lower <= redundant_time_slices)
        & (redundant_time_slices <= redundant_time_upper)
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ("tab:red", "tab:green", "tab:blue")
    for redundant_time, color in zip(redundant_time_slices, colors, strict=False):
        potential_array = _potential_punctuality_v12(
            operation_time=operation_time_array,
            redundant_operation_time=np.full_like(operation_time_array, redundant_time),
            schedule_time=schedule_time,
        )
        ax.plot(
            operation_time_array,
            potential_array,
            color=color,
            linewidth=2,
            label=rf"$\rho = {redundant_time:.0f}s$",
        )

    ax.set_xlim(operation_time_lower, operation_time_upper)

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        ax.axvline(
            schedule_time,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$t = T_p$",
        )
        ax.set_xlabel("Operation Time (s)")
        ax.set_ylabel("Punctuality Potential")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


def plot_punctuality_potential_heatmap(
    schedule_time: float = 430.0,
    dp_curve_dir: str = DP_DEFAULT_SEARCH_DIR,
    position_points: int = 2000,
    relative_band_points: int = 2000,
    band_ratio: float = 0.2,
    min_band_half_width: float = 1.0,
    *,
    minimal: bool = False,
) -> Figure:
    if schedule_time <= 0.0:
        raise ValueError("schedule_time must be positive")
    if band_ratio <= 0.0:
        raise ValueError("band_ratio must be positive")
    if min_band_half_width <= 0.0:
        raise ValueError("min_band_half_width must be positive")

    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time
    )
    ors = ORS(vehicle=vehicle, track=track, factor=safeguard_utility.gamma)
    (
        ref_pos_arr,
        _ref_speed_arr,
        _ref_cum_time_arr,
        ref_redundant_arr,
    ) = ors.load_or_build_ref_redundant_operation_time_from_dp(
        start_position=train_service.start_position,
        start_speed=train_service.start_speed,
        target_position=train_service.target_position,
        target_speed=0.0,
        schedule_time_s=schedule_time,
        dp_curve_dir=dp_curve_dir,
    )

    ref_pos_arr = np.asarray(ref_pos_arr, dtype=np.float64)
    ref_redundant_arr = np.asarray(ref_redundant_arr, dtype=np.float64)
    if ref_pos_arr.ndim != 1 or ref_redundant_arr.ndim != 1:
        raise ValueError("Reference position and redundancy arrays must be 1-D")
    if ref_pos_arr.size != ref_redundant_arr.size:
        raise ValueError("Reference position and redundancy arrays must match length")
    if ref_pos_arr.size < 2:
        raise ValueError("Reference DP curve must contain at least two points")

    sort_idx = np.argsort(ref_pos_arr)
    ref_pos_sorted = ref_pos_arr[sort_idx]
    ref_redundant_sorted = ref_redundant_arr[sort_idx]
    unique_pos, unique_idx = np.unique(ref_pos_sorted, return_index=True)
    unique_ref_redundant = ref_redundant_sorted[unique_idx]
    if unique_pos.size < 2:
        raise ValueError(
            "Reference DP curve must contain at least two unique positions"
        )

    position_array = np.linspace(
        float(unique_pos[0]),
        float(unique_pos[-1]),
        max(int(position_points), 2),
        dtype=np.float64,
    )
    ref_redundant_at_pos = np.interp(
        position_array,
        unique_pos,
        unique_ref_redundant,
    )
    half_width = np.maximum(
        band_ratio * np.abs(ref_redundant_at_pos),
        float(min_band_half_width),
    )
    relative_band = np.linspace(
        -1.0,
        1.0,
        max(int(relative_band_points), 2),
        dtype=np.float64,
    )
    POSITION, RELATIVE_BAND = np.meshgrid(position_array, relative_band)
    REF_REDUNDANT = np.broadcast_to(
        ref_redundant_at_pos,
        POSITION.shape,
    )
    HALF_WIDTH = np.broadcast_to(half_width, POSITION.shape)
    REDUNDANT_TIME = REF_REDUNDANT + RELATIVE_BAND * HALF_WIDTH
    POTENTIAL = _potential_punctuality_v19(
        redundant_operation_time=REDUNDANT_TIME,
        ref_redundant_operation_time=REF_REDUNDANT,
    )

    finite_potential = POTENTIAL[np.isfinite(POTENTIAL)]
    if finite_potential.size == 0:
        raise ValueError("The configured ranges do not contain finite potential values")

    z_min = float(np.min(finite_potential))
    z_max = float(np.max(finite_potential))
    if z_min < 0.0 < z_max:
        color_norm = TwoSlopeNorm(vmin=z_min, vcenter=0.0, vmax=z_max)
    else:
        color_norm = None

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("Spectral")
    heatmap = ax.pcolormesh(
        POSITION,
        REDUNDANT_TIME,
        POTENTIAL,
        cmap=cmap,
        vmin=-0.5,
        vmax=0.0,
    )
    # heatmap = ax.contourf(
    #     POSITION,
    #     REDUNDANT_TIME,
    #     POTENTIAL,
    #     levels=60,
    #     cmap="coolwarm",
    #     norm=color_norm,
    # )
    ax.plot(
        position_array,
        ref_redundant_at_pos,
        color="#111111",
        linestyle="--",
        linewidth=2.0,
        label=r"$\rho ^*$",
    )
    ax.set_xlim(float(position_array[0]), float(position_array[-1]))
    ax.set_ylim(
        float(np.nanmin(REDUNDANT_TIME)),
        float(np.nanmax(REDUNDANT_TIME)),
    )

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.colorbar(heatmap, ax=ax, extend="min")
        # colorbar = fig.colorbar(heatmap, ax=ax)
        # colorbar.set_label("Punctuality potential")
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Redundant Operation Time (s)")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle=":")

    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


PLOT_TYPE_CHOICES: tuple[str, ...] = (
    "safety-speed",
    "safety-position",
    "stopping-heatmap",
    "stopping-slices",
    "punctuality-curve",
    "punctuality-heatmap",
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
        "punctuality-curve": lambda: plot_punctuality_potential_curve(minimal=minimal),
        "punctuality-heatmap": lambda: plot_punctuality_potential_heatmap(
            minimal=minimal
        ),
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
