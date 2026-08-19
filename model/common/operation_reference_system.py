from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from model.common.energy_consumption_calculator import ECC
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.indexing_utils import get_interval_index_scalar_numba


@njit(cache=True)
def _get_speed_limits_interval_index_numba(
    speed_limit_intervals: NDArray[np.float64],
    pos: float,
    ascend: bool,
) -> int:
    """根据列车位置查询所在的分段限速区间索引。

    Args:
        speed_limit_intervals: 分段限速区间分界点数组 (m)。
        pos: 当前列车位置 (m)。
        ascend: True 表示向前正向加速搜索（右侧边界闭合），
            False 表示向后逆向制动搜索（左侧边界闭合）。

    Returns:
        所属限速区间的索引。
    """
    return get_interval_index_scalar_numba(pos, speed_limit_intervals, ascend)


@njit(cache=True)
def _calc_mb_descend_operation_numba(
    end_pos: float,
    end_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_dec_abs: float,
    tol: float,
) -> tuple[float, float, int]:
    """向后推导最大制动减速度工况（逆向制动段求解）。

    从目标位置与目标速度 (end_pos, end_speed) 出发，以最大减速度绝对值逆向回溯，
    计算为了满足后续限速约束所需的制动起始位置、耗时以及制动起始点所在的限速区间。

    Args:
        end_pos: 制动目标终止位置 (m)。
        end_speed: 制动目标终止速度 (m/s)。
        speed_limits: 各区间限速值数组 (m/s)。
        speed_limit_intervals: 区间分界点数组 (m)。
        factor: 限速安全折减系数。
        max_dec_abs: 最大制动减速度绝对值 (m/s²)。
        tol: 浮点误差容限 (m)。

    Returns:
        (begin_pos, operation_time, begin_interval) 元组：
        - begin_pos: 制动起始位置 (m)
        - operation_time: 制动耗时 (s)
        - begin_interval: 起始限速区间索引
    """
    n_limits = speed_limits.size
    begin_interval = _get_speed_limits_interval_index_numba(
        speed_limit_intervals, end_pos, False
    )
    begin_pos = end_pos
    operation_time = 0.0

    while begin_interval >= 0:
        mark_pos = speed_limit_intervals[begin_interval]
        begin_speed = speed_limits[begin_interval] * factor
        operation_time = (begin_speed - end_speed) / max_dec_abs
        begin_pos = end_pos - (begin_speed * begin_speed - end_speed * end_speed) / (
            2.0 * max_dec_abs
        )

        # 若逆推的制动起点落在此区间起始分界点之后，说明在当前限速区间内即可完成制动
        if begin_pos > mark_pos or np.abs(begin_pos - mark_pos) <= tol:
            break

        # 否则需要跨越当前区间边界向后回溯，计算在边界处的期望速度
        distance = end_pos - mark_pos
        edge_speed_2 = end_speed * end_speed + 2.0 * max_dec_abs * distance
        if edge_speed_2 < 0.0:
            edge_speed_2 = 0.0
        edge_speed = np.sqrt(edge_speed_2)

        next_idx = begin_interval - 1
        if next_idx < 0:
            next_idx = 0
        next_interval_speed_limit = speed_limits[next_idx] * factor

        # 若边界速度小于前一区间的限速上限，则可以继续向前一个区间扩展制动段
        if (
            edge_speed < next_interval_speed_limit
            or np.abs(edge_speed - next_interval_speed_limit) <= tol
        ):
            begin_interval -= 1
        else:
            break

    if n_limits <= 0:
        clipped_idx = 0
    else:
        if begin_interval < 0:
            clipped_idx = 0
        elif begin_interval >= n_limits:
            clipped_idx = n_limits - 1
        else:
            clipped_idx = begin_interval

    return begin_pos, operation_time, clipped_idx


@njit(cache=True)
def _calc_ma_ascend_operation_numba(
    begin_pos: float,
    begin_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_acc: float,
    tol: float,
) -> tuple[float, float, int]:
    """向前推导最大牵引加速度工况（正向加速段求解）。

    从起始位置与速度 (begin_pos, begin_speed) 出发，以最大加速度正向推导，
    计算列车加速至当前或后续限速上限所需的终止位置、耗时以及加速终止点所在区间。

    Args:
        begin_pos: 加速起始位置 (m)。
        begin_speed: 加速起始速度 (m/s)。
        speed_limits: 各区间限速值数组 (m/s)。
        speed_limit_intervals: 区间分界点数组 (m)。
        factor: 限速安全折减系数。
        max_acc: 最大牵引加速度 (m/s²)。
        tol: 浮点误差容限 (m)。

    Returns:
        (end_pos, operation_time, end_interval) 元组：
        - end_pos: 加速终止位置 (m)
        - operation_time: 加速耗时 (s)
        - end_interval: 终止限速区间索引
    """
    n_limits = speed_limits.size
    end_interval = _get_speed_limits_interval_index_numba(
        speed_limit_intervals, begin_pos, True
    )
    end_pos = begin_pos
    operation_time = 0.0

    while end_interval <= n_limits - 1:
        mark_pos = speed_limit_intervals[end_interval + 1]
        end_speed = speed_limits[end_interval] * factor
        operation_time = (end_speed - begin_speed) / max_acc
        end_pos = begin_pos + (end_speed * end_speed - begin_speed * begin_speed) / (
            2.0 * max_acc
        )

        # 若加速达到限速上限的位置在当前区间右边界之前，说明在当前区间内完成加速
        if end_pos < mark_pos or np.abs(end_pos - mark_pos) <= tol:
            break

        # 否则需要跨越当前区间边界向前推导，计算到达边界处的速度
        distance = mark_pos - begin_pos
        edge_speed_2 = begin_speed * begin_speed + 2.0 * max_acc * distance
        if edge_speed_2 < 0.0:
            edge_speed_2 = 0.0
        edge_speed = np.sqrt(edge_speed_2)

        next_idx = end_interval + 1
        if next_idx < 0:
            next_idx = 0
        elif next_idx >= n_limits:
            next_idx = n_limits - 1
        next_interval_speed_limit = speed_limits[next_idx] * factor

        # 若边界速度小于下一区间的限速上限，则可以继续向后一区间扩展加速段
        if (
            edge_speed < next_interval_speed_limit
            or np.abs(edge_speed - next_interval_speed_limit) <= tol
        ):
            end_interval += 1
        else:
            break

    if n_limits <= 0:
        clipped_idx = 0
    else:
        if end_interval < 0:
            clipped_idx = 0
        elif end_interval >= n_limits:
            clipped_idx = n_limits - 1
        else:
            clipped_idx = end_interval

    return end_pos, operation_time, clipped_idx


@njit(cache=True)
def _append_operation_numba(
    acc_arr: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    count: int,
    acc: float,
    operation_time: float,
) -> int:
    """向预分配的控制工况数组追加一段操作，并返回更新后的有效计数。"""
    acc_arr[count] = acc
    time_arr[count] = operation_time
    return count + 1


@njit(cache=True)
def _nocruise_scenario_numba(
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
    max_acc: float,
    max_dec: float,
    max_dec_abs: float,
    acc_arr: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    count: int,
) -> tuple[int, float, bool]:
    """计算无巡航阶段的极限加减速工况（三角速度曲线或纯制动曲线）。

    当加速曲线与制动曲线在区间内相交，不存在平稳巡航空间时调用。
    根据初始速度和目标速度解析求解交点处的峰值速度 v_peak，
    生成先全加速后全制动的三角工况。
    若初始速度过高以致最小制动距离已超过区间总长度，则全段施加最大制动。

    Returns:
        (count, sceptical_pos, has_sceptical_pos) 元组：
        - count: 更新后的操作计数
        - sceptical_pos: 可疑位置 (m)
        - has_sceptical_pos: 是否存在可疑位置
    """
    min_brake_distance = (begin_speed**2 - end_speed**2) / (2.0 * max_dec_abs)
    sceptical_pos = 0.0
    has_sceptical_pos = False
    if min_brake_distance > (end_pos - begin_pos):
        # 初始速度过高，全段最大制动仍无法完全吻合终点要求时的保护处理
        sceptical_pos = begin_pos - (begin_speed**2 - end_speed**2) / (
            2.0 * max_dec_abs
        )
        has_sceptical_pos = True
        operation_time = (begin_speed - end_speed) / max_dec_abs
        count = _append_operation_numba(
            acc_arr, time_arr, count, max_dec, operation_time
        )
    else:
        # 三角速度曲线：解析求解加速直线与制动直线的交点峰值速度
        speed_peak_2 = (
            2.0 * max_acc * max_dec_abs * (end_pos - begin_pos)
            + max_dec_abs * begin_speed**2
            + max_acc * end_speed**2
        ) / (max_acc + max_dec_abs)
        speed_peak = np.sqrt(speed_peak_2)
        forward_time = (speed_peak - begin_speed) / max_acc
        backward_time = (speed_peak - end_speed) / max_dec_abs
        count = _append_operation_numba(acc_arr, time_arr, count, max_acc, forward_time)
        count = _append_operation_numba(
            acc_arr, time_arr, count, max_dec, backward_time
        )
    return count, sceptical_pos, has_sceptical_pos


@njit(cache=True)
def _cruise_scenario_numba(
    cruise_begin_pos: float,
    cruise_end_pos: float,
    cruise_interval: int,
    ma_time: float,
    mb_time: float,
    speed_limits: NDArray[np.float64],
    factor: float,
    max_acc: float,
    max_dec: float,
    acc_arr: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    count: int,
) -> int:
    """计算含巡航阶段的三段式梯形工况（最大加速 -> 匀速巡航 -> 最大制动）。"""
    cruise_speed = speed_limits[cruise_interval] * factor
    cruise_time = (cruise_end_pos - cruise_begin_pos) / cruise_speed
    count = _append_operation_numba(acc_arr, time_arr, count, max_acc, ma_time)
    count = _append_operation_numba(acc_arr, time_arr, count, 0.0, cruise_time)
    count = _append_operation_numba(acc_arr, time_arr, count, max_dec, mb_time)
    return count


@njit(cache=True)
def min_runtime_operations_numba(
    current_pos: float,
    current_speed: float,
    end_pos: float,
    end_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_acc: float,
    max_dec: float,
    max_dec_abs: float,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """在 Numba nopython 模式下求解满足分段限速约束的最短运行时间工况序列。

    算法核心采用极大值原理（Bang-Coast-Bang 控制策略）：
    1. **边界推导**：
       - 从起点正向求解最大加速段（tow），获得初始加速终点与所在区间；
       - 从终点逆向求解最大制动段（brake），获得最终制动起点与所在区间。
    2. **单区间/交叠快速路径**：
       - 若加速终点区间 >= 制动起点区间，说明加减速在同一区间或交叠，
         直接通过无巡航三角工况或含巡航梯形工况完成求解。
    3. **多区间双向推导与匹配**：
       - 正向扫描限速上升沿（speed_limits[i+1] > speed_limits[i]），
         构建加速操作表 asc_；
       - 逆向扫描限速下降沿（speed_limits[j+1] < speed_limits[j]），
         构建制动操作表 desc_；
       - 沿区间索引推进，综合正向加速与逆向制动边界，动态决定当前区间是
         执行三角加速/制动、梯形巡航还是匀速通过。

    Args:
        current_pos: 起始位置 (m)。
        current_speed: 起始速度 (m/s)。
        end_pos: 目标位置 (m)。
        end_speed: 目标速度 (m/s)。
        speed_limits: 各区间限速值 (m/s)。
        speed_limit_intervals: 各区间分界点位置 (m)。
        factor: 限速安全折减系数（如 0.99）。
        max_acc: 最大牵引加速度 (m/s²)。
        max_dec: 最大制动减速度（负值，m/s²）。
        max_dec_abs: 最大制动减速度绝对值（正值，m/s²）。
        tol: 距离比较的数值容差 (m)。

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            (加速度序列 acc_arr, 对应各段耗时序列 time_arr)。
    """
    n_limits = speed_limits.size
    speed_limits_scaled = speed_limits * factor

    # 1. 求解初始正向牵引段与终止逆向制动段
    tow_end_pos, tow_operation_time, tow_end_interval = _calc_ma_ascend_operation_numba(
        current_pos,
        current_speed,
        speed_limits,
        speed_limit_intervals,
        factor,
        max_acc,
        tol,
    )
    brake_begin_pos, brake_operation_time, brake_begin_interval = (
        _calc_mb_descend_operation_numba(
            end_pos,
            end_speed,
            speed_limits,
            speed_limit_intervals,
            factor,
            max_dec_abs,
            tol,
        )
    )

    capacity = 3 * n_limits + 8
    acc_arr = np.empty(capacity, dtype=np.float64)
    time_arr = np.empty(capacity, dtype=np.float64)
    count = 0

    # 2. 单区间或加减速区间重叠时的快速求解
    if tow_end_interval >= brake_begin_interval:
        if tow_end_pos > brake_begin_pos:
            count, _, _ = _nocruise_scenario_numba(
                current_pos,
                current_speed,
                end_pos,
                end_speed,
                max_acc,
                max_dec,
                max_dec_abs,
                acc_arr,
                time_arr,
                count,
            )
        else:
            count = _cruise_scenario_numba(
                tow_end_pos,
                brake_begin_pos,
                brake_begin_interval,
                tow_operation_time,
                brake_operation_time,
                speed_limits,
                factor,
                max_acc,
                max_dec,
                acc_arr,
                time_arr,
                count,
            )
        return acc_arr[:count], time_arr[:count]

    # 3. 跨多区间情况：预分配正向加速表与逆向制动表
    table_size = n_limits + 4
    asc_begin_pos = np.empty(table_size)
    asc_begin_speed = np.empty(table_size)
    asc_operation_time = np.empty(table_size)
    asc_end_pos = np.empty(table_size)
    asc_end_interval = np.empty(table_size, dtype=np.int64)

    desc_end_pos = np.empty(table_size)
    desc_end_speed = np.empty(table_size)
    desc_operation_time = np.empty(table_size)
    desc_begin_pos = np.empty(table_size)
    desc_begin_interval = np.empty(table_size, dtype=np.int64)
    desc_end_interval = np.empty(table_size, dtype=np.int64)

    # 填充初始牵引段
    n_ascend = 1
    asc_begin_pos[0] = current_pos
    asc_begin_speed[0] = current_speed
    asc_operation_time[0] = tow_operation_time
    asc_end_pos[0] = tow_end_pos
    asc_end_interval[0] = tow_end_interval

    # 正向扫描：在所有限速抬升点推导加速段
    prev_ascend_end_interval = tow_end_interval
    for i in range(tow_end_interval, brake_begin_interval):
        if i >= n_limits - 1:
            continue
        if speed_limits[i + 1] <= speed_limits[i]:
            continue
        next_interval = i + 1
        if next_interval > prev_ascend_end_interval:
            end_pos_i, operation_time_i, end_interval_i = (
                _calc_ma_ascend_operation_numba(
                    speed_limit_intervals[next_interval],
                    speed_limits_scaled[i],
                    speed_limits,
                    speed_limit_intervals,
                    factor,
                    max_acc,
                    tol,
                )
            )
            asc_begin_pos[n_ascend] = speed_limit_intervals[next_interval]
            asc_begin_speed[n_ascend] = speed_limits_scaled[i]
            asc_operation_time[n_ascend] = operation_time_i
            asc_end_pos[n_ascend] = end_pos_i
            asc_end_interval[n_ascend] = end_interval_i
            n_ascend += 1
            prev_ascend_end_interval = end_interval_i

    # 填充终止制动段
    n_descend = 1
    desc_end_pos[0] = end_pos
    desc_end_speed[0] = end_speed
    desc_operation_time[0] = brake_operation_time
    desc_begin_pos[0] = brake_begin_pos
    desc_begin_interval[0] = brake_begin_interval
    desc_end_interval[0] = _get_speed_limits_interval_index_numba(
        speed_limit_intervals, end_pos, False
    )

    # 逆向扫描：在所有限速下降点推导制动段
    prev_descend_begin_interval = brake_begin_interval
    for j in range(brake_begin_interval - 1, tow_end_interval - 1, -1):
        if j < 0:
            continue
        if speed_limits[j + 1] >= speed_limits[j]:
            continue
        if j < prev_descend_begin_interval:
            begin_pos_j, operation_time_j, begin_interval_j = (
                _calc_mb_descend_operation_numba(
                    speed_limit_intervals[j + 1],
                    speed_limits_scaled[j + 1],
                    speed_limits,
                    speed_limit_intervals,
                    factor,
                    max_dec_abs,
                    tol,
                )
            )
            desc_end_pos[n_descend] = speed_limit_intervals[j + 1]
            desc_end_speed[n_descend] = speed_limits_scaled[j + 1]
            desc_operation_time[n_descend] = operation_time_j
            desc_begin_pos[n_descend] = begin_pos_j
            desc_begin_interval[n_descend] = begin_interval_j
            desc_end_interval[n_descend] = j
            n_descend += 1
            prev_descend_begin_interval = j

    # 4. 沿途区间扫描组装最优工况序列
    sceptical_pos = 0.0
    has_sceptical_pos = False
    current_interval = tow_end_interval
    while current_interval <= brake_begin_interval:
        ascend_idx = -1
        for k in range(n_ascend):
            if asc_end_interval[k] >= current_interval:
                ascend_idx = k
                break
        descend_idx = -1
        for k in range(n_descend):
            if desc_begin_interval[k] <= current_interval:
                descend_idx = k
                break

        if ascend_idx >= 0 and descend_idx >= 0:
            if asc_end_pos[ascend_idx] > desc_begin_pos[descend_idx]:
                count, sceptical_pos, has_sceptical_pos = _nocruise_scenario_numba(
                    asc_begin_pos[ascend_idx],
                    asc_begin_speed[ascend_idx],
                    desc_end_pos[descend_idx],
                    desc_end_speed[descend_idx],
                    max_acc,
                    max_dec,
                    max_dec_abs,
                    acc_arr,
                    time_arr,
                    count,
                )
            else:
                count = _cruise_scenario_numba(
                    asc_end_pos[ascend_idx],
                    desc_begin_pos[descend_idx],
                    current_interval,
                    asc_operation_time[ascend_idx],
                    desc_operation_time[descend_idx],
                    speed_limits,
                    factor,
                    max_acc,
                    max_dec,
                    acc_arr,
                    time_arr,
                    count,
                )
            current_interval = desc_end_interval[descend_idx] + 1
        elif ascend_idx >= 0:
            count = _append_operation_numba(
                acc_arr,
                time_arr,
                count,
                max_acc,
                asc_operation_time[ascend_idx],
            )
            cruise_speed = speed_limits[current_interval] * factor
            cruise_distance = (
                speed_limit_intervals[asc_end_interval[ascend_idx] + 1]
                - asc_end_pos[ascend_idx]
            )
            count = _append_operation_numba(
                acc_arr, time_arr, count, 0.0, cruise_distance / cruise_speed
            )
            current_interval = asc_end_interval[ascend_idx] + 1
        elif descend_idx >= 0:
            cruise_start_pos = (
                sceptical_pos
                if has_sceptical_pos
                else speed_limit_intervals[current_interval]
            )
            cruise_distance = desc_begin_pos[descend_idx] - cruise_start_pos
            descend_time = desc_operation_time[descend_idx]
            if cruise_distance < 0.0:
                count = _append_operation_numba(
                    acc_arr, time_arr, count, max_dec, descend_time
                )
                sceptical_pos += (
                    speed_limits[current_interval] * factor * descend_time
                    + 0.5 * max_dec * descend_time**2
                )
                has_sceptical_pos = True
            else:
                cruise_speed = speed_limits[current_interval] * factor
                count = _append_operation_numba(
                    acc_arr, time_arr, count, 0.0, cruise_distance / cruise_speed
                )
                count = _append_operation_numba(
                    acc_arr, time_arr, count, max_dec, descend_time
                )
                sceptical_pos = 0.0
                has_sceptical_pos = False
            current_interval = desc_end_interval[descend_idx] + 1
        else:
            cruise_speed = speed_limits[current_interval] * factor
            cruise_distance = (
                speed_limit_intervals[current_interval + 1]
                - speed_limit_intervals[current_interval]
            )
            count = _append_operation_numba(
                acc_arr, time_arr, count, 0.0, cruise_distance / cruise_speed
            )
            current_interval += 1

    return acc_arr[:count], time_arr[:count]


@njit(cache=True)
def min_operation_time_numba(
    current_pos: float,
    current_speed: float,
    end_pos: float,
    end_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_acc: float,
    max_dec: float,
    max_dec_abs: float,
    tol: float,
) -> float:
    """在 Numba nopython 模式下快速计算并返回两点间最短运行时间总秒数。"""
    _, time_arr = min_runtime_operations_numba(
        current_pos,
        current_speed,
        end_pos,
        end_speed,
        speed_limits,
        speed_limit_intervals,
        factor,
        max_acc,
        max_dec,
        max_dec_abs,
        tol,
    )
    return float(np.sum(time_arr))


def min_operation_time(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> float:
    """计算列车从指定起点状态到终点状态的最短运行时间 (s)。

    Args:
        vehicle: 车辆参数模型。
        track: 轨道与线路分段限速模型。
        factor: 限速安全折减系数。
        begin_pos: 起始位置 (m)。
        begin_speed: 起始速度 (m/s)。
        end_pos: 目标位置 (m)。
        end_speed: 目标速度 (m/s)。

    Returns:
        最短运行时间 (s)。
    """
    return float(
        min_operation_time_numba(
            current_pos=begin_pos,
            current_speed=begin_speed,
            end_pos=end_pos,
            end_speed=end_speed,
            speed_limits=track.speed_limits,
            speed_limit_intervals=track.speed_limit_intervals,
            factor=float(factor),
            max_acc=float(vehicle.max_acc),
            max_dec=float(vehicle.max_dec),
            max_dec_abs=float(vehicle.max_dec_abs),
            tol=1e-9,
        )
    )


def min_operation_time_curve(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """将最短运行时间控制序列积分采样为平滑的 (位置, 速度) 轨迹曲线。

    按固定时间步长 dt = 0.1s 对各控制工况进行运动学采样，生成连续曲线。

    Args:
        vehicle: 车辆参数模型。
        track: 轨道与线路分段限速模型。
        factor: 限速安全折减系数。
        begin_pos: 起始位置 (m)。
        begin_speed: 起始速度 (m/s)。
        end_pos: 目标位置 (m)。
        end_speed: 目标速度 (m/s)。

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            (采样位置序列 pos_arr, 采样速度序列 speed_arr)。
    """
    acc_arr, time_arr = min_runtime_operations_numba(
        current_pos=begin_pos,
        current_speed=begin_speed,
        end_pos=end_pos,
        end_speed=end_speed,
        speed_limits=track.speed_limits,
        speed_limit_intervals=track.speed_limit_intervals,
        factor=float(factor),
        max_acc=float(vehicle.max_acc),
        max_dec=float(vehicle.max_dec),
        max_dec_abs=float(vehicle.max_dec_abs),
        tol=1e-9,
    )
    curve_pos_array = np.array([begin_pos], dtype=np.float64)
    curve_speed_array = np.array([begin_speed], dtype=np.float64)
    for acc, operation_time in zip(acc_arr, time_arr, strict=True):
        operation_time_value = float(operation_time)
        if operation_time_value <= 0:
            continue
        dt = 0.1
        n_steps = max(int(np.floor(operation_time_value / dt)), 2)
        t_samples = np.linspace(
            0.0, operation_time_value, n_steps, endpoint=True, dtype=np.float64
        )
        acc_value = float(acc)
        speeds = begin_speed + acc_value * t_samples
        positions = begin_pos + begin_speed * t_samples + 0.5 * acc_value * t_samples**2
        curve_pos_array = np.concatenate((curve_pos_array[:-1], positions))
        curve_speed_array = np.concatenate((curve_speed_array[:-1], speeds))
        begin_pos = curve_pos_array[-1]
        begin_speed = curve_speed_array[-1]

    if curve_pos_array.size > 1:
        keep_mask = np.empty(curve_pos_array.size, dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = np.diff(curve_pos_array) != 0.0
        curve_pos_array = curve_pos_array[keep_mask]
        curve_speed_array = curve_speed_array[keep_mask]

    return curve_pos_array, curve_speed_array


def max_energy_and_min_operation_time(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    energy_con_calc: ECC,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
    distance: float,
) -> tuple[float, float, float]:
    """计算沿最短运行时间控制序列行进指定距离所需的牵引能耗、悬浮能耗与运行时间。

    常用于在强化学习中提供能量与时间的基准参考值。

    Args:
        vehicle: 车辆参数模型。
        track: 轨道与线路分段限速模型。
        factor: 限速安全折减系数。
        energy_con_calc: 能耗计算器实例 (ECC)。
        begin_pos: 起始位置 (m)。
        begin_speed: 起始速度 (m/s)。
        end_pos: 目标位置 (m)。
        end_speed: 目标速度 (m/s)。
        distance: 需要累积计算的位移距离 (m)。

    Returns:
        tuple[float, float, float]:
            (理论最大牵引能耗 ref_mec (kJ),
             悬浮能耗 ref_lec (kJ),
             运行时间 ref_operation_time (s))。
    """
    acc_arr, time_arr = min_runtime_operations_numba(
        current_pos=begin_pos,
        current_speed=begin_speed,
        end_pos=end_pos,
        end_speed=end_speed,
        speed_limits=track.speed_limits,
        speed_limit_intervals=track.speed_limit_intervals,
        factor=float(factor),
        max_acc=float(vehicle.max_acc),
        max_dec=float(vehicle.max_dec),
        max_dec_abs=float(vehicle.max_dec_abs),
        tol=1e-9,
    )

    ref_mec = 0.0
    ref_lec = 0.0
    ref_operation_time = 0.0
    accumulated_distance = 0.0

    current_pos_i = float(begin_pos)
    current_speed_i = float(begin_speed)

    for acc, operation_time in zip(acc_arr, time_arr, strict=True):
        acc_value = float(acc)
        operation_time_value = float(operation_time)
        segment_distance = (
            current_speed_i * operation_time_value
            + 0.5 * acc_value * operation_time_value**2
        )
        if accumulated_distance + segment_distance >= float(distance):
            remaining_displacement = float(distance) - accumulated_distance
            if np.abs(acc_value) < 1e-9:
                actual_time = remaining_displacement / np.maximum(current_speed_i, 1e-6)
            else:
                discriminant = (
                    current_speed_i**2 + 2 * acc_value * remaining_displacement
                )
                discriminant = max(discriminant, 0)
                actual_time = (np.sqrt(discriminant) - current_speed_i) / acc_value
            pec, lec = energy_con_calc.calc_energy(
                begin_pos=current_pos_i,
                begin_speed=current_speed_i,
                acc=acc_value,
                distance=remaining_displacement,
                direction=1,
                operation_time=actual_time,
                vehicle=vehicle,
                track=track,
            )
            ref_mec += pec
            ref_lec += lec
            ref_operation_time += actual_time
            break
        pec, lec = energy_con_calc.calc_energy(
            begin_pos=current_pos_i,
            begin_speed=current_speed_i,
            acc=acc_value,
            distance=segment_distance,
            direction=1,
            operation_time=operation_time_value,
            vehicle=vehicle,
            track=track,
        )
        ref_mec += pec
        ref_lec += lec
        ref_operation_time += operation_time_value
        accumulated_distance += segment_distance
        current_pos_i += segment_distance
        current_speed_i += acc_value * operation_time_value

    return float(ref_mec), float(ref_lec), float(ref_operation_time)
