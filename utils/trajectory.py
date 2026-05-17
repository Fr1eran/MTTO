from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from model.common import ECC
from model.track import TrackProfile
from model.vehicle import VehicleInfo

__all__ = [
    "OptimizedCurveArtifact",
    "smooth_trajectory",
    "compute_segment_accelerations",
    "compute_comfort_metrics_from_trajectory",
    "compute_cumulative_energy_from_trajectory",
]


@dataclass(frozen=True)
class OptimizedCurveArtifact:
    """优化速度曲线产物的文件定位信息（DP 与 RL 通用）。"""

    npz_path: str
    metrics_path: str


def _deduplicate_consecutive_positions(
    pos_arr: np.ndarray,
    speed_arr: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if pos_arr.size < 2:
        return pos_arr.copy(), speed_arr.copy()

    dedup_pos: list[float] = [float(pos_arr[0])]
    dedup_speed: list[float] = [float(speed_arr[0])]

    for idx in range(1, pos_arr.size):
        pos_val = float(pos_arr[idx])
        speed_val = float(speed_arr[idx])
        if abs(pos_val - dedup_pos[-1]) <= tolerance:
            # 连续位置重复时保留最新速度，避免后续分段速度失真。
            dedup_speed[-1] = speed_val
            continue
        dedup_pos.append(pos_val)
        dedup_speed.append(speed_val)

    return np.asarray(dedup_pos, dtype=np.float64), np.asarray(
        dedup_speed,
        dtype=np.float64,
    )


def smooth_trajectory(
    pos_arr: Sequence[float] | np.ndarray,
    speed_arr: Sequence[float] | np.ndarray,
    *,
    samples_per_segment: int = 20,
    method: str = "uniform_acceleration",
    remove_duplicate_pos: bool = True,
    duplicate_tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """对离散轨迹进行平滑采样（用于展示/分析）。

    默认方法按匀变速模型分段加密采样。输入输出均以位置(m)和速度(m/s)
    序列表示，返回值长度通常大于输入长度。

    Args:
        pos_arr: 离散位置序列 (m)，一维。
        speed_arr: 离散速度序列 (m/s)，一维，与 pos_arr 等长。
        samples_per_segment: 每段采样点数（含端点），必须 >= 2。
        method: 平滑方法，当前仅支持 "uniform_acceleration"。
        remove_duplicate_pos: 是否在平滑前去除连续重复位置点。
        duplicate_tolerance: 连续重复位置判定阈值 (m)，必须 >= 0。

    Returns:
        (smooth_pos, smooth_speed) 二元组，均为 float64 数组。
    """
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)

    if pos.ndim != 1 or speed.ndim != 1:
        raise ValueError("pos_arr and speed_arr must be 1-D arrays")
    if pos.size != speed.size:
        raise ValueError("pos_arr and speed_arr must have equal length")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be >= 2")
    if duplicate_tolerance < 0.0:
        raise ValueError("duplicate_tolerance must be >= 0")
    if method != "uniform_acceleration":
        raise ValueError(f"Unknown method '{method}'. Choices: uniform_acceleration")

    if remove_duplicate_pos:
        pos, speed = _deduplicate_consecutive_positions(
            pos,
            speed,
            tolerance=duplicate_tolerance,
        )

    if pos.size < 2:
        return pos.copy(), speed.copy()

    smooth_pos_parts: list[np.ndarray] = []
    smooth_speed_parts: list[np.ndarray] = []

    for seg_idx in range(pos.size - 1):
        p0 = float(pos[seg_idx])
        p1 = float(pos[seg_idx + 1])
        v0 = float(speed[seg_idx])
        v1 = float(speed[seg_idx + 1])

        ds = p1 - p0
        if abs(ds) <= duplicate_tolerance:
            continue

        s_local = np.linspace(0.0, ds, samples_per_segment, dtype=np.float64)
        if seg_idx > 0:
            s_local = s_local[1:]

        acc = (v1**2 - v0**2) / (2.0 * ds)
        v_sq_local = v0**2 + 2.0 * acc * s_local
        v_local = np.sqrt(np.maximum(v_sq_local, 0.0))
        p_local = p0 + s_local

        smooth_pos_parts.append(p_local)
        smooth_speed_parts.append(v_local)

    if not smooth_pos_parts:
        return pos.copy(), speed.copy()

    smooth_pos = np.concatenate(smooth_pos_parts).astype(np.float64, copy=False)
    smooth_speed = np.concatenate(smooth_speed_parts).astype(np.float64, copy=False)

    smooth_pos[0] = float(pos[0])
    smooth_speed[0] = float(speed[0])
    smooth_pos[-1] = float(pos[-1])
    smooth_speed[-1] = float(speed[-1])

    return smooth_pos, smooth_speed


def compute_segment_accelerations(
    pos_arr: Sequence[float] | np.ndarray,
    speed_arr: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """由离散位置/速度序列反推每段加速度。

    使用匀变速模型：a = (v₁² - v₀²) / (2 * ds)。
    返回长度为 len(pos_arr) - 1 的加速度数组。

    Args:
        pos_arr: 位置序列 (m)，一维。
        speed_arr: 速度序列 (m/s)，一维，与 pos_arr 等长。

    Returns:
        加速度数组 (m/s²)，长度 N-1；输入长度 < 2 时返回空数组。
    """
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)

    if pos.ndim != 1 or speed.ndim != 1:
        raise ValueError("pos_arr and speed_arr must be 1-D arrays")
    if pos.size != speed.size:
        raise ValueError("pos_arr and speed_arr must have equal length")
    if pos.size < 2:
        return np.asarray([], dtype=np.float64)

    ds = np.diff(pos)
    delta_speed_sq = np.square(speed[1:]) - np.square(speed[:-1])
    return np.divide(
        delta_speed_sq,
        2.0 * ds,
        out=np.zeros_like(delta_speed_sq, dtype=np.float64),
        where=np.abs(ds) > 1e-9,
    )


def compute_comfort_metrics_from_trajectory(
    pos_arr: Sequence[float] | np.ndarray,
    speed_arr: Sequence[float] | np.ndarray,
    *,
    max_acc_change: float,
) -> dict[str, float]:
    """按 MTTOEnv 口径计算舒适度指标。

    指标定义:
    - delta_acc_t = |a_t - a_{t-1}|，且 a_0 = 0
    - comfort_tav = sum(delta_acc_t)
    - comfort_rms = sqrt(sum(delta_acc_t²) / N)
    - comfort_er_pct = count(delta_acc_t > max_acc_change) / N * 100

    Args:
        pos_arr: 位置序列 (m)。
        speed_arr: 速度序列 (m/s)。
        max_acc_change: 舒适度超限判定阈值 (m/s²)，对应 TrainService.max_acc_change。

    Returns:
        含 comfort_tav、comfort_rms、comfort_er_pct 的字典。
    """
    if max_acc_change <= 0.0:
        raise ValueError("max_acc_change must be positive")

    acc_arr = compute_segment_accelerations(pos_arr=pos_arr, speed_arr=speed_arr)
    if acc_arr.size == 0:
        return {
            "comfort_tav": 0.0,
            "comfort_rms": 0.0,
            "comfort_er_pct": 0.0,
        }

    prev_acc = np.empty_like(acc_arr)
    prev_acc[0] = 0.0
    prev_acc[1:] = acc_arr[:-1]
    delta_acc = np.abs(acc_arr - prev_acc)

    num_steps = int(delta_acc.size)
    comfort_tav = float(np.sum(delta_acc))
    comfort_rms = float(np.sqrt(np.sum(delta_acc**2) / num_steps))
    comfort_er_pct = float(np.sum(delta_acc > max_acc_change) / num_steps * 100.0)

    return {
        "comfort_tav": comfort_tav,
        "comfort_rms": comfort_rms,
        "comfort_er_pct": comfort_er_pct,
    }


def compute_cumulative_energy_from_trajectory(
    pos_arr: Sequence[float] | np.ndarray,
    speed_arr: Sequence[float] | np.ndarray,
    *,
    vehicle: VehicleInfo,
    trackprofile: TrackProfile,
    ecc: ECC,
) -> np.ndarray:
    """从位置/速度序列逐段重计算累积牵引能耗曲线。

    使用匀变速模型反推每段加速度后，调用 ECC.calc_energy 计算各段能耗，
    返回与 pos_arr 等长的累积能耗数组（第 0 个元素为 0.0）。

    Args:
        pos_arr: 位置序列 (m)，一维，长度 N。
        speed_arr: 速度序列 (m/s)，一维，与 pos_arr 等长。
        vehicle: 车辆参数实例。
        trackprofile: 轨道坡度/曲率查询接口。
        ecc: 牵引能耗计算器实例。

    Returns:
        累积能耗数组 (kJ)，长度 N，cum_energy[0] = 0.0。
    """
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)

    n = pos.size
    cum_energy = np.zeros(n, dtype=np.float64)

    if n < 2:
        return cum_energy

    acc_arr = compute_segment_accelerations(pos, speed)

    for i in range(n - 1):
        displacement = float(pos[i + 1] - pos[i])
        if abs(displacement) < 1e-9:
            cum_energy[i + 1] = cum_energy[i]
            continue

        v0 = float(speed[i])
        v1 = float(speed[i + 1])
        acc = float(acc_arr[i])

        if abs(acc) < 1e-9:
            if v0 < 1e-9:
                cum_energy[i + 1] = cum_energy[i]
                continue
            t = abs(displacement) / v0
        else:
            t = (v1 - v0) / acc

        try:
            prop_e, levi_e = ecc.calc_energy(
                begin_pos=float(pos[i]),
                begin_speed=v0,
                acc=acc,
                distance=abs(displacement),
                direction=1 if displacement > 0 else -1,
                operation_time=t,
                vehicle=vehicle,
                trackprofile=trackprofile,
            )
            cum_energy[i + 1] = cum_energy[i] + (prop_e + levi_e) / 1000.0
        except Exception:
            cum_energy[i + 1] = cum_energy[i]

    return cum_energy
