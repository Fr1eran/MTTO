from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from mtto.domain.dynamics import (
    Vehicle,
    calc_brake_deceleration_scalar_numba,
    calc_levi_deceleration_scalar_numba,
)
from mtto.domain.line import Line
from mtto.domain.safeguard.static_region import StaticRegion, build_static_region

__all__ = [
    "SafeguardParams",
    "Safeguard",
    "build_safeguard",
    "SafeGuardCurves",
    "SafeguardCurveConfig",
    "CalculationInputs",
    "calculate_curves",
]


@dataclass(frozen=True, slots=True)
class SafeguardParams:
    """Safeguard operating parameters and curve generation inputs."""

    factor: float
    step_delay_s: float
    distance_step_m: float
    position_error_m: float
    speed_error_mps: float
    traction_cutoff_delay_s: float
    vortex_brake_delay_s: float
    min_curve_position_offset_m: float
    generation_danger_points_m: tuple[float, ...]

    def __post_init__(self) -> None:
        numeric_values = (
            self.factor,
            self.step_delay_s,
            self.distance_step_m,
            self.position_error_m,
            self.speed_error_mps,
            self.traction_cutoff_delay_s,
            self.vortex_brake_delay_s,
            self.min_curve_position_offset_m,
        )
        if not all(math.isfinite(v) for v in numeric_values):
            raise ValueError("All SafeguardParams numeric values must be finite")
        if not (0.0 < self.factor <= 1.0):
            raise ValueError(f"factor must be in (0, 1], got {self.factor}")
        if self.step_delay_s <= 0.0:
            raise ValueError(f"step_delay_s must be positive, got {self.step_delay_s}")
        if self.distance_step_m <= 0.0:
            raise ValueError(
                f"distance_step_m must be positive, got {self.distance_step_m}"
            )
        if self.position_error_m < 0.0:
            raise ValueError(
                f"position_error_m must be non-negative, got {self.position_error_m}"
            )
        if self.speed_error_mps < 0.0:
            raise ValueError(
                f"speed_error_mps must be non-negative, got {self.speed_error_mps}"
            )
        if self.traction_cutoff_delay_s < 0.0:
            raise ValueError(
                "traction_cutoff_delay_s must be non-negative, "
                f"got {self.traction_cutoff_delay_s}"
            )
        if self.vortex_brake_delay_s < 0.0:
            raise ValueError(
                "vortex_brake_delay_s must be non-negative, "
                f"got {self.vortex_brake_delay_s}"
            )
        if len(self.generation_danger_points_m) == 0:
            raise ValueError("generation_danger_points_m must not be empty")
        if not all(math.isfinite(dp) for dp in self.generation_danger_points_m):
            raise ValueError("All points in generation_danger_points_m must be finite")
        for i in range(len(self.generation_danger_points_m) - 1):
            if (
                self.generation_danger_points_m[i + 1]
                <= self.generation_danger_points_m[i]
            ):
                raise ValueError(
                    "generation_danger_points_m must be strictly increasing"
                )


@dataclass(frozen=True, slots=True, eq=False)
class Safeguard:
    params: SafeguardParams
    speed_limits: NDArray[np.float64]  # m/s，与 Line 共用同一只读数组
    speed_limit_intervals: NDArray[np.float64]
    levi_curves: tuple[NDArray[np.float64], ...]  # 经 _sanitize_curve 处理
    brake_curves: tuple[NDArray[np.float64], ...]
    min_curves: tuple[NDArray[np.float64], ...]
    max_curves: tuple[NDArray[np.float64], ...]
    # numba 查询用的打包数组（原 _build_speed_query_cache 的结果）
    min_pos_packed: NDArray[np.float64]
    min_speed_packed: NDArray[np.float64]
    min_lengths: NDArray[np.int32]
    max_pos_packed: NDArray[np.float64]
    max_speed_packed: NDArray[np.float64]
    max_lengths: NDArray[np.int32]
    static_region: StaticRegion  # 构造时即计算（原惰性的区域缓存）


def _sanitize_curve(curve: NDArray[np.floating]) -> NDArray[np.float64]:
    """标准化防护曲线, 并将速度数组投影为单调不增。"""
    curve_arr = np.asarray(curve, dtype=np.float64)
    if curve_arr.ndim != 2 or curve_arr.shape[0] != 2:
        raise ValueError("curve must have shape (2, N)")

    curve_pos = np.asarray(curve_arr[0, :], dtype=np.float64)
    curve_speed = np.asarray(curve_arr[1, :], dtype=np.float64)

    if curve_pos.shape != curve_speed.shape:
        raise ValueError("curve_pos and curve_speed must have the same shape")
    if curve_pos.size == 0:
        raise ValueError("curve must contain at least one point")
    if curve_pos.size > 1 and np.any(np.diff(curve_pos) <= 0.0):
        raise ValueError("curve_pos must be strictly increasing")

    if curve_speed.size > 1:
        curve_speed = np.minimum.accumulate(curve_speed)

    return np.stack([curve_pos, curve_speed], axis=0, dtype=np.float64)


def _sanitize_curve_list(
    curves: Sequence[NDArray[np.floating]],
    *,
    curve_name: str,
) -> list[NDArray[np.float64]]:
    sanitized: list[NDArray[np.float64]] = []
    for idx, curve in enumerate(curves):
        try:
            sanitized.append(_sanitize_curve(curve))
        except ValueError as exc:
            raise ValueError(f"{curve_name}[{idx}] is invalid: {exc}") from exc
    return sanitized


def build_safeguard(
    params: SafeguardParams,
    line: Line,
    levi_curves: Sequence[NDArray[np.floating]],
    brake_curves: Sequence[NDArray[np.floating]],
    min_curves: Sequence[NDArray[np.floating]],
    max_curves: Sequence[NDArray[np.floating]],
) -> Safeguard:
    sanitized_levi = _sanitize_curve_list(levi_curves, curve_name="levi_curves")
    sanitized_brake = _sanitize_curve_list(brake_curves, curve_name="brake_curves")
    sanitized_min = _sanitize_curve_list(min_curves, curve_name="min_curves")
    sanitized_max = _sanitize_curve_list(max_curves, curve_name="max_curves")

    n_max = len(sanitized_max)
    n_min = len(sanitized_min)
    if n_max != n_min + 1:
        raise ValueError(
            "Safeguard requires len(max_curves) == len(min_curves) + 1, "
            f"got len(max_curves)={n_max} and len(min_curves)={n_min}"
        )

    # 构造打包查询缓存
    min_curve_count = len(sanitized_min)
    max_curve_count = len(sanitized_max)

    min_max_len = max((curve.shape[1] for curve in sanitized_min), default=1)
    max_max_len = max((curve.shape[1] for curve in sanitized_max), default=1)

    min_pos_packed = np.empty((min_curve_count, min_max_len), dtype=np.float64)
    min_speed_packed = np.empty((min_curve_count, min_max_len), dtype=np.float64)
    min_lengths = np.empty((min_curve_count,), dtype=np.int32)

    for idx, curve in enumerate(sanitized_min):
        curve_len = int(curve.shape[1])
        min_lengths[idx] = curve_len
        min_pos_packed[idx, :curve_len] = curve[0, :]
        min_speed_packed[idx, :curve_len] = curve[1, :]
        if curve_len < min_max_len:
            min_pos_packed[idx, curve_len:] = curve[0, curve_len - 1]
            min_speed_packed[idx, curve_len:] = curve[1, curve_len - 1]

    max_pos_packed = np.empty((max_curve_count, max_max_len), dtype=np.float64)
    max_speed_packed = np.empty((max_curve_count, max_max_len), dtype=np.float64)
    max_lengths = np.empty((max_curve_count,), dtype=np.int32)

    for idx, curve in enumerate(sanitized_max):
        curve_len = int(curve.shape[1])
        max_lengths[idx] = curve_len
        max_pos_packed[idx, :curve_len] = curve[0, :]
        max_speed_packed[idx, :curve_len] = curve[1, :]
        if curve_len < max_max_len:
            max_pos_packed[idx, curve_len:] = curve[0, curve_len - 1]
            max_speed_packed[idx, curve_len:] = curve[1, curve_len - 1]

    static_region = build_static_region(sanitized_min, sanitized_max)

    # 全部派生计算完成后，逐一设为只读
    line.speed_limits.flags.writeable = False
    line.speed_limit_intervals.flags.writeable = False
    for c in sanitized_levi:
        c.flags.writeable = False
    for c in sanitized_brake:
        c.flags.writeable = False
    for c in sanitized_min:
        c.flags.writeable = False
    for c in sanitized_max:
        c.flags.writeable = False
    min_pos_packed.flags.writeable = False
    min_speed_packed.flags.writeable = False
    min_lengths.flags.writeable = False
    max_pos_packed.flags.writeable = False
    max_speed_packed.flags.writeable = False
    max_lengths.flags.writeable = False

    return Safeguard(
        params=params,
        speed_limits=line.speed_limits,
        speed_limit_intervals=line.speed_limit_intervals,
        levi_curves=tuple(sanitized_levi),
        brake_curves=tuple(sanitized_brake),
        min_curves=tuple(sanitized_min),
        max_curves=tuple(sanitized_max),
        min_pos_packed=min_pos_packed,
        min_speed_packed=min_speed_packed,
        min_lengths=min_lengths,
        max_pos_packed=max_pos_packed,
        max_speed_packed=max_speed_packed,
        max_lengths=max_lengths,
        static_region=static_region,
    )


# ---------------------------------------------------------------------------
# 防护曲线计算与辅助函数（原 safe_guard_curves.py 全部内容）
# ---------------------------------------------------------------------------


def _get_slope(
    pos: NDArray[np.float64],
    slopes: NDArray[np.float64],
    slope_intervals: NDArray[np.float64],
) -> NDArray[np.float64]:
    idx = np.clip(
        np.searchsorted(slope_intervals, pos, side="right") - 1,
        0,
        len(slopes) - 1,
    )
    return slopes[idx].astype(np.float32).astype(np.float64)


def _get_speed_limit(
    pos: NDArray[np.float64],
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
) -> NDArray[np.float64]:
    idx = np.clip(
        np.searchsorted(speed_limit_intervals, pos, side="right") - 1,
        0,
        len(speed_limits) - 1,
    )
    return speed_limits[idx].astype(np.float32).astype(np.float64)


def _calc_levi_deceleration(
    mass: float,
    numoftrainsets: int,
    speed: NDArray[np.float64],
    slope: NDArray[np.float64],
) -> NDArray[np.float64]:
    speed = np.asarray(speed, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    speed_km = 3.6 * speed
    u = -0.003 * speed_km + 0.27
    f_sledge = np.where(
        speed_km <= 10.0,
        0.1 * u * mass * 100.0 / np.sqrt(100.0**2 + slope**2) * 9.8,
        0.0,
    )
    f_air = 2.8 * (0.53 * numoftrainsets / 2 + 0.3) * speed**2 / 1000.0
    f_guide = numoftrainsets * (
        0.1 * np.power(speed_km, 0.5) + 0.02 * np.power(speed_km, 0.7)
    )
    f_gen = np.piecewise(
        speed_km,
        [
            speed_km < 20,
            (speed_km >= 20) & (speed_km < 70),
            (speed_km >= 70) & (speed_km < 600),
        ],
        [
            lambda s: 0.0,
            lambda s: 7.3 * numoftrainsets,
            lambda s: 146.0 * 3.6 * numoftrainsets / s - 0.2,
        ],
    )
    f_grad = 9.8 * mass * slope / 100.0
    f_total = f_air + f_guide + f_gen + f_grad + f_sledge
    return f_total / mass


@njit(cache=True)
def _build_curve_backward_with_truncate_numba(
    begin: float,
    ds: float,
    slopes: NDArray[np.float64],
    speed_limits: NDArray[np.float64],
    mass: float,
    numoftrainsets: float,
    dec_mode: int,
):
    speed = 0.0
    pos = begin
    max_steps = int(pos // ds)
    speed_arr = np.empty(max_steps + 1, dtype=np.float64)
    speed_arr[0] = 0.0
    idx = 0

    for j in range(max_steps):
        if dec_mode == 0:
            dec = calc_levi_deceleration_scalar_numba(
                speed,
                slopes[j],
                mass,
                numoftrainsets,
            )
        else:
            dec = calc_brake_deceleration_scalar_numba(
                speed,
                slopes[j],
                mass,
                numoftrainsets,
                0,
            )
        next_speed_squared = speed**2 + 2.0 * dec * ds
        if next_speed_squared < 0.0:
            break
        next_speed = np.sqrt(next_speed_squared)
        if next_speed >= speed_limits[j]:
            break
        pos = pos - ds
        speed = next_speed
        idx += 1
        speed_arr[idx] = next_speed

    out_len = idx
    out_pos = np.empty(out_len, dtype=np.float64)
    out_speed = np.empty(out_len, dtype=np.float64)
    for k in range(out_len):
        out_pos[k] = begin - ds * (out_len - 1 - k)
        out_speed[k] = speed_arr[out_len - 1 - k]
    return out_pos, out_speed


@njit(cache=True)
def _build_curve_backward_without_truncate_numba(
    begin: float,
    ds: float,
    slopes: NDArray[np.float64],
    mass: float,
    numoftrainsets: float,
    dec_mode: int,
):
    n = int(np.ceil(begin / ds))
    if n <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    pos_desc = np.empty(n, dtype=np.float64)
    speed_desc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        pos_desc[i] = begin - ds * i

    speed = 0.0
    for j in range(1, n):
        if dec_mode == 0:
            dec = calc_levi_deceleration_scalar_numba(
                speed,
                slopes[j - 1],
                mass,
                numoftrainsets,
            )
        else:
            dec = calc_brake_deceleration_scalar_numba(
                speed,
                slopes[j - 1],
                mass,
                numoftrainsets,
                0,
            )
        next_speed_squared = speed**2 + 2.0 * dec * ds
        if next_speed_squared < 0.0:
            next_speed_squared = 0.0
        speed = np.sqrt(next_speed_squared)
        speed_desc[j] = speed

    out_pos = np.empty(n, dtype=np.float64)
    out_speed = np.empty(n, dtype=np.float64)
    for k in range(n):
        out_pos[k] = pos_desc[n - 1 - k]
        out_speed[k] = speed_desc[n - 1 - k]
    return out_pos, out_speed


class SafeGuardCurves:
    """上海磁浮示范线线路安全防护计算类"""

    def __init__(self, track: Line):
        self.track: Line = track

    def _calculate_curves_backward_with_truncate(
        self,
        begins: NDArray[np.floating],
        ds: float,
        *,
        dec_mode: int,
        vehicle: Vehicle,
    ) -> list[NDArray[np.float64]]:
        mass = float(vehicle.mass)
        numoftrainsets = float(vehicle.numoftrainsets)
        curve_list: list[NDArray[np.float64]] = []
        for i in range(len(begins)):
            begin = float(begins[i])
            pos_desc = np.arange(begin, 0.0, -ds)
            slopes = _get_slope(pos_desc, self.track.slopes, self.track.slope_intervals)
            speed_limits = _get_speed_limit(
                pos_desc, self.track.speed_limits, self.track.speed_limit_intervals
            )
            pos_arr, speed_arr = _build_curve_backward_with_truncate_numba(
                begin=begin,
                ds=ds,
                slopes=slopes,
                speed_limits=speed_limits,
                mass=mass,
                numoftrainsets=numoftrainsets,
                dec_mode=dec_mode,
            )
            curve_list.append(np.stack([pos_arr, speed_arr], axis=0, dtype=np.float64))
        return curve_list

    def _calculate_curves_backward_without_truncate(
        self,
        begins: NDArray[np.floating],
        ds: float,
        *,
        dec_mode: int,
        vehicle: Vehicle,
    ) -> list[NDArray[np.float64]]:
        mass = float(vehicle.mass)
        numoftrainsets = float(vehicle.numoftrainsets)
        curve_list: list[NDArray[np.float64]] = []
        for i in range(len(begins)):
            begin = float(begins[i])
            pos_desc = np.arange(begin, 0.0, -ds)
            slopes = _get_slope(pos_desc, self.track.slopes, self.track.slope_intervals)
            pos_arr, speed_arr = _build_curve_backward_without_truncate_numba(
                begin=begin,
                ds=ds,
                slopes=slopes,
                mass=mass,
                numoftrainsets=numoftrainsets,
                dec_mode=dec_mode,
            )
            curve_list.append(np.stack([pos_arr, speed_arr], axis=0, dtype=np.float64))
        return curve_list

    def _get_deccelerate_for_min_curves(
        self, speed: float | NDArray[np.floating], vehicle: Vehicle
    ):
        speed_km = 3.6 * np.asarray(speed, dtype=np.float64)
        dec = np.piecewise(
            speed_km,
            [(speed_km >= 0) & (speed_km <= 100), speed_km > 100],
            [
                lambda s: (vehicle.max_dec_abs - 0.1) / 100 * s + 0.1,
                lambda s: vehicle.max_dec_abs,
            ],
        )
        return dec

    def _truncate_curve_by_speed_limit(
        self, curve_pos_arr: NDArray[np.floating], curve_speed_arr: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        speed_limits = _get_speed_limit(
            curve_pos_arr,
            self.track.speed_limits,
            self.track.speed_limit_intervals,
        )
        exceed_mask = curve_speed_arr >= speed_limits
        exceed_indices = np.where(exceed_mask)[0]
        if len(exceed_indices) == 0:
            return curve_pos_arr, curve_speed_arr
        truncate_idx = exceed_indices[-1]
        return curve_pos_arr[truncate_idx + 1 :], curve_speed_arr[truncate_idx + 1 :]

    @staticmethod
    def _enforce_monotone_decreasing_speed(
        curve_speed_arr: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        speed_arr = np.asarray(curve_speed_arr, dtype=np.float64)
        if speed_arr.size <= 1:
            return speed_arr
        return np.minimum.accumulate(speed_arr)

    def calc_min_curves(
        self,
        levi_curves_list: list[NDArray[np.float64]],
        vehicle: Vehicle,
        pos_error: float,
        speed_error: float,
        pos_offset: float,
        delay_time_until_DPS_done: float,
    ) -> list[NDArray[np.float64]]:
        curve_list = []
        for i in range(len(levi_curves_list)):
            levi_curve = levi_curves_list[i]
            levi_pos_arr = levi_curve[0, :]
            levi_speed_arr = levi_curve[1, :]
            min_dec_arr = self._get_deccelerate_for_min_curves(
                speed=levi_speed_arr, vehicle=vehicle
            )
            # 计算最小速度曲线
            min_speed_arr = (
                speed_error + levi_speed_arr + min_dec_arr * delay_time_until_DPS_done
            )
            min_pos_arr = (
                pos_error
                + levi_pos_arr
                + min_speed_arr * delay_time_until_DPS_done
                - 0.5 * min_dec_arr * delay_time_until_DPS_done**2
                + pos_offset
            )
            # 截断超出区间限速的部分
            min_pos_arr_truncated, min_speed_arr_truncated = (
                self._truncate_curve_by_speed_limit(
                    curve_pos_arr=min_pos_arr, curve_speed_arr=min_speed_arr
                )
            )
            min_speed_arr_truncated = self._enforce_monotone_decreasing_speed(
                min_speed_arr_truncated
            )
            # 补齐至末端速度为0，同时保持位置数组严格递增
            if min_speed_arr_truncated[-1] > 0:
                min_pos_arr_truncated = np.append(
                    min_pos_arr_truncated,
                    min_pos_arr_truncated[-1]
                    + min_speed_arr_truncated[-1] ** 2
                    / (
                        2
                        * self._get_deccelerate_for_min_curves(
                            min_speed_arr_truncated[-1],
                            vehicle,
                        )
                    ),
                )
                min_speed_arr_truncated = np.append(min_speed_arr_truncated, 0.0)
            else:
                min_speed_arr_truncated[-1] = 0.0
            curve_list.append(
                np.stack(
                    [min_pos_arr_truncated, min_speed_arr_truncated],
                    axis=0,
                    dtype=np.float64,
                )
            )
        return curve_list

    def calc_max_curves(
        self,
        brake_curves_list: list[NDArray[np.float64]],
        vehicle: Vehicle,
        pos_error: float,
        speed_error: float,
        delay_time_until_DPS_done: float,
        delay_time_until_VB_begin: float,
    ) -> list[NDArray[np.float64]]:
        curve_list = []
        for i in range(len(brake_curves_list)):
            brake_speed_arr = brake_curves_list[i][1, :]
            brake_pos_arr = brake_curves_list[i][0, :]
            max_acc = vehicle.max_acc
            levi_dec_arr = _calc_levi_deceleration(
                mass=vehicle.mass,
                numoftrainsets=vehicle.numoftrainsets,
                speed=brake_speed_arr,
                slope=_get_slope(
                    brake_pos_arr,
                    self.track.slopes,
                    self.track.slope_intervals,
                ),
            )
            max_speed_arr = (
                speed_error
                + brake_speed_arr
                - max_acc * delay_time_until_DPS_done
                - levi_dec_arr * delay_time_until_VB_begin
            )
            max_pos_arr = (
                pos_error
                + brake_pos_arr
                - (
                    brake_speed_arr
                    * (delay_time_until_DPS_done + delay_time_until_VB_begin)
                    + max_acc * delay_time_until_DPS_done * delay_time_until_VB_begin
                    + 0.5 * max_acc * delay_time_until_DPS_done**2
                    - 0.5 * levi_dec_arr * delay_time_until_VB_begin**2
                )
            )

            max_pos_arr_truncated, max_speed_arr_truncated = (
                self._truncate_curve_by_speed_limit(
                    curve_pos_arr=max_pos_arr, curve_speed_arr=max_speed_arr
                )
            )
            valid_mask = max_speed_arr_truncated > 0
            if np.any(valid_mask):
                last_valid_idx = np.where(valid_mask)[0][-1]
                max_pos_arr_truncated = max_pos_arr_truncated[: last_valid_idx + 1]
                max_speed_arr_truncated = max_speed_arr_truncated[: last_valid_idx + 1]

            max_speed_arr_truncated = self._enforce_monotone_decreasing_speed(
                max_speed_arr_truncated
            )

            if max_speed_arr_truncated[-1] > 0:
                s_end = float(max_speed_arr_truncated[-1])
                p_end = float(max_pos_arr_truncated[-1])
                sl_end = float(
                    _get_slope(
                        np.array([p_end]),
                        self.track.slopes,
                        self.track.slope_intervals,
                    )[0]
                )
                b_dec = calc_brake_deceleration_scalar_numba(
                    s_end,
                    sl_end,
                    float(vehicle.mass),
                    float(vehicle.numoftrainsets),
                    0,
                )
                max_pos_arr_truncated = np.append(
                    max_pos_arr_truncated,
                    p_end + s_end**2 / (2 * b_dec),
                )
                max_speed_arr_truncated = np.append(max_speed_arr_truncated, 0.0)
            else:
                max_speed_arr_truncated[-1] = 0.0

            curve_list.append(
                np.stack([max_pos_arr_truncated, max_speed_arr_truncated], axis=0)
            )
        return curve_list

    def calc_levi_curves(
        self,
        apoffsets: NDArray[np.floating],
        vehicle: Vehicle,
        ds: float = 1.0,
    ) -> list[NDArray[np.float64]]:
        return self._calculate_curves_backward_with_truncate(
            begins=apoffsets, ds=ds, dec_mode=0, vehicle=vehicle
        )

    def calc_brake_curves(
        self,
        dpoffsets: NDArray[np.floating],
        vehicle: Vehicle,
        ds: float = 1.0,
    ) -> list[NDArray[np.float64]]:
        return self._calculate_curves_backward_with_truncate(
            begins=dpoffsets, ds=ds, dec_mode=1, vehicle=vehicle
        )

    def calc_levi_and_min_curves(
        self,
        apoffsets: NDArray[np.floating],
        vehicle: Vehicle,
        ds: float = 1.0,
        pos_error: float = 10.0,
        speed_error: float = 1.0,
        pos_offset: float = 0.0,
        delay_time_until_DPS_done: float = 0.6,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        levi_curves_list_without_truncate = (
            self._calculate_curves_backward_without_truncate(
                begins=apoffsets,
                ds=ds,
                dec_mode=0,
                vehicle=vehicle,
            )
        )
        min_curves_list = self.calc_min_curves(
            levi_curves_list=levi_curves_list_without_truncate,
            vehicle=vehicle,
            pos_error=pos_error,
            speed_error=speed_error,
            pos_offset=pos_offset,
            delay_time_until_DPS_done=delay_time_until_DPS_done,
        )
        levi_curves_list_truncated: list[NDArray[np.float64]] = []
        for i in range(len(levi_curves_list_without_truncate)):
            levi_curve = levi_curves_list_without_truncate[i]
            levi_curve_pos_arr = levi_curve[0, :]
            levi_curve_speed_arr = levi_curve[1, :]
            levi_curve_pos_arr_truncated, levi_curve_speed_arr_truncated = (
                self._truncate_curve_by_speed_limit(
                    levi_curve_pos_arr, levi_curve_speed_arr
                )
            )
            levi_curves_list_truncated.append(
                np.stack(
                    [levi_curve_pos_arr_truncated, levi_curve_speed_arr_truncated],
                    axis=0,
                    dtype=np.float64,
                )
            )

        return levi_curves_list_truncated, min_curves_list

    def calc_brake_and_max_curves(
        self,
        dpoffsets: NDArray[np.floating],
        vehicle: Vehicle,
        ds: float = 1.0,
        pos_error: float = -10.0,
        speed_error: float = -1.0,
        delay_time_until_DPS_done: float = 0.6,
        delay_time_until_VB_begin: float = 0.6,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        brake_curves_list_without_truncate = (
            self._calculate_curves_backward_without_truncate(
                begins=dpoffsets,
                ds=ds,
                dec_mode=1,
                vehicle=vehicle,
            )
        )
        max_curves_list = self.calc_max_curves(
            brake_curves_list=brake_curves_list_without_truncate,
            vehicle=vehicle,
            pos_error=pos_error,
            speed_error=speed_error,
            delay_time_until_DPS_done=delay_time_until_DPS_done,
            delay_time_until_VB_begin=delay_time_until_VB_begin,
        )
        brake_curves_list_truncated: list[NDArray[np.float64]] = []
        for i in range(len(brake_curves_list_without_truncate)):
            brake_curve = brake_curves_list_without_truncate[i]
            brake_curve_pos_arr = brake_curve[0, :]
            brake_curve_speed_arr = brake_curve[1, :]
            brake_curve_pos_arr_truncated, brake_curve_speed_arr_truncated = (
                self._truncate_curve_by_speed_limit(
                    brake_curve_pos_arr, brake_curve_speed_arr
                )
            )
            brake_curves_list_truncated.append(
                np.stack(
                    [brake_curve_pos_arr_truncated, brake_curve_speed_arr_truncated],
                    axis=0,
                    dtype=np.float64,
                )
            )

        return brake_curves_list_truncated, max_curves_list


@dataclass(frozen=True)
class SafeguardCurveConfig:
    distance_step_m: float
    mass_tonnes: float
    trainset_count: int
    max_acceleration_mps2: float
    max_deceleration_mps2: float
    position_error_m: float
    speed_error_mps: float
    traction_cutoff_delay_s: float
    vortex_brake_delay_s: float
    min_curve_position_offset_m: float


@dataclass(frozen=True)
class CalculationInputs:
    calculator: SafeGuardCurves
    vehicle: Vehicle
    accessible_points: NDArray[np.float64]
    dangerous_points: NDArray[np.float64]


def calculate_curves(
    config: SafeguardCurveConfig,
    inputs: CalculationInputs,
) -> dict[str, list[NDArray[np.float64]]]:
    levi_curves, min_curves = inputs.calculator.calc_levi_and_min_curves(
        apoffsets=inputs.accessible_points,
        vehicle=inputs.vehicle,
        ds=config.distance_step_m,
        pos_error=config.position_error_m,
        speed_error=config.speed_error_mps,
        pos_offset=config.min_curve_position_offset_m,
        delay_time_until_DPS_done=config.traction_cutoff_delay_s,
    )
    brake_curves, max_curves = inputs.calculator.calc_brake_and_max_curves(
        dpoffsets=inputs.dangerous_points,
        vehicle=inputs.vehicle,
        ds=config.distance_step_m,
        pos_error=-config.position_error_m,
        speed_error=-config.speed_error_mps,
        delay_time_until_DPS_done=config.traction_cutoff_delay_s,
        delay_time_until_VB_begin=config.vortex_brake_delay_s,
    )
    return {
        "levi_curves_list": levi_curves,
        "brake_curves_list": brake_curves,
        "min_curves_list": min_curves,
        "max_curves_list": max_curves,
    }
