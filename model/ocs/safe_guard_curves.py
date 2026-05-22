from collections.abc import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from model.track import TrackInfo, get_slope, get_speed_limit
from model.vehicle import (
    VehicleInfo,
    calc_brake_deceleration,
    calc_brake_deceleration_scalar_numba,
    calc_levi_deceleration,
    calc_levi_deceleration_scalar_numba,
)

Numeric = float | np.floating | NDArray[np.floating]
AccFunc = Callable[[Numeric, Numeric], Numeric]


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
                speed, slopes[j], mass, numoftrainsets
            )
        else:
            dec = calc_brake_deceleration_scalar_numba(
                speed, slopes[j], mass, numoftrainsets, 0
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
                speed, slopes[j - 1], mass, numoftrainsets
            )
        else:
            dec = calc_brake_deceleration_scalar_numba(
                speed, slopes[j - 1], mass, numoftrainsets, 0
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
    """
    上海磁浮示范线线路安全防护计算类
    Attributes:
        track: 轨道数据类

    Methods:
        GetsDesPoints(apoffsets, dpoffsets): 获取辅助停车区目标点位置
        CalLeviCurves(apoffsets, vehicle, ds): 计算安全悬浮曲线
        CalBrakeCurves(dpoffsets, vehicle, ds): 计算安全制动曲线
    """

    def __init__(self, track: TrackInfo):
        self.track = track

    def _calculate_curves_backward_with_truncate(
        self,
        begins: NDArray[np.floating],
        ds: float,
        dec_func: AccFunc,
        *,
        dec_mode: int | None = None,
        vehicle: VehicleInfo | None = None,
    ) -> list[NDArray[np.float64]]:
        """
        通用防护曲线计算方法
        Args:
            begins: 起点数组(apoffsets或dpoffsets)
            ds: 距离步长
            dec_func: 减速度大小函数，签名为 dec_func(speed, slope, *args)
        Returns:
            [np.ndarray[pos_array, speed_array], ...]
        """
        if dec_mode is not None and vehicle is not None:
            mass = float(vehicle.mass)
            numoftrainsets = float(vehicle.numoftrainsets)
            curve_list = []
            for i in range(len(begins)):
                begin = float(begins[i])
                pos_desc = np.arange(begin, 0.0, -ds)
                slopes = np.asarray(
                    get_slope(
                        pos_desc,
                        self.track.slopes,
                        self.track.slope_intervals,
                    ),
                    dtype=np.float64,
                )
                speed_limits = np.asarray(
                    get_speed_limit(
                        pos_desc,
                        self.track.speed_limits,
                        self.track.speed_limit_intervals,
                    ),
                    dtype=np.float64,
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
                curve_list.append(
                    np.stack([pos_arr, speed_arr], axis=0, dtype=np.float64)
                )
            return curve_list

        curve_list = []
        for i in range(len(begins)):
            speed = 0.0  # 初速度
            pos = begins[i]  # 初位置
            max_steps = int(pos // ds)
            speed_arr = np.empty(max_steps + 1)
            pos_desc = np.arange(pos, 0.0, -ds)
            slopes = get_slope(
                pos_desc,
                self.track.slopes,
                self.track.slope_intervals,
            )
            speed_limits = get_speed_limit(
                pos_desc,
                self.track.speed_limits,
                self.track.speed_limit_intervals,
            )
            idx = 0
            speed_arr[0] = 0.0
            for j in range(max_steps):
                dec = dec_func(speed, slopes[j])
                next_speed_squared = speed**2 + 2 * dec * ds
                next_speed = np.sqrt(next_speed_squared)
                if next_speed >= speed_limits[j]:
                    break
                pos = pos - ds
                speed = next_speed
                idx += 1
                speed_arr[idx] = next_speed
            pos_arr = np.arange(begins[i], pos, -ds)
            curve_list.append(
                np.stack(
                    [pos_arr[::-1], speed_arr[:idx][::-1]], axis=0, dtype=np.float64
                )
            )
        return curve_list

    def _calculate_curves_backward_without_truncate(
        self,
        begins: NDArray[np.floating],
        ds: float,
        dec_func: AccFunc,
        *,
        dec_mode: int | None = None,
        vehicle: VehicleInfo | None = None,
    ) -> list[NDArray[np.float64]]:
        """
        通用防护曲线计算方法
        Args:
            begins: 起点数组(apoffsets或dpoffsets)
            ds: 距离步长
            dec_func: 减速度大小函数，签名为 dec_func(speed, slope, *args)
        Returns:
            [np.ndarray[pos_array, speed_array], ...]
        """
        if dec_mode is not None and vehicle is not None:
            mass = float(vehicle.mass)
            numoftrainsets = float(vehicle.numoftrainsets)
            curve_list = []
            for i in range(len(begins)):
                begin = float(begins[i])
                pos_desc = np.arange(begin, 0.0, -ds)
                slopes = np.asarray(
                    get_slope(
                        pos_desc,
                        self.track.slopes,
                        self.track.slope_intervals,
                    ),
                    dtype=np.float64,
                )
                pos_arr, speed_arr = _build_curve_backward_without_truncate_numba(
                    begin=begin,
                    ds=ds,
                    slopes=slopes,
                    mass=mass,
                    numoftrainsets=numoftrainsets,
                    dec_mode=dec_mode,
                )
                curve_list.append(
                    np.stack([pos_arr, speed_arr], axis=0, dtype=np.float64)
                )
            return curve_list

        curve_list = []
        for i in range(len(begins)):
            speed = 0.0  # 初速度
            pos_arr = np.arange(begins[i], 0.0, -ds)
            speed_arr = np.zeros_like(pos_arr)
            slopes = get_slope(
                pos_arr,
                self.track.slopes,
                self.track.slope_intervals,
            )
            for j in range(1, speed_arr.shape[0]):
                dec = dec_func(speed, slopes[j - 1])
                next_speed_squared = speed**2 + 2 * dec * ds
                next_speed = np.sqrt(next_speed_squared)
                speed = next_speed
                speed_arr[j] = next_speed
            curve_list.append(
                np.stack([pos_arr[::-1], speed_arr[::-1]], axis=0, dtype=np.float64)
            )
        return curve_list

    def _get_deccelerate_for_min_curves(self, speed: Numeric, vehicle: VehicleInfo):
        """
        获得计算最小速度曲线时的减速度大小

        Args:
            speed: 速度(单位: m/s)
            vehicle: 车辆特性

        Returns:
            减速度大小
        """

        speed_km = 3.6 * np.asarray(speed, dtype=np.float64)
        dec = np.piecewise(
            speed_km,
            [(speed_km >= 0) & (speed_km <= 100), speed_km > 100],
            [
                lambda speed: (vehicle.max_dec_abs - 0.1) / 100 * speed + 0.1,
                lambda speed: vehicle.max_dec_abs,
            ],
        )

        return dec

    def _truncate_curve_by_speed_limit(
        self, curve_pos_arr: NDArray[np.floating], curve_speed_arr: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        截断曲线中超出当前区间限速的部分

        Args:
            curve_pos_arr: 截断前的曲线位置数组
            curve_speed_arr: 截断前的曲线速度数组

        Returns:
            截断后的曲线位置数组、截断后的曲线速度数组
        """

        # 获取曲线上各点对应的限速
        speed_limits = get_speed_limit(
            curve_pos_arr,
            self.track.speed_limits,
            self.track.speed_limit_intervals,
            dtype=np.float64,
        )

        # 找到最后一个超出限速的点
        exceed_mask = curve_speed_arr >= speed_limits
        exceed_indices = np.where(exceed_mask)[0]

        if len(exceed_indices) == 0:
            # 没有超出限速的点，立即返回
            return curve_pos_arr, curve_speed_arr

        # 在最后一个超出限速的点处截断
        truncate_idx = exceed_indices[-1]

        # 返回截断后的曲线
        return curve_pos_arr[truncate_idx + 1 :], curve_speed_arr[truncate_idx + 1 :]

    @staticmethod
    def _enforce_monotone_decreasing_speed(
        curve_speed_arr: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """将速度数组投影为单调不增，消除局部回升。"""

        speed_arr = np.asarray(curve_speed_arr, dtype=np.float64)
        if speed_arr.size <= 1:
            return speed_arr
        return np.minimum.accumulate(speed_arr)

    def calc_min_curves(
        self,
        levi_curves_list: list[NDArray[np.float64]],
        vehicle: VehicleInfo,
        pos_error: float = 1.0,
        speed_error: float = 0.1,
        pos_offset: float = 0.0,
        delay_time_until_DPS_done: float = 0.5,
    ) -> list[NDArray[np.float64]]:
        """
        计算最小速度曲线
        Args:
            levi_curves_list: 安全悬浮曲线列表
            vehicle: 车辆特性
            pos_error: 里程测量误差
            speed_error: 速度测量误差
            pos_offset: 最小速度曲线相对于目标点方向的位置偏移量
            delay_time_until_DPS_done: 牵引切断命令发出到完成切断延时

        Returns:
            [np.ndarray[pos_array, speed_array], ...]
        """
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
                            min_speed_arr_truncated[-1], vehicle
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
        vehicle: VehicleInfo,
        pos_error: float = -1.0,
        speed_error: float = -0.1,
        delay_time_until_DPS_done: float = 0.5,
        delay_time_until_VB_begin: float = 0.5,
    ) -> list[NDArray[np.float64]]:
        """
        计算最大速度曲线
        Args:
            brake_curves_list: 安全制动曲线列表
            vehicle: 车辆特性
            pos_error: 里程测量误差
            speed_error: 速度测量误差
            delay_time_until_DPS_done: 牵引切断命令发出到完成切断延时
            delay_time_until_VB_begin: 牵引完成切断到涡流制动启动延时

        Returns:
            [np.ndarray[pos_array, speed_array], ...]
        """
        curve_list = []
        for i in range(len(brake_curves_list)):
            brake_speed_arr = brake_curves_list[i][1, :]
            brake_pos_arr = brake_curves_list[i][0, :]
            max_acc = vehicle.max_acc
            levi_dec_arr = calc_levi_deceleration(
                mass=vehicle.mass,
                numoftrainsets=vehicle.numoftrainsets,
                speed=brake_speed_arr,
                slope=get_slope(
                    brake_pos_arr,
                    self.track.slopes,
                    self.track.slope_intervals,
                ),
            )
            # 计算最大速度曲线
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

            # 截断超出区间限速的部分
            max_pos_arr_truncated, max_speed_arr_truncated = (
                self._truncate_curve_by_speed_limit(
                    curve_pos_arr=max_pos_arr, curve_speed_arr=max_speed_arr
                )
            )
            # 截断至所有点速度都大于0
            valid_mask = max_speed_arr_truncated > 0
            if np.any(valid_mask):
                last_valid_idx = np.where(valid_mask)[0][-1]
                max_pos_arr_truncated = max_pos_arr_truncated[: last_valid_idx + 1]
                max_speed_arr_truncated = max_speed_arr_truncated[: last_valid_idx + 1]

            max_speed_arr_truncated = self._enforce_monotone_decreasing_speed(
                max_speed_arr_truncated
            )

            # 补齐至末端速度为0，同时保持位置数组严格递增
            if max_speed_arr_truncated[-1] > 0:
                max_pos_arr_truncated = np.append(
                    max_pos_arr_truncated,
                    max_pos_arr_truncated[-1]
                    + max_speed_arr_truncated[-1] ** 2
                    / (
                        2
                        * calc_brake_deceleration(
                            mass=vehicle.mass,
                            numoftrainsets=vehicle.numoftrainsets,
                            speed=max_speed_arr_truncated[-1],
                            slope=get_slope(
                                max_pos_arr_truncated[-1],
                                self.track.slopes,
                                self.track.slope_intervals,
                            ),
                            level=0,
                        )
                    ),
                )
                max_speed_arr_truncated = np.append(max_speed_arr_truncated, 0.0)
            else:
                max_speed_arr_truncated[-1] = 0.0
            curve_list.append(
                np.stack(
                    [max_pos_arr_truncated, max_speed_arr_truncated],
                    axis=0,
                    dtype=np.float64,
                )
            )
        return curve_list

    def calc_levi_curves(
        self, apoffsets: NDArray[np.floating], vehicle: VehicleInfo, ds: float = 5.0
    ):
        """
        计算安全悬浮曲线
        Args:
            apoffsets: 辅助停车区可达点位置数组
            vehicle: 车辆特性
            ds: 距离步长
        Returns:
            [np.ndarray[s_array, v_array], ...]
        """
        return self._calculate_curves_backward_with_truncate(
            apoffsets,
            ds,
            lambda speed, slope: calc_levi_deceleration(
                mass=vehicle.mass,
                numoftrainsets=vehicle.numoftrainsets,
                speed=speed,
                slope=slope,
            ),
            dec_mode=0,
            vehicle=vehicle,
        )

    def calc_levi_and_min_curves(
        self,
        apoffsets: NDArray[np.floating],
        vehicle: VehicleInfo,
        ds: float = 5.0,
        pos_error: float = 1.0,
        speed_error: float = 0.1,
        pos_offset: float = 0.0,
        delay_time_until_DPS_done: float = 0.5,
    ):
        """
        计算磁浮列车安全悬浮速度曲线和最小速度曲线

        Args:
            apoffsets: 辅助停车区可达点位置
            vehicle: 车辆特性
            ds: 距离步长
            pos_error: 里程测量误差
            speed_error: 速度测量误差
            pos_offset: 最小速度曲线相对于目标点方向的位置偏移量
            delay_time_until_DPS_done: 牵引切断命令发出到完成切断延时

        Returns:
            levi_curves_list, min_curves_list
        """
        levi_curves_list_without_truncate = (
            self._calculate_curves_backward_without_truncate(
                apoffsets,
                ds,
                lambda speed, slope: calc_levi_deceleration(
                    mass=vehicle.mass,
                    numoftrainsets=vehicle.numoftrainsets,
                    speed=speed,
                    slope=slope,
                ),
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
        levi_curves_list_truncated = []
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

    def calc_brake_curves(
        self, dpoffsets: NDArray[np.floating], vehicle: VehicleInfo, ds: float = 5.0
    ):
        """
        计算安全制动曲线

        Args:
            dpoffsets: 辅助停车区危险点位置数组
            vehicle: 车辆特性
            ds: 距离步长

        Returns:
            [np.ndarray[s_array, v_array], ...]
        """
        return self._calculate_curves_backward_with_truncate(
            dpoffsets,
            ds,
            lambda speed, slope: calc_brake_deceleration(
                mass=vehicle.mass,
                numoftrainsets=vehicle.numoftrainsets,
                speed=speed,
                slope=slope,
                level=0,
            ),
            dec_mode=1,
            vehicle=vehicle,
        )

    def calc_brake_and_max_curves(
        self,
        dpoffsets: NDArray[np.floating],
        vehicle: VehicleInfo,
        ds: float = 5.0,
        pos_error: float = -1.0,
        speed_error: float = -0.1,
        delay_time_until_DPS_done: float = 0.5,
        delay_time_until_VB_begin: float = 0.5,
    ):
        """
        计算在给定辅助停车区危险点、车辆特性、距离步长、牵引切断命令执行延时下
        磁浮列车安全制动速度曲线和最大速度曲线
        Args:
            dpoffsets: 辅助停车区危险点位置数组
            vehicle: 车辆特性
            ds: 距离步长
            pos_error: 里程测量误差
            speed_error: 速度测量误差
            delay_time_until_DPS_done: 牵引切断命令发出到完成切断延时
            delay_time_until_VB_begin: 牵引完成切断到涡流制动启动延时
        Returns:
            brake_curves_list, max_curves_list
        """
        brake_curves_list_without_truncate = (
            self._calculate_curves_backward_without_truncate(
                dpoffsets,
                ds,
                lambda speed, slope: calc_brake_deceleration(
                    mass=vehicle.mass,
                    numoftrainsets=vehicle.numoftrainsets,
                    speed=speed,
                    slope=slope,
                    level=0,
                ),
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
        brake_curves_list_truncated = []
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
