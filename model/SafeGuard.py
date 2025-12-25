import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from typing import Sequence, overload, Callable, Union

from utils.misc import GetIntervalIndex
from utils.curve import (
    Padding2CurvesList,
    ConcatenateCurvesWithNaN,
    CalRegions,
    DrawRegions,
)
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile

Numeric = Union[float | np.floating | NDArray[np.floating]]
AccFunc = Callable[[Numeric, Numeric], Numeric]


class SafeGuardCurves:
    """
    上海磁浮示范线线路安全防护计算类
    Attributes:
        trackprofile: 轨道纵断面特性计算类

    Methods:
        GetsDesPoints(apoffsets, dpoffsets): 获取辅助停车区目标点位置
        CalLeviCurves(apoffsets, vehicle, ds): 计算安全悬浮曲线
        CalBrakeCurves(dpoffsets, vehicle, ds): 计算安全制动曲线
    """

    def __init__(self, trackprofile: TrackProfile):
        self.trackprofile = trackprofile

    def _calculate_curves_backward_with_truncate(
        self, begins: NDArray[np.floating], ds: float, dec_func: AccFunc
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
        curve_list = []
        for i in range(len(begins)):
            speed = 0.0  # 初速度
            pos = begins[i]  # 初位置
            max_steps = int(pos // ds)
            speed_arr = np.empty(max_steps + 1)
            slopes = self.trackprofile.GetSlope(
                pos=np.arange(pos, 0.0, -ds), interpolate=True
            )
            speed_limits = self.trackprofile.GetSpeedlimit(pos=np.arange(pos, 0.0, -ds))
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
        self, begins: NDArray[np.floating], ds: float, dec_func: AccFunc
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
        curve_list = []
        for i in range(len(begins)):
            speed = 0.0  # 初速度
            pos_arr = np.arange(begins[i], 0.0, -ds)
            speed_arr = np.zeros_like(pos_arr)
            slopes = self.trackprofile.GetSlope(pos=pos_arr, interpolate=True)
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

    def _get_deccelerate_for_min_curves(self, speed: Numeric, vehicle: Vehicle):
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
                lambda speed: (vehicle.max_dec - 0.1) / 100 * speed + 0.1,
                lambda speed: vehicle.max_dec,
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
        speed_limits = self.trackprofile.GetSpeedlimit(
            pos=curve_pos_arr, dtype=np.float64
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

    def CalcMinCurves(
        self,
        levi_curves_list: list[NDArray[np.float64]],
        vehicle: Vehicle,
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
            # 插值补齐至末端速度为0
            min_pos_arr_truncated = np.append(
                min_pos_arr_truncated,
                min_pos_arr_truncated[-1]
                - min_speed_arr_truncated[-1] ** 2
                / (
                    2
                    * self._get_deccelerate_for_min_curves(
                        min_speed_arr_truncated[-1], vehicle
                    )
                ),
            )
            min_speed_arr_truncated = np.append(min_speed_arr_truncated, 0.0)
            curve_list.append(
                np.stack(
                    [min_pos_arr_truncated, min_speed_arr_truncated],
                    axis=0,
                    dtype=np.float64,
                )
            )
        return curve_list

    def CalcMaxCurves(
        self,
        brake_curves_list: list[NDArray[np.float64]],
        vehicle: Vehicle,
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
            levi_dec_arr = VehicleDynamic.CalcLeviDec(
                vehicle=vehicle,
                speed=brake_speed_arr,
                slope=self.trackprofile.GetSlope(pos=brake_pos_arr, interpolate=True),
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

            # 插值补齐至末端速度为0
            max_pos_arr_truncated = np.append(
                max_pos_arr_truncated,
                max_pos_arr_truncated[-1]
                + max_speed_arr_truncated[-1] ** 2
                / (
                    2
                    * VehicleDynamic.CalcBrakeDec(
                        vehicle=vehicle,
                        speed=max_speed_arr_truncated[-1],
                        slope=self.trackprofile.GetSlope(
                            pos=max_pos_arr_truncated[-1], interpolate=True
                        ),
                        level=0,
                    )
                ),
            )
            max_speed_arr_truncated = np.append(max_speed_arr_truncated, 0.0)
            curve_list.append(
                np.stack(
                    [max_pos_arr_truncated, max_speed_arr_truncated],
                    axis=0,
                    dtype=np.float64,
                )
            )
        return curve_list

    def CalcLeviCurves(
        self, apoffsets: NDArray[np.floating], vehicle: Vehicle, ds: float = 5.0
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
            lambda speed, slope: VehicleDynamic.CalcLeviDec(
                vehicle=vehicle, speed=speed, slope=slope
            ),
        )

    def CalcLeviAndMinCurves(
        self,
        apoffsets: NDArray[np.floating],
        vehicle: Vehicle,
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
            ([np.ndarray[pos_array, speed_array], ...], [np.ndarray[pos_array, speed_array], ...])
        """
        levi_curves_list_without_truncate = (
            self._calculate_curves_backward_without_truncate(
                apoffsets,
                ds,
                lambda speed, slope: VehicleDynamic.CalcLeviDec(
                    vehicle=vehicle, speed=speed, slope=slope
                ),
            )
        )
        min_curves_list = self.CalcMinCurves(
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

    def CalcBrakeCurves(
        self, dpoffsets: NDArray[np.floating], vehicle: Vehicle, ds: float = 5.0
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
            lambda speed, slope: VehicleDynamic.CalcBrakeDec(
                vehicle=vehicle, speed=speed, slope=slope, level=0
            ),
        )

    def CalcBrakeAndMaxCurves(
        self,
        dpoffsets: NDArray[np.floating],
        vehicle: Vehicle,
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
            ([np.ndarray[pos_array, speed_array], ...], [np.ndarray[pos_array, speed_array], ...])
        """
        brake_curves_list_without_truncate = (
            self._calculate_curves_backward_without_truncate(
                dpoffsets,
                ds,
                lambda speed, slope: VehicleDynamic.CalcBrakeDec(
                    vehicle=vehicle, speed=speed, slope=slope, level=0
                ),
            )
        )
        max_curves_list = self.CalcMaxCurves(
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


class SafeGuardUtility:
    """
    通用速度防护类
    Attributes:
        speed_limits : 限速值
        speed_limit_intervals : 限速区间
        min_curves_list : 最小速度曲线集合
        max_curves_list : 最大速度曲线集合
        gamma : 限速因子
    Methods:
        DetectDanger() : 检查速度是否超出限速或落入危险速度域
    """

    def __init__(
        self,
        *,
        speed_limits: Sequence[float] | NDArray[np.floating],
        speed_limit_intervals: Sequence[float] | NDArray[np.floating],
        min_curves_list: list[NDArray],
        max_curves_list: list[NDArray],
        gamma: float,
    ):
        self.speed_limits = np.asarray(speed_limits)
        self.speed_limit_intervals = np.asarray(speed_limit_intervals)
        self.min_curves_list = min_curves_list
        self.max_curves_list = max_curves_list

        # 计算危险交叉点和危险交叉点之后的部分防护曲线
        idp_points, self.min_curves_part_list, self.max_curves_part_list = CalRegions(
            min_curves_list, max_curves_list[:-1]
        )
        self.idp_points_x = idp_points[0, :]
        self.idp_points_y = idp_points[1, :]

        self.min_curves_part_list_padded, self.max_curves_part_list_padded = (
            Padding2CurvesList(self.min_curves_part_list, self.max_curves_part_list)
        )
        self.min_curves_part_x_padded = [
            curve[0, :] for curve in self.min_curves_part_list_padded
        ]
        self.min_curves_part_y_padded = [
            curve[1, :] for curve in self.min_curves_part_list_padded
        ]
        self.max_curves_part_x_padded = [
            curve[0, :] for curve in self.max_curves_part_list_padded
        ]
        self.max_curves_part_y_padded = [
            curve[1, :] for curve in self.max_curves_part_list_padded
        ]
        self.numofregions = idp_points.shape[1]
        self.gamma = gamma

    # def GetMinAndMaxSpeed(
    #     self, current_pos: float, current_speed: float, current_sp: int | None
    # ):
    #     """
    #     获得当前位置的最小防护速度和最大仿防护速度

    #     Args:
    #         current_pos: 当前位置
    #         current_speed: 当前速度
    #         current_sp: 当前步进的停车点编号, 从0开始

    #     Returns:
    #         min_speed, max_speed, current_sp
    #     """
    #     if current_sp is None:
    #         min_speed = 0.0
    #         current_max_curve = self.max_curves_list[0]
    #         if current_pos > current_max_curve[0, 0]:
    #             max_speed = np.interp(
    #                 current_pos, current_max_curve[0, :], current_max_curve[1, :]
    #             )
    #         else:
    #             max_speed = self.speed_limits[
    #                 np.clip(
    #                     GetIntervalIndex(current_pos, self.speed_limit_intervals),
    #                     0,
    #                     len(self.speed_limits) - 1,
    #                 )
    #             ]
    #         next_min_curve = self.min_curves_list[0]
    #         if current_speed > np.interp(
    #             current_pos, next_min_curve[0, :], next_min_curve[1, :]
    #         ):
    #             current_sp = 0
    #     else:
    #         current_min_curve = self.min_curves_list[current_sp]
    #         current_max_curve = self.max_curves_list[current_sp + 1]
    #         if current_pos < current_min_curve[0, -1]:
    #             min_speed = np.interp(
    #                 current_pos, current_min_curve[0, :], current_min_curve[1, :]
    #             )
    #         else:
    #             min_speed = 0.0
    #         if current_pos > current_max_curve[0, 0]:
    #             max_speed = np.interp(
    #                 current_pos, current_max_curve[0, :], current_max_curve[1, :]
    #             )
    #         else:
    #             max_speed = self.speed_limits[
    #                 np.clip(
    #                     GetIntervalIndex(current_pos, self.speed_limit_intervals),
    #                     0,
    #                     len(self.speed_limits) - 1,
    #                 )
    #             ]
    #         if (current_sp + 1) != len(self.min_curves_list):
    #             next_min_curve = self.min_curves_list[current_sp]
    #             if current_speed > np.interp(
    #                 current_pos, next_min_curve[0, :], next_min_curve[1, :]
    #             ):
    #                 current_sp += 1

    #     return min_speed, max_speed, current_sp

    def GetMinAndMaxSpeed(
        self, current_pos: float, current_speed: float, current_sp: int | None
    ) -> tuple[float, float, int]:
        """
        获得当前位置的最小防护速度和最大防护速度

        Args:
            current_pos: 当前位置
            current_speed: 当前速度
            current_sp: 当前目标停车点编号, 从 -1 开始

        Returns:
            tuple(current_min_speed, current_max_speed, current_sp)
        """
        current_min_speed = 0.0
        current_max_speed = 0.0
        if current_sp is None:
            # 没有给定当前目标停车点编号，需要遍历确定
            # 设置初始停车点编号为-1
            current_sp = -1
            # 遍历所有停车点对应的最小速度曲线
            for current_min_curve in self.min_curves_list:
                if current_pos > current_min_curve[0, -1]:
                    # 当前位置大于最小速度曲线的右端点
                    # 设置最小速度为0
                    current_min_speed = 0.0
                else:
                    # 当前位置小于最小速度曲线的右端点
                    # 设置最小速度为最小速度曲线在当前位置的插值
                    min_speed = np.interp(
                        current_pos, current_min_curve[0, :], current_min_curve[1, :]
                    )
                    # 未步进到当前停车点
                    if current_speed < min_speed:
                        break
                    else:
                        current_min_speed = min_speed
                current_sp += 1
            current_max_curve = self.max_curves_list[current_sp + 1]
            if current_pos > current_max_curve[0, 0]:
                # 当前位置大于最大速度曲线的左端点
                # 设置最大速度为最大速度曲线在当前位置的插值
                max_speed = np.interp(
                    current_pos, current_max_curve[0, :], current_max_curve[1, :]
                )
            else:
                # 当前位置小于最大速度曲线的左端点
                # 设置最大速度为当前位置区间限速
                max_speed = self.speed_limits[
                    np.clip(
                        GetIntervalIndex(current_pos, self.speed_limit_intervals),
                        0,
                        len(self.speed_limits) - 1,
                    )
                ]
            current_max_speed = max_speed
        else:
            # 已给定当前目标停车点编号
            if current_sp == -1:
                # 还在加速区，尚未步进到第一个辅助停车区
                min_speed = 0.0
            else:
                # 已开始停车点步进
                current_min_curve = self.min_curves_list[current_sp]
                if current_pos > current_min_curve[0, -1]:
                    # 当前位置大于最小速度曲线的右端点
                    # 设置最小速度为0
                    min_speed = 0.0
                else:
                    # 当前位置小于最小速度曲线的右端点
                    # 设置最小速度为最小速度曲线在当前位置的插值
                    min_speed = np.interp(
                        current_pos, current_min_curve[0, :], current_min_curve[1, :]
                    )
            current_min_speed = min_speed
            current_max_curve = self.max_curves_list[current_sp + 1]
            if current_pos > current_max_curve[0, 0]:
                # 当前位置大于最大速度曲线的左端点
                # 设置最大速度为最大速度曲线在当前位置的插值
                max_speed = max(
                    0.0,
                    np.interp(
                        current_pos, current_max_curve[0, :], current_max_curve[1, :]
                    ),
                )
            else:
                # 当前位置小于最大速度曲线的左端点
                # 设置最大速度为当前位置区间限速
                max_speed = self.speed_limits[
                    np.clip(
                        GetIntervalIndex(current_pos, self.speed_limit_intervals),
                        0,
                        len(self.speed_limits) - 1,
                    )
                ]
            current_max_speed = max_speed
            next_sp = current_sp + 1
            if next_sp != len(self.min_curves_list):
                # 未步进到最后一个停车点
                # 检查是否能步进到下一个停车点
                next_min_curve = self.min_curves_list[next_sp]
                if current_speed > np.interp(
                    current_pos, next_min_curve[0, :], next_min_curve[1, :]
                ):
                    # 可以步进到下一个停车点，且立即执行
                    current_sp = next_sp

        return float(current_min_speed), float(current_max_speed), current_sp

    @overload
    def DetectDanger(
        self, pos: float | np.number, speed: float | np.number
    ) -> np.bool_: ...

    @overload
    def DetectDanger(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ) -> NDArray[np.bool_]: ...

    def DetectDanger(
        self,
        pos: float | np.number | NDArray[np.floating],
        speed: float | np.number | NDArray[np.floating],
    ) -> np.bool_ | NDArray[np.bool_]:
        """
        检查速度是否超出限速或落入危险速度域

        Args:
            pos : 磁浮列车当前位置, 单位: m
            speed : 磁浮列车当前速度, 单位: m/s

        Returns:
            当前磁浮列车状态是否危险
        """
        result1 = self._DetectSpeedExceed(pos, speed)
        result2 = self._DetectDangerousRegionEnter(pos, speed)
        return result1 | result2  # type: ignore

    def _DetectSpeedExceed(
        self, pos: float | np.number | NDArray, speed: float | np.number | NDArray
    ) -> np.bool_ | NDArray[np.bool_]:
        speed_limit = self.speed_limits[
            np.clip(
                GetIntervalIndex(pos, self.speed_limit_intervals),
                0,
                len(self.speed_limits) - 1,
            )
        ]
        return speed >= speed_limit * self.gamma

    def _DetectDangerousRegionEnter(
        self, pos: float | np.number | NDArray, speed: float | np.number | NDArray
    ) -> bool | NDArray[np.bool_]:
        result = np.zeros_like(pos, dtype=bool)
        for i in range(self.numofregions):
            # 区间判断
            mask = (pos > self.idp_points_x[i]) & (
                pos < self.min_curves_part_x_padded[i][-1]
            )
            # 上下界插值
            above_v = np.interp(
                pos,  # type: ignore[arg-type]
                self.min_curves_part_x_padded[i],
                self.min_curves_part_y_padded[i],
            )
            below_v = np.interp(
                pos,  # type: ignore[arg-type]
                self.max_curves_part_x_padded[i],
                self.max_curves_part_y_padded[i],
            )
            # 速度判断
            mask = mask & (speed <= above_v) & (speed >= below_v)
            result = result | mask
        if np.isscalar(pos):
            return bool(result)
        else:
            return result

    def Render(self, ax: Axes) -> None:
        """绘制区间限速、危险交叉点、部分安全防护曲线和围成的危险速度域"""
        ax.step(
            self.speed_limit_intervals[:-1],
            self.speed_limits * 3.6,
            where="post",
            color="red",
            linestyle="dashdot",
            label="区间限速",
        )
        min_curves_parts_pos_con, min_curves_parts_speed_con = ConcatenateCurvesWithNaN(
            self.min_curves_part_list
        )
        max_curves_parts_pos_con, max_curves_parts_speed_con = ConcatenateCurvesWithNaN(
            self.max_curves_part_list
        )
        ax.plot(
            min_curves_parts_pos_con,
            min_curves_parts_speed_con * 3.6,
            label="最小速度曲线",
            color="blue",
            alpha=0.7,
        )
        ax.plot(
            max_curves_parts_pos_con,
            max_curves_parts_speed_con * 3.6,
            label="最大速度曲线",
            color="red",
            alpha=0.7,
        )
        DrawRegions(
            ax=ax,
            above_curves_list=self.min_curves_part_list_padded,
            below_curves_list=self.max_curves_part_list_padded,
            label="危险速度域",
            color="red",
            alpha=0.5,
        )
        # 绘制危险交叉点
        ax.scatter(
            x=self.idp_points_x,
            y=self.idp_points_y * 3.6,
            color="black",
            label="危险交叉点",
            linewidths=0.5,
        )
