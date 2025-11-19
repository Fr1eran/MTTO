import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from typing import Sequence, overload

from utils.misc import GetIntervalIndex
from utils.curve import Padding2CurvesList, ConcatenateCurvesWithNaN, DrawRegion
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile


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

    def _calculate_curves_backward(
        self, begins: NDArray[np.floating], ds: float, acc_func
    ) -> list[NDArray]:
        """
        通用防护曲线计算方法
        Args:
            begins: 起点数组（apoffsets或dpoffsets）
            ds: 步长
            acc_func: 加速度函数，签名为 acc_func(v, slope, *args)
        Returns:
            [np.ndarray[s_array, v_array], ...]
        """
        curve_list = []
        for i in range(len(begins)):
            v = 0.0  # 初速度
            s = begins[i]  # 初位置
            max_steps = int(s // ds)
            v_arr = np.empty(max_steps + 1)
            slopes = self.trackprofile.GetSlope(
                pos=np.arange(s, 0.0, -ds), interpolate=True
            )
            speed_limits = self.trackprofile.GetSpeedlimit(pos=np.arange(s, 0.0, -ds))
            idx = 0
            v_arr[0] = 0.0
            for j in range(max_steps):
                a = acc_func(v, slopes[j])  # type: ignore
                next_speed_2 = v**2 + 2 * a * ds
                next_speed = np.sqrt(next_speed_2)
                if next_speed >= speed_limits[j]:  # type: ignore
                    break
                s = s - ds
                v = next_speed
                idx += 1
                v_arr[idx] = next_speed
            s_list = np.arange(begins[i], s, -ds)
            curve_list.append(np.stack([s_list[::-1], v_arr[:idx][::-1]], axis=0))
        return curve_list

    def CalcLeviCurves(
        self, apoffsets: NDArray[np.floating], vehicle: Vehicle, ds: float
    ):
        """
        根据车辆计算辅助停车区安全悬浮曲线
        """
        return self._calculate_curves_backward(
            apoffsets,
            ds,
            lambda v, slope: VehicleDynamic.CalcLeviDacc(vehicle, v, slope),
        )

    def CalcBrakeCurves(
        self, dpoffsets: NDArray[np.floating], vehicle: Vehicle, ds: float
    ):
        """
        根据车辆计算辅助停车区安全制动曲线
        """
        return self._calculate_curves_backward(
            dpoffsets,
            ds,
            lambda v, slope: VehicleDynamic.CalcBrakeDacc(vehicle, v, slope, 0),
        )


class SafeGuardUtility:
    """
    通用速度防护类
    Attributes:
        speed_limits : 限速值
        speed_limit_intervals : 限速区间
        idp_points_x : 危险交叉点横坐标
        idp_points_y : 危险交叉点纵坐标
        levi_curves_part_list : 部分安全悬浮曲线集合
        brake_curves_part_list : 部分安全制动曲线集合
        numofregions : 危险速度域个数
        gamma : 限速因子
    Methods:
        DetectDanger() : 检查速度是否超出限速或落入危险速度域
    """

    def __init__(
        self,
        *,
        speed_limits: Sequence[float] | NDArray[np.floating],
        speed_limit_intervals: Sequence[float] | NDArray[np.floating],
        idp_points: NDArray[np.floating],
        levi_curves_part_list: list[NDArray[np.floating]],
        brake_curves_part_list: list[NDArray[np.floating]],
        gamma: float,
    ):
        self.speed_limits = np.asarray(speed_limits)
        self.speed_limit_intervals = np.asarray(speed_limit_intervals)
        self.idp_points_x = idp_points[0, :]
        self.idp_points_y = idp_points[1, :]
        self.levi_curves_part_list = levi_curves_part_list
        self.brake_curves_part_list = brake_curves_part_list
        self.levi_curves_part_list_padded, self.brake_curves_part_list_padded = (
            Padding2CurvesList(self.levi_curves_part_list, self.brake_curves_part_list)
        )
        self.levi_curves_part_x_padded = [
            curve[0, :] for curve in self.levi_curves_part_list_padded
        ]
        self.levi_curves_part_y_padded = [
            curve[1, :] for curve in self.levi_curves_part_list_padded
        ]
        self.brake_curves_part_x_padded = [
            curve[0, :] for curve in self.brake_curves_part_list_padded
        ]
        self.brake_curves_part_y_padded = [
            curve[1, :] for curve in self.brake_curves_part_list_padded
        ]
        self.numofregions = idp_points.shape[1]
        self.gamma = gamma

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
        """检查速度是否超出限速或落入危险速度域
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
                pos < self.levi_curves_part_x_padded[i][-1]
            )
            # 上下界插值
            above_v = np.interp(
                pos,  # type: ignore[arg-type]
                self.levi_curves_part_x_padded[i],
                self.levi_curves_part_y_padded[i],
            )
            below_v = np.interp(
                pos,  # type: ignore[arg-type]
                self.brake_curves_part_x_padded[i],
                self.brake_curves_part_y_padded[i],
            )
            # 速度判断
            mask = mask & (speed <= above_v) & (speed >= below_v)
            result = result | mask
        if np.isscalar(pos):
            return bool(result)
        else:
            return result

    def render(self, ax: Axes) -> None:
        """绘制区间限速、危险交叉点、部分安全防护曲线和围成的危险速度域"""
        ax.step(
            self.speed_limit_intervals[:-1],
            self.speed_limits * 3.6,
            where="post",
            color="red",
            linestyle="dashdot",
            label="区间限速",
        )
        levi_curves_parts_pos_con, levi_curves_parts_speed_con = (
            ConcatenateCurvesWithNaN(self.levi_curves_part_list)
        )
        brake_curves_parts_pos_con, brake_curves_parts_speed_con = (
            ConcatenateCurvesWithNaN(self.brake_curves_part_list)
        )
        ax.plot(
            levi_curves_parts_pos_con,
            levi_curves_parts_speed_con * 3.6,
            label="安全悬浮曲线",
            color="blue",
            alpha=0.7,
        )
        ax.plot(
            brake_curves_parts_pos_con,
            brake_curves_parts_speed_con * 3.6,
            label="安全制动曲线",
            color="red",
            alpha=0.7,
        )
        DrawRegion(
            ax=ax,
            above_curves_list=self.levi_curves_part_list_padded,
            below_curves_list=self.brake_curves_part_list_padded,
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
