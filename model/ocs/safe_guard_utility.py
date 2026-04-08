import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from typing import Sequence, overload

from utils.indexing_utils import get_interval_index
from utils.curve_geometry import pad_2curve_lists, cal_regions
from utils.curve_plot import concatenate_curves_with_NaN, draw_regions


class SafeGuardUtility:
    """
    通用速度防护类
    Attributes:
        speed_limits : 限速值
        speed_limit_intervals : 限速区间
        min_curves_list : 最小速度曲线集合
        max_curves_list : 最大速度曲线集合
        factor : 限速因子

    Methods:
        GetMinAndMaxPosition() : 根据速度反查最小位置和最大位置
        DetectDanger() : 检查速度是否超出限速或落入危险速度域
    """

    def __init__(
        self,
        *,
        speed_limits: Sequence[float] | NDArray[np.floating],
        speed_limit_intervals: Sequence[float] | NDArray[np.floating],
        min_curves_list: list[NDArray],
        max_curves_list: list[NDArray],
        factor: float,
    ):
        self.speed_limits = np.asarray(speed_limits)
        self.speed_limit_intervals = np.asarray(speed_limit_intervals)
        self.min_curves_list = min_curves_list
        self.max_curves_list = max_curves_list
        self._min_curve_pos_list = [
            np.asarray(curve[0, :], dtype=np.float64) for curve in self.min_curves_list
        ]
        self._min_curve_speed_list = [
            np.asarray(curve[1, :], dtype=np.float64) for curve in self.min_curves_list
        ]
        self._max_curve_pos_list = [
            np.asarray(curve[0, :], dtype=np.float64) for curve in self.max_curves_list
        ]
        self._max_curve_speed_list = [
            np.asarray(curve[1, :], dtype=np.float64) for curve in self.max_curves_list
        ]

        # 计算危险交叉点和危险交叉点之后的部分防护曲线
        idp_points, self.min_curves_part_list, self.max_curves_part_list = cal_regions(
            min_curves_list, max_curves_list[:-1]
        )
        self.idp_points_x = idp_points[0, :]
        self.idp_points_y = idp_points[1, :]

        self.min_curves_part_list_padded, self.max_curves_part_list_padded = (
            pad_2curve_lists(self.min_curves_part_list, self.max_curves_part_list)
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
        self.gamma = factor

    def get_intersecting_dangerous_point(self) -> NDArray:
        return self.idp_points_x

    def get_current_stopping_point(
        self, current_pos: float, current_speed: float
    ) -> int:
        """
        根据当前状态获得列车的目标停车点编号

        Args:
            current_pos: 当前位置
            current_speed: 当前速度

        Returns:
            目标停车点编号

        """
        # 设置初始停车点编号为-1
        current_sp = -1
        # 遍历所有停车点对应的最小速度曲线
        for current_min_curve in self.min_curves_list:
            if current_pos <= current_min_curve[0, -1]:
                # 当前位置小于最小速度曲线的右端点
                # 设置最小速度为最小速度曲线在当前位置的插值
                min_speed = np.interp(
                    current_pos, current_min_curve[0, :], current_min_curve[1, :]
                )
                # 未步进到当前停车点
                if current_speed <= min_speed:
                    break

            current_sp += 1

        return current_sp

    def get_min_and_max_speed(
        self, current_pos: float, current_sp: int
    ) -> tuple[float, float]:
        """
        获得当前位置在目标辅助停车区下的最小防护速度和最大防护速度

        Args:
            current_pos: 当前位置
            current_sp: 当前目标停车点编号

        Returns:
            current_min_speed, current_max_speed
        """

        # 根据当前状态计算最小防护速度
        if current_sp == -1:
            # 还在加速区，尚未步进到第一个辅助停车区
            current_min_speed = 0.0
        else:
            # 已开始停车点步进
            # 根据当前状态计算最小防护速度
            current_min_curve = self.min_curves_list[current_sp]
            if current_pos > current_min_curve[0, -1]:
                # 当前位置大于最小速度曲线的右端点
                # 设置当前最小防护速度为0
                current_min_speed = 0.0
            else:
                # 当前位置小于最小速度曲线的右端点
                # 设置最小防护速度为最小速度曲线在当前位置的插值
                current_min_speed = np.interp(
                    current_pos, current_min_curve[0, :], current_min_curve[1, :]
                )

        # 根据当前状态计算最大防护速度
        current_max_curve = self.max_curves_list[current_sp + 1]
        if current_pos > current_max_curve[0, 0]:
            # 当前位置大于最大速度曲线左端点
            # 设置最大速度为最大速度曲线在当前位置的插值
            current_max_speed = max(
                0.0,
                np.interp(
                    current_pos, current_max_curve[0, :], current_max_curve[1, :]
                ),
            )
        else:
            # 当前位置小于最大速度曲线的左端点
            # 设置最大速度为当前位置区间限速
            current_max_speed = (
                self.speed_limits[
                    np.clip(
                        get_interval_index(current_pos, self.speed_limit_intervals),
                        0,
                        len(self.speed_limits) - 1,
                    )
                ]
                * self.gamma
            )

        return float(current_min_speed), float(current_max_speed)

    def get_latest_traction_and_braking_intervention_points(
        self, current_speed: float | np.number, current_sp: int
    ) -> tuple[float, float]:
        """
        根据速度反查最小位置和最大位置。

        Args:
            current_speed: 当前速度, 单位: m/s, 需大于等于0
            current_sp: 当前目标停车点编号

        Returns:
            current_min_pos, current_max_pos
        """

        current_speed_value = float(current_speed)

        if current_sp == -1:
            current_min_pos = 0.0
        else:
            # if current_sp < -1 or current_sp >= len(self._min_curve_pos_list):
            #     raise IndexError(f"current_sp {current_sp} 超出范围")
            current_min_pos = self._get_monotone_curve_position_by_speed(
                curve_pos=self._min_curve_pos_list[current_sp],
                curve_speed=self._min_curve_speed_list[current_sp],
                current_speed=current_speed_value,
            )

        # if current_sp + 1 >= len(self._max_curve_pos_list):
        #     raise IndexError(f"current_sp {current_sp} 无法映射到最大速度曲线")

        current_max_pos = self._get_monotone_curve_position_by_speed(
            curve_pos=self._max_curve_pos_list[current_sp + 1],
            curve_speed=self._max_curve_speed_list[current_sp + 1],
            current_speed=current_speed_value,
        )

        return float(current_min_pos), float(current_max_pos)

    @staticmethod
    def _get_monotone_curve_position_by_speed(
        curve_pos: NDArray[np.floating],
        curve_speed: NDArray[np.floating],
        current_speed: float,
    ) -> float:
        """根据单调递减曲线的速度值反查位置。"""

        curve_pos = np.asarray(curve_pos, dtype=np.float64)
        curve_speed = np.asarray(curve_speed, dtype=np.float64)

        if curve_pos.shape != curve_speed.shape:
            raise ValueError("curve_pos and curve_speed must have the same shape")
        if curve_pos.size == 0:
            raise ValueError("curve must contain at least one point")
        if curve_pos.size == 1:
            return float(curve_pos[0])

        if np.any(np.diff(curve_pos) <= 0.0):
            raise ValueError("curve_pos must be strictly increasing")

        target_speed = float(current_speed)
        speed_scale = max(1.0, float(np.max(np.abs(curve_speed))))
        speed_tol = np.finfo(np.float64).eps * speed_scale * 16.0
        if np.any(np.diff(curve_speed) > speed_tol):
            raise ValueError("curve_speed must be monotone decreasing")

        ascending_pos = curve_pos[::-1]
        ascending_speed = curve_speed[::-1]
        unique_speed, unique_indices = np.unique(ascending_speed, return_index=True)
        unique_pos = ascending_pos[unique_indices]

        if unique_speed.size == 1:
            return float(unique_pos[0])

        if target_speed <= unique_speed[0]:
            speed0 = unique_speed[0]
            speed1 = unique_speed[1]
            pos0 = unique_pos[0]
            pos1 = unique_pos[1]
        elif target_speed >= unique_speed[-1]:
            speed0 = unique_speed[-2]
            speed1 = unique_speed[-1]
            pos0 = unique_pos[-2]
            pos1 = unique_pos[-1]
        else:
            return float(np.interp(target_speed, unique_speed, unique_pos))

        return float(pos0 + (target_speed - speed0) * (pos1 - pos0) / (speed1 - speed0))

    @overload
    def detect_danger(
        self, pos: float | np.number, speed: float | np.number
    ) -> np.bool_: ...

    @overload
    def detect_danger(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ) -> NDArray[np.bool_]: ...

    def detect_danger(
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
        result1 = self._detect_speed_exceed(pos, speed)
        result2 = self._detect_dangerous_region_enter(pos, speed)
        return result1 | result2  # type: ignore

    def _detect_speed_exceed(
        self, pos: float | np.number | NDArray, speed: float | np.number | NDArray
    ) -> np.bool_ | NDArray[np.bool_]:
        speed_limit = self.speed_limits[
            np.clip(
                get_interval_index(pos, self.speed_limit_intervals),
                0,
                len(self.speed_limits) - 1,
            )
        ]
        return speed >= speed_limit * self.gamma

    def _detect_dangerous_region_enter(
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
        min_curves_parts_pos_con, min_curves_parts_speed_con = (
            concatenate_curves_with_NaN(self.min_curves_part_list)
        )
        max_curves_parts_pos_con, max_curves_parts_speed_con = (
            concatenate_curves_with_NaN(self.max_curves_part_list)
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
        draw_regions(
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
