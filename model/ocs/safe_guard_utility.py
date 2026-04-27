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
        levi_curves_list : 安全悬浮曲线集合
        brake_curves_list : 安全制动曲线集合
        min_curves_list : 最小速度曲线集合
        max_curves_list : 最大速度曲线集合
        factor : 限速因子

    Methods:
        get_latest_traction_and_braking_intervention_points() : 根据速度反查最小位置和最大位置
        detect_danger() : 检查速度是否超出限速或落入危险速度域
        render() : 按图层选择性绘制原有和新增防护曲线/危险点
    """

    # 默认危险域视图：使用交叉点后的局部 min/max 曲线与危险区域。
    DANGER_VIEW_LAYERS: tuple[str, ...] = (
        "speed_limit",
        "danger_region",
        "min_curve_part",
        "max_curve_part",
        "idp_points",
    )
    # 默认全量曲线视图：展示完整 levi/brake/min/max 曲线。
    FULL_CURVE_VIEW_LAYERS: tuple[str, ...] = (
        "speed_limit",
        "levi_curve_full",
        "brake_curve_full",
        "min_curve_full",
        "max_curve_full",
    )
    # 固定绘制顺序：避免图层遮挡关系因调用顺序变化而不稳定。
    _LAYER_RENDER_ORDER: tuple[str, ...] = (
        "speed_limit",
        "danger_region",
        "min_curve_part",
        "max_curve_part",
        "levi_curve_full",
        "brake_curve_full",
        "min_curve_full",
        "max_curve_full",
        "idp_points",
    )
    _REGION_RENDER_LAYERS: frozenset[str] = frozenset({
        "danger_region",
        "min_curve_part",
        "max_curve_part",
        "idp_points",
    })
    _FULL_CURVE_RENDER_LAYERS: frozenset[str] = frozenset({
        "levi_curve_full",
        "brake_curve_full",
        "min_curve_full",
        "max_curve_full",
    })
    # 同名曲线“局部段”和“全量曲线”互斥，避免语义冲突与重复绘制。
    _MUTUALLY_EXCLUSIVE_LAYER_PAIRS: tuple[tuple[str, str], ...] = (
        ("min_curve_part", "min_curve_full"),
        ("max_curve_part", "max_curve_full"),
    )
    _VALID_RENDER_LAYERS: frozenset[str] = frozenset(_LAYER_RENDER_ORDER)

    def __init__(
        self,
        *,
        speed_limits: Sequence[float] | NDArray[np.floating],
        speed_limit_intervals: Sequence[float] | NDArray[np.floating],
        levi_curves_list: list[NDArray],
        brake_curves_list: list[NDArray],
        min_curves_list: list[NDArray],
        max_curves_list: list[NDArray],
        factor: float,
    ):
        self.speed_limits = np.asarray(speed_limits, dtype=np.float64)
        self.speed_limit_intervals = np.asarray(speed_limit_intervals, dtype=np.float64)
        self.levi_curves_list = self._sanitize_curve_list(
            levi_curves_list, curve_name="levi_curves_list"
        )
        self.brake_curves_list = self._sanitize_curve_list(
            brake_curves_list, curve_name="brake_curves_list"
        )
        self.min_curves_list = self._sanitize_curve_list(
            min_curves_list, curve_name="min_curves_list"
        )
        self.max_curves_list = self._sanitize_curve_list(
            max_curves_list, curve_name="max_curves_list"
        )
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

        self.gamma = factor

        # render/detect 所需的区域缓存，首次使用时按需计算
        self._region_cache_ready = False
        self._idp_points_x = np.asarray([], dtype=np.float64)
        self._idp_points_y = np.asarray([], dtype=np.float64)
        self._min_curves_part_list: list[NDArray[np.float64]] = []
        self._max_curves_part_list: list[NDArray[np.float64]] = []
        self._min_curves_part_list_padded: list[NDArray[np.float64]] = []
        self._max_curves_part_list_padded: list[NDArray[np.float64]] = []
        self._min_curves_part_x_padded: list[NDArray[np.float64]] = []
        self._min_curves_part_y_padded: list[NDArray[np.float64]] = []
        self._max_curves_part_x_padded: list[NDArray[np.float64]] = []
        self._max_curves_part_y_padded: list[NDArray[np.float64]] = []
        self._min_curves_parts_pos_con = np.asarray([], dtype=np.float64)
        self._min_curves_parts_speed_con = np.asarray([], dtype=np.float64)
        self._max_curves_parts_pos_con = np.asarray([], dtype=np.float64)
        self._max_curves_parts_speed_con = np.asarray([], dtype=np.float64)
        self._num_regions = 0

        # 完整曲线渲染缓存，首次渲染完整曲线时按需计算
        self._full_curve_cache_ready = False
        self._levi_curves_pos_con = np.asarray([], dtype=np.float64)
        self._levi_curves_speed_con = np.asarray([], dtype=np.float64)
        self._brake_curves_pos_con = np.asarray([], dtype=np.float64)
        self._brake_curves_speed_con = np.asarray([], dtype=np.float64)
        self._min_curves_pos_con = np.asarray([], dtype=np.float64)
        self._min_curves_speed_con = np.asarray([], dtype=np.float64)
        self._max_curves_pos_con = np.asarray([], dtype=np.float64)
        self._max_curves_speed_con = np.asarray([], dtype=np.float64)

    @staticmethod
    def _sanitize_curve(curve: NDArray[np.floating]) -> NDArray[np.float64]:
        """标准化防护曲线，并将速度数组投影为单调不增。"""

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

    @classmethod
    def _sanitize_curve_list(
        cls,
        curves: Sequence[NDArray[np.floating]],
        *,
        curve_name: str,
    ) -> list[NDArray[np.float64]]:
        sanitized_curves: list[NDArray[np.float64]] = []
        for idx, curve in enumerate(curves):
            try:
                sanitized_curves.append(cls._sanitize_curve(curve))
            except ValueError as exc:
                raise ValueError(f"{curve_name}[{idx}] is invalid: {exc}") from exc
        return sanitized_curves

    @staticmethod
    def _get_speed_scale(speed_unit: str) -> float:
        if speed_unit == "km/h":
            return 3.6
        if speed_unit == "m/s":
            return 1.0
        raise ValueError("speed_unit must be either 'm/s' or 'km/h'")

    def _normalize_render_layers(self, layers: Sequence[str] | None) -> tuple[str, ...]:
        """规范化图层参数并执行合法性校验。

        规则：
            1. layers 为 None 时，使用默认危险域视图。
            2. 拒绝未知图层名。
            3. 拒绝互斥图层组合（min/max 的 part 与 full 不可并存）。
        """
        if layers is None:
            return self.DANGER_VIEW_LAYERS

        normalized_layers = tuple(layers)
        normalized_layer_set = set(normalized_layers)
        unknown_layers = [
            layer
            for layer in normalized_layers
            if layer not in self._VALID_RENDER_LAYERS
        ]
        if unknown_layers:
            raise ValueError(
                "Unknown render layers: "
                f"{unknown_layers}. Supported layers: {sorted(self._VALID_RENDER_LAYERS)}"
            )

        for layer_a, layer_b in self._MUTUALLY_EXCLUSIVE_LAYER_PAIRS:
            if layer_a in normalized_layer_set and layer_b in normalized_layer_set:
                raise ValueError(
                    f"Render layers '{layer_a}' and '{layer_b}' are mutually exclusive"
                )

        return normalized_layers

    def _ensure_region_cache(self) -> None:
        """按需构建危险域相关缓存（render/detect 共用）。"""
        if self._region_cache_ready:
            return

        idp_points, min_curves_part_list, max_curves_part_list = cal_regions(
            self.min_curves_list,
            self.max_curves_list[:-1],
        )
        self._idp_points_x = np.asarray(idp_points[0, :], dtype=np.float64)
        self._idp_points_y = np.asarray(idp_points[1, :], dtype=np.float64)

        self._min_curves_part_list = [
            np.asarray(curve, dtype=np.float64) for curve in min_curves_part_list
        ]
        self._max_curves_part_list = [
            np.asarray(curve, dtype=np.float64) for curve in max_curves_part_list
        ]

        self._min_curves_part_list_padded, self._max_curves_part_list_padded = (
            pad_2curve_lists(self._min_curves_part_list, self._max_curves_part_list)
        )
        self._min_curves_part_x_padded = [
            np.asarray(curve[0, :], dtype=np.float64)
            for curve in self._min_curves_part_list_padded
        ]
        self._min_curves_part_y_padded = [
            np.asarray(curve[1, :], dtype=np.float64)
            for curve in self._min_curves_part_list_padded
        ]
        self._max_curves_part_x_padded = [
            np.asarray(curve[0, :], dtype=np.float64)
            for curve in self._max_curves_part_list_padded
        ]
        self._max_curves_part_y_padded = [
            np.asarray(curve[1, :], dtype=np.float64)
            for curve in self._max_curves_part_list_padded
        ]
        (
            self._min_curves_parts_pos_con,
            self._min_curves_parts_speed_con,
        ) = concatenate_curves_with_NaN(self._min_curves_part_list)
        (
            self._max_curves_parts_pos_con,
            self._max_curves_parts_speed_con,
        ) = concatenate_curves_with_NaN(self._max_curves_part_list)
        self._num_regions = int(idp_points.shape[1])
        self._region_cache_ready = True

    def _ensure_full_curve_cache(self) -> None:
        """按需构建完整曲线渲染缓存。"""
        if self._full_curve_cache_ready:
            return

        self._levi_curves_pos_con, self._levi_curves_speed_con = (
            concatenate_curves_with_NaN(self.levi_curves_list)
        )
        self._brake_curves_pos_con, self._brake_curves_speed_con = (
            concatenate_curves_with_NaN(self.brake_curves_list)
        )
        self._min_curves_pos_con, self._min_curves_speed_con = (
            concatenate_curves_with_NaN(self.min_curves_list)
        )
        self._max_curves_pos_con, self._max_curves_speed_con = (
            concatenate_curves_with_NaN(self.max_curves_list)
        )
        self._full_curve_cache_ready = True

    def _plot_curve(
        self,
        ax: Axes,
        *,
        pos: NDArray[np.float64],
        speed: NDArray[np.float64],
        speed_scale: float,
        label: str,
        color: str,
        linestyle: str = "solid",
        alpha: float = 0.7,
        linewidth: float = 2.0,
    ) -> None:
        ax.plot(
            pos,
            speed * speed_scale,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
        )

    def get_intersecting_dangerous_point(self) -> NDArray:
        self._ensure_region_cache()
        return self._idp_points_x

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
        self._ensure_region_cache()

        result = np.zeros_like(pos, dtype=bool)
        for i in range(self._num_regions):
            # 区间判断
            mask = (pos > self._idp_points_x[i]) & (
                pos < self._min_curves_part_x_padded[i][-1]
            )
            # 上下界插值
            above_v = np.interp(
                pos,  # type: ignore[arg-type]
                self._min_curves_part_x_padded[i],
                self._min_curves_part_y_padded[i],
            )
            below_v = np.interp(
                pos,  # type: ignore[arg-type]
                self._max_curves_part_x_padded[i],
                self._max_curves_part_y_padded[i],
            )
            # 速度判断
            mask = mask & (speed <= above_v) & (speed >= below_v)
            result = result | mask
        if np.isscalar(pos):
            return bool(result)
        else:
            return result

    def render(
        self,
        ax: Axes,
        *,
        layers: Sequence[str] | None = None,
        speed_unit: str = "km/h",
    ) -> None:
        """按图层绘制防护曲线和危险域。

        Args:
            ax: Matplotlib 坐标轴。
            layers: 需要绘制的图层序列。
                - None 时使用 `DANGER_VIEW_LAYERS`。
                - 允许混合选择危险域图层和完整曲线图层。
                - 互斥约束：`min_curve_part` 与 `min_curve_full` 不能同时出现；
                  `max_curve_part` 与 `max_curve_full` 不能同时出现。
            speed_unit: 速度显示单位，仅支持 "m/s" 与 "km/h"。
        """
        selected_layers = self._normalize_render_layers(layers)
        if not selected_layers:
            return

        # 仅在需要时触发对应预处理缓存，避免不必要的计算开销。
        if any(layer in self._REGION_RENDER_LAYERS for layer in selected_layers):
            self._ensure_region_cache()
        if any(layer in self._FULL_CURVE_RENDER_LAYERS for layer in selected_layers):
            self._ensure_full_curve_cache()

        speed_scale = self._get_speed_scale(speed_unit)
        selected_layer_set = set(selected_layers)

        for layer in self._LAYER_RENDER_ORDER:
            if layer not in selected_layer_set:
                continue

            if layer == "speed_limit":
                ax.step(
                    self.speed_limit_intervals[:-1],
                    self.speed_limits * speed_scale,
                    where="post",
                    color="red",
                    linestyle="dashdot",
                    label="speed limit",
                    linewidth=1.5,
                )
            elif layer == "danger_region":
                draw_regions(
                    ax=ax,
                    above_curves_list=self._min_curves_part_list_padded,
                    below_curves_list=self._max_curves_part_list_padded,
                    label="dangerous speed region",
                    color="red",
                    alpha=0.5,
                )
            elif layer == "min_curve_part":
                self._plot_curve(
                    ax=ax,
                    pos=self._min_curves_parts_pos_con,
                    speed=self._min_curves_parts_speed_con,
                    speed_scale=speed_scale,
                    label="minimum speed curve",
                    color="blue",
                    linewidth=1.5,
                )
            elif layer == "max_curve_part":
                self._plot_curve(
                    ax=ax,
                    pos=self._max_curves_parts_pos_con,
                    speed=self._max_curves_parts_speed_con,
                    speed_scale=speed_scale,
                    label="maximum speed curve",
                    color="red",
                    linewidth=1.5,
                )
            elif layer == "levi_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._levi_curves_pos_con,
                    speed=self._levi_curves_speed_con,
                    speed_scale=speed_scale,
                    label="levitation safety curve",
                    color="blue",
                    linestyle="dashed",
                    linewidth=1.5,
                )
            elif layer == "brake_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._brake_curves_pos_con,
                    speed=self._brake_curves_speed_con,
                    speed_scale=speed_scale,
                    label="braking safety curve",
                    color="red",
                    linestyle="dashed",
                    linewidth=1.5,
                )
            elif layer == "min_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._min_curves_pos_con,
                    speed=self._min_curves_speed_con,
                    speed_scale=speed_scale,
                    label="minimum speed curve (full)",
                    color="blue",
                    linewidth=1.5,
                )
            elif layer == "max_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._max_curves_pos_con,
                    speed=self._max_curves_speed_con,
                    speed_scale=speed_scale,
                    label="maximum speed curve (full)",
                    color="red",
                    linewidth=1.5,
                )
            elif layer == "idp_points":
                ax.scatter(
                    x=self._idp_points_x,
                    y=self._idp_points_y * speed_scale,
                    color="black",
                    label="intersecting dangerous point",
                    linewidths=0.2,
                )
