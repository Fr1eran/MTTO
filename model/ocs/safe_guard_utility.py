from collections.abc import Sequence
from typing import overload

import numpy as np
from matplotlib.axes import Axes
from numba import (
    njit,
)
from numpy.typing import ArrayLike, NDArray

from utils.curve_geometry import cal_regions, pad_2curve_lists
from utils.curve_plot import concatenate_curves_with_NaN, draw_regions
from utils.indexing_utils import get_interval_index, get_interval_index_scalar_numba

ScalarNumeric = float | np.floating


@njit(cache=True)
def _interp_scalar_numba(
    x: float,
    xp_row: NDArray[np.float64],
    fp_row: NDArray[np.float64],
    n: int,
) -> float:
    if n <= 0:
        return 0.0
    if x <= xp_row[0]:
        return fp_row[0]
    last = n - 1
    if x >= xp_row[last]:
        return fp_row[last]

    lo = 0
    hi = last
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if x < xp_row[mid]:
            hi = mid
        else:
            lo = mid

    x0 = xp_row[lo]
    x1 = xp_row[hi]
    y0 = fp_row[lo]
    y1 = fp_row[hi]
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


@njit(cache=True)
def _get_min_speed_numba(
    current_pos: float,
    current_sp: int,
    min_pos_packed: NDArray[np.float64],
    min_speed_packed: NDArray[np.float64],
    min_lengths: NDArray[np.int32],
) -> float:
    if current_sp == -1:
        return 0.0

    curve_len = int(min_lengths[current_sp])
    if curve_len <= 0:
        return 0.0

    if current_pos > min_pos_packed[current_sp, curve_len - 1]:
        return 0.0

    return _interp_scalar_numba(
        current_pos,
        min_pos_packed[current_sp],
        min_speed_packed[current_sp],
        curve_len,
    )


@njit(cache=True)
def _get_max_speed_numba(
    current_pos: float,
    current_sp: int,
    max_pos_packed: NDArray[np.float64],
    max_speed_packed: NDArray[np.float64],
    max_lengths: NDArray[np.int32],
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    gamma: float,
) -> float:
    max_curve_idx = current_sp + 1
    max_curve_len = int(max_lengths[max_curve_idx])
    if current_pos > max_pos_packed[max_curve_idx, 0]:
        current_max_speed = _interp_scalar_numba(
            current_pos,
            max_pos_packed[max_curve_idx],
            max_speed_packed[max_curve_idx],
            max_curve_len,
        )
        if current_max_speed < 0.0:
            current_max_speed = 0.0
    else:
        idx = get_interval_index_scalar_numba(current_pos, speed_limit_intervals)
        if idx < 0:
            idx = 0
        elif idx >= speed_limits.size:
            idx = speed_limits.size - 1
        current_max_speed = speed_limits[idx] * gamma
    return current_max_speed


@njit(cache=True)
def _get_min_and_max_speed_numba(
    current_pos: float,
    current_sp: int,
    min_pos_packed: NDArray[np.float64],
    min_speed_packed: NDArray[np.float64],
    min_lengths: NDArray[np.int32],
    max_pos_packed: NDArray[np.float64],
    max_speed_packed: NDArray[np.float64],
    max_lengths: NDArray[np.int32],
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    gamma: float,
) -> tuple[float, float]:
    current_min_speed = _get_min_speed_numba(
        current_pos,
        current_sp,
        min_pos_packed,
        min_speed_packed,
        min_lengths,
    )

    current_max_speed = _get_max_speed_numba(
        current_pos,
        current_sp,
        max_pos_packed,
        max_speed_packed,
        max_lengths,
        speed_limits,
        speed_limit_intervals,
        gamma,
    )

    return current_min_speed, current_max_speed


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
        get_latest_traction_and_braking_intervention_points()
        : 根据速度反查最小位置和最大位置
        detect_danger() : 检查速度是否超出限速或落入危险速度域
        render() : 按图层选择性绘制原有和新增防护曲线/危险点
    """

    # 默认危险域视图: 使用交叉点后的局部 min/max 曲线与危险区域。
    DANGER_VIEW_LAYERS: tuple[str, ...] = (
        "speed_limit",
        "danger_region",
        "min_curve_part",
        "max_curve_part",
        # "idp_points",
    )
    # 默认全量曲线视图: 展示完整 levi/brake/min/max 曲线。
    FULL_CURVE_VIEW_LAYERS: tuple[str, ...] = (
        "speed_limit",
        "levi_curve_full",
        "brake_curve_full",
        "min_curve_full",
        "max_curve_full",
    )
    # 固定绘制顺序: 避免图层遮挡关系因调用顺序变化而不稳定。
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
    _REGION_RENDER_LAYERS: frozenset[str] = frozenset(
        {
            "danger_region",
            "min_curve_part",
            "max_curve_part",
            "idp_points",
        }
    )
    _FULL_CURVE_RENDER_LAYERS: frozenset[str] = frozenset(
        {
            "levi_curve_full",
            "brake_curve_full",
            "min_curve_full",
            "max_curve_full",
        }
    )
    # 同名曲线“局部段”和“全量曲线”互斥, 避免语义冲突与重复绘制。
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
        levi_curves_list: list[NDArray[np.float64]],
        brake_curves_list: list[NDArray[np.float64]],
        min_curves_list: list[NDArray[np.float64]],
        max_curves_list: list[NDArray[np.float64]],
        factor: float,
    ):
        self.speed_limits: NDArray[np.float64] = np.asarray(
            speed_limits, dtype=np.float64
        )
        self.speed_limit_intervals: NDArray[np.float64] = np.asarray(
            speed_limit_intervals, dtype=np.float64
        )
        self.levi_curves_list: list[NDArray[np.float64]] = self._sanitize_curve_list(
            levi_curves_list, curve_name="levi_curves_list"
        )
        self.brake_curves_list: list[NDArray[np.float64]] = self._sanitize_curve_list(
            brake_curves_list, curve_name="brake_curves_list"
        )
        self.min_curves_list: list[NDArray[np.float64]] = self._sanitize_curve_list(
            min_curves_list, curve_name="min_curves_list"
        )
        self.max_curves_list: list[NDArray[np.float64]] = self._sanitize_curve_list(
            max_curves_list, curve_name="max_curves_list"
        )
        self._min_curve_pos_list: list[NDArray[np.float64]] = [
            np.asarray(curve[0, :], dtype=np.float64) for curve in self.min_curves_list
        ]
        self._min_curve_speed_list: list[NDArray[np.float64]] = [
            np.asarray(curve[1, :], dtype=np.float64) for curve in self.min_curves_list
        ]
        self._max_curve_pos_list: list[NDArray[np.float64]] = [
            np.asarray(curve[0, :], dtype=np.float64) for curve in self.max_curves_list
        ]
        self._max_curve_speed_list: list[NDArray[np.float64]] = [
            np.asarray(curve[1, :], dtype=np.float64) for curve in self.max_curves_list
        ]

        self.gamma: float = factor
        self._speed_query_cache_ready: bool = False

        self._min_curves_pos_packed: NDArray[np.float64] = np.zeros(
            (0, 1), dtype=np.float64
        )
        self._min_curves_speed_packed: NDArray[np.float64] = np.zeros(
            (0, 1), dtype=np.float64
        )
        self._min_curves_lengths: NDArray[np.int32] = np.zeros((0,), dtype=np.int32)
        self._max_curves_pos_packed: NDArray[np.float64] = np.zeros(
            (0, 1), dtype=np.float64
        )
        self._max_curves_speed_packed: NDArray[np.float64] = np.zeros(
            (0, 1), dtype=np.float64
        )
        self._max_curves_lengths: NDArray[np.int32] = np.zeros((0,), dtype=np.int32)

        self._build_speed_query_cache()

        # render/detect 所需的区域缓存, 首次使用时按需计算
        self._region_cache_ready: bool = False
        self._idp_points_x: NDArray[np.float64] = np.asarray([], dtype=np.float64)
        self._idp_points_y: NDArray[np.float64] = np.asarray([], dtype=np.float64)
        self._min_curves_part_list: list[NDArray[np.float64]] = []
        self._max_curves_part_list: list[NDArray[np.float64]] = []
        self._min_curves_part_list_padded: list[NDArray[np.float64]] = []
        self._max_curves_part_list_padded: list[NDArray[np.float64]] = []
        self._min_curves_part_x_padded: list[NDArray[np.float64]] = []
        self._min_curves_part_y_padded: list[NDArray[np.float64]] = []
        self._max_curves_part_x_padded: list[NDArray[np.float64]] = []
        self._max_curves_part_y_padded: list[NDArray[np.float64]] = []
        self._min_curves_parts_pos_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._min_curves_parts_speed_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._max_curves_parts_pos_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._max_curves_parts_speed_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._num_regions: int = 0

        # 完整曲线渲染缓存, 首次渲染完整曲线时按需计算
        self._full_curve_cache_ready: bool = False
        self._levi_curves_pos_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._levi_curves_speed_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._brake_curves_pos_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._brake_curves_speed_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._min_curves_pos_con: NDArray[np.float64] = np.asarray([], dtype=np.float64)
        self._min_curves_speed_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )
        self._max_curves_pos_con: NDArray[np.float64] = np.asarray([], dtype=np.float64)
        self._max_curves_speed_con: NDArray[np.float64] = np.asarray(
            [], dtype=np.float64
        )

    @staticmethod
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

    def _build_speed_query_cache(self) -> None:
        min_curve_count = len(self.min_curves_list)
        max_curve_count = len(self.max_curves_list)

        min_max_len = max((curve.shape[1] for curve in self.min_curves_list), default=1)
        max_max_len = max((curve.shape[1] for curve in self.max_curves_list), default=1)

        self._min_curves_pos_packed = np.empty(
            (min_curve_count, min_max_len), dtype=np.float64
        )
        self._min_curves_speed_packed = np.empty(
            (min_curve_count, min_max_len), dtype=np.float64
        )
        self._min_curves_lengths = np.empty((min_curve_count,), dtype=np.int32)

        for idx, curve in enumerate(self.min_curves_list):
            curve_len = int(curve.shape[1])
            self._min_curves_lengths[idx] = curve_len
            self._min_curves_pos_packed[idx, :curve_len] = curve[0, :]
            self._min_curves_speed_packed[idx, :curve_len] = curve[1, :]
            if curve_len < min_max_len:
                self._min_curves_pos_packed[idx, curve_len:] = curve[0, curve_len - 1]
                self._min_curves_speed_packed[idx, curve_len:] = curve[1, curve_len - 1]

        self._max_curves_pos_packed = np.empty(
            (max_curve_count, max_max_len), dtype=np.float64
        )
        self._max_curves_speed_packed = np.empty(
            (max_curve_count, max_max_len), dtype=np.float64
        )
        self._max_curves_lengths = np.empty((max_curve_count,), dtype=np.int32)

        for idx, curve in enumerate(self.max_curves_list):
            curve_len = int(curve.shape[1])
            self._max_curves_lengths[idx] = curve_len
            self._max_curves_pos_packed[idx, :curve_len] = curve[0, :]
            self._max_curves_speed_packed[idx, :curve_len] = curve[1, :]
            if curve_len < max_max_len:
                self._max_curves_pos_packed[idx, curve_len:] = curve[0, curve_len - 1]
                self._max_curves_speed_packed[idx, curve_len:] = curve[1, curve_len - 1]

        self._speed_query_cache_ready = True

    def _ensure_speed_query_cache(self) -> None:
        if self._speed_query_cache_ready:
            return
        self._build_speed_query_cache()

    @staticmethod
    def _get_speed_scale(speed_unit: str) -> float:
        if speed_unit == "km/h":
            return 3.6
        if speed_unit == "m/s":
            return 1.0
        raise ValueError("speed_unit must be either 'm/s' or 'km/h'")

    def _normalize_render_layers(self, layers: Sequence[str] | None) -> tuple[str, ...]:
        """规范化图层参数并执行合法性校验。

        规则:
            1. layers 为 None 时, 使用默认危险域视图。
            2. 拒绝未知图层名。
            3. 拒绝互斥图层组合(min/max 的 part 与 full 不可并存)。
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
                + f"{unknown_layers}. Supported layers: \
                  {sorted(self._VALID_RENDER_LAYERS)}"
            )

        for layer_a, layer_b in self._MUTUALLY_EXCLUSIVE_LAYER_PAIRS:
            if layer_a in normalized_layer_set and layer_b in normalized_layer_set:
                raise ValueError(
                    f"Render layers '{layer_a}' and '{layer_b}' are mutually exclusive"
                )

        return normalized_layers

    def _ensure_region_cache(self) -> None:
        """按需构建危险域相关缓存(render/detect 共用)。"""
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
        _ = ax.plot(
            pos,
            speed * speed_scale,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
        )

    def get_intersecting_dangerous_point(self) -> NDArray[np.float64]:
        self._ensure_region_cache()
        return self._idp_points_x

    def get_current_stopping_point(
        self, current_pos: ScalarNumeric, current_speed: ScalarNumeric
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

    def _get_min_and_max_speed_legacy(
        self,
        current_pos: float,
        current_sp: int,
    ) -> tuple[float, float]:
        if current_sp == -1:
            current_min_speed = 0.0
        else:
            current_min_curve = self.min_curves_list[current_sp]
            if current_pos > current_min_curve[0, -1]:
                current_min_speed = 0.0
            else:
                current_min_speed = np.interp(
                    current_pos, current_min_curve[0, :], current_min_curve[1, :]
                )

        current_max_curve = self.max_curves_list[current_sp + 1]
        if current_pos > current_max_curve[0, 0]:
            current_max_speed = max(
                0.0,
                np.interp(
                    current_pos,
                    current_max_curve[0, :],
                    current_max_curve[1, :],
                ),
            )
        else:
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

    def get_min_speed(self, current_pos: ScalarNumeric, current_sp: int) -> float:
        current_pos_value = float(current_pos)
        current_sp_value = int(current_sp)
        self._ensure_speed_query_cache()
        return float(
            _get_min_speed_numba(
                current_pos_value,
                current_sp_value,
                self._min_curves_pos_packed,
                self._min_curves_speed_packed,
                self._min_curves_lengths,
            )
        )

    def get_max_speed(self, current_pos: ScalarNumeric, current_sp: int) -> float:
        current_pos_value = float(current_pos)
        current_sp_value = int(current_sp)
        self._ensure_speed_query_cache()
        return float(
            _get_max_speed_numba(
                current_pos_value,
                current_sp_value,
                self._max_curves_pos_packed,
                self._max_curves_speed_packed,
                self._max_curves_lengths,
                self.speed_limits,
                self.speed_limit_intervals,
                float(self.gamma),
            )
        )

    def get_min_and_max_speed(
        self, current_pos: ScalarNumeric, current_sp: int
    ) -> tuple[float, float]:
        """
        获得当前位置在目标辅助停车区下的最小防护速度和最大防护速度

        Args:
            current_pos: 当前位置
            current_sp: 当前目标停车点编号

        Returns:
            current_min_speed, current_max_speed
        """

        current_pos_value = float(current_pos)
        current_sp_value = int(current_sp)
        self._ensure_speed_query_cache()
        current_min_speed, current_max_speed = _get_min_and_max_speed_numba(
            current_pos_value,
            current_sp_value,
            self._min_curves_pos_packed,
            self._min_curves_speed_packed,
            self._min_curves_lengths,
            self._max_curves_pos_packed,
            self._max_curves_speed_packed,
            self._max_curves_lengths,
            self.speed_limits,
            self.speed_limit_intervals,
            float(self.gamma),
        )
        return float(current_min_speed), float(current_max_speed)

    def get_latest_traction_and_braking_intervention_points(
        self, current_speed: ScalarNumeric, current_sp: int
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
    def detect_danger(self, pos: ScalarNumeric, speed: ScalarNumeric) -> bool: ...

    @overload
    def detect_danger(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ) -> NDArray[np.bool]: ...

    def detect_danger(
        self,
        pos: ScalarNumeric | NDArray[np.floating],
        speed: ScalarNumeric | NDArray[np.floating],
    ) -> bool | NDArray[np.bool]:
        """
        检查速度是否超出限速或落入危险速度域

        Args:
            pos : 磁浮列车当前位置, 单位: m
            speed : 磁浮列车当前速度, 单位: m/s

        Returns:
            当前磁浮列车状态是否危险
        """
        pos, speed = np.broadcast_arrays(
            np.asarray(pos, dtype=np.float64),
            np.asarray(speed, dtype=np.float64),
        )
        result1 = self._detect_speed_exceed(pos, speed)
        result2 = self._detect_dangerous_region_enter(pos, speed)
        result = result1 | result2
        if result.ndim == 0:
            return bool(result)
        return result

    def detect_any_danger(
        self,
        pos: ScalarNumeric | ArrayLike,
        speed: ScalarNumeric | ArrayLike,
    ) -> bool:
        """检查输入序列中是否存在任一危险状态(早停语义)。"""
        pos_arr, speed_arr = np.broadcast_arrays(
            np.asarray(pos, dtype=np.float64),
            np.asarray(speed, dtype=np.float64),
        )

        if self._detect_speed_exceed_any(pos_arr, speed_arr):
            return True
        return self._detect_dangerous_region_enter_any(pos_arr, speed_arr)

    def _detect_speed_exceed(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ):
        speed_limit = self.speed_limits[
            np.clip(
                get_interval_index(pos, self.speed_limit_intervals),
                0,
                len(self.speed_limits) - 1,
            )
        ]
        return speed >= speed_limit * self.gamma

    def _detect_speed_exceed_any(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ) -> bool:
        if pos.ndim == 0:
            idx = int(get_interval_index(float(pos), self.speed_limit_intervals))
            idx = min(max(idx, 0), len(self.speed_limits) - 1)
            return bool(float(speed) >= self.speed_limits[idx] * self.gamma)

        speed_limit = self.speed_limits[
            np.clip(
                np.searchsorted(self.speed_limit_intervals, pos, side="right") - 1,
                0,
                len(self.speed_limits) - 1,
            )
        ]
        return bool(np.any(speed >= speed_limit * self.gamma))

    def _detect_dangerous_region_enter(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ):
        self._ensure_region_cache()

        if pos.ndim == 0:
            pos_value = float(pos)
            speed_value = float(speed)
            for i in range(self._num_regions):
                if not (
                    pos_value > self._idp_points_x[i]
                    and pos_value < self._min_curves_part_x_padded[i][-1]
                ):
                    continue
                above_v = float(
                    np.interp(
                        pos_value,
                        self._min_curves_part_x_padded[i],
                        self._min_curves_part_y_padded[i],
                    )
                )
                below_v = float(
                    np.interp(
                        pos_value,
                        self._max_curves_part_x_padded[i],
                        self._max_curves_part_y_padded[i],
                    )
                )
                return bool(speed_value <= above_v and speed_value >= below_v)
            return False

        result = np.zeros_like(pos, dtype=bool)
        for i in range(self._num_regions):
            # 区间判断
            mask = (pos > self._idp_points_x[i]) & (
                pos < self._min_curves_part_x_padded[i][-1]
            )
            if not np.any(mask):
                continue

            pos_masked = pos[mask]
            speed_masked = speed[mask]
            # 上下界插值
            above_v = np.interp(
                pos_masked,
                self._min_curves_part_x_padded[i],
                self._min_curves_part_y_padded[i],
            )
            below_v = np.interp(
                pos_masked,
                self._max_curves_part_x_padded[i],
                self._max_curves_part_y_padded[i],
            )
            # 速度判断
            result[mask] |= (speed_masked <= above_v) & (speed_masked >= below_v)
        return result

    def _detect_dangerous_region_enter_any(
        self, pos: NDArray[np.floating], speed: NDArray[np.floating]
    ) -> bool:
        self._ensure_region_cache()

        if pos.ndim == 0:
            return bool(self._detect_dangerous_region_enter(pos, speed))

        for i in range(self._num_regions):
            mask = (pos > self._idp_points_x[i]) & (
                pos < self._min_curves_part_x_padded[i][-1]
            )
            if not np.any(mask):
                continue

            pos_masked = pos[mask]
            speed_masked = speed[mask]
            above_v = np.interp(
                pos_masked,
                self._min_curves_part_x_padded[i],
                self._min_curves_part_y_padded[i],
            )
            below_v = np.interp(
                pos_masked,
                self._max_curves_part_x_padded[i],
                self._max_curves_part_y_padded[i],
            )
            if np.any((speed_masked <= above_v) & (speed_masked >= below_v)):
                return True
        return False

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
                - 互斥约束: `min_curve_part` 与 `min_curve_full` 不能同时出现;
                  `max_curve_part` 与 `max_curve_full` 不能同时出现。
            speed_unit: 速度显示单位, 仅支持 "m/s" 与 "km/h"。
        """
        selected_layers = self._normalize_render_layers(layers)
        if not selected_layers:
            return

        # 仅在需要时触发对应预处理缓存, 避免不必要的计算开销。
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
                _ = ax.step(
                    self.speed_limit_intervals[:-1],
                    self.speed_limits * speed_scale,
                    where="post",
                    color="red",
                    linestyle="dashdot",
                    label="Track speed limit",
                    linewidth=1.5,
                )
            elif layer == "danger_region":
                draw_regions(
                    ax=ax,
                    above_curves_list=self._min_curves_part_list_padded,
                    below_curves_list=self._max_curves_part_list_padded,
                    label="Dangerous speed region",
                    color="red",
                    alpha=0.5,
                )
            elif layer == "min_curve_part":
                self._plot_curve(
                    ax=ax,
                    pos=self._min_curves_parts_pos_con,
                    speed=self._min_curves_parts_speed_con,
                    speed_scale=speed_scale,
                    label="Minimum speed curve",
                    color="blue",
                    linewidth=1.2,
                )
            elif layer == "max_curve_part":
                self._plot_curve(
                    ax=ax,
                    pos=self._max_curves_parts_pos_con,
                    speed=self._max_curves_parts_speed_con,
                    speed_scale=speed_scale,
                    label="Maximum speed curve",
                    color="red",
                    linewidth=1.2,
                )
            elif layer == "levi_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._levi_curves_pos_con,
                    speed=self._levi_curves_speed_con,
                    speed_scale=speed_scale,
                    label="Safe levitation curve",
                    color="blue",
                    linestyle="dashed",
                    linewidth=1.2,
                )
            elif layer == "brake_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._brake_curves_pos_con,
                    speed=self._brake_curves_speed_con,
                    speed_scale=speed_scale,
                    label="Safe braking curve",
                    color="red",
                    linestyle="dashed",
                    linewidth=1.2,
                )
            elif layer == "min_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._min_curves_pos_con,
                    speed=self._min_curves_speed_con,
                    speed_scale=speed_scale,
                    label="Minimum speed curve",
                    color="blue",
                    linewidth=1.2,
                )
            elif layer == "max_curve_full":
                self._plot_curve(
                    ax=ax,
                    pos=self._max_curves_pos_con,
                    speed=self._max_curves_speed_con,
                    speed_scale=speed_scale,
                    label="Maximum speed curve",
                    color="red",
                    linewidth=1.2,
                )
            elif layer == "idp_points":
                _ = ax.scatter(
                    x=self._idp_points_x,
                    y=self._idp_points_y * speed_scale,
                    color="black",
                    label="Intersecting dangerous point",
                    linewidths=0.2,
                )
