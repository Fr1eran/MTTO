import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing import Union, overload
from scipy.interpolate import CubicSpline
from dataclasses import dataclass

from utils.misc import GetIntervalIndex


Numeric = Union[float, np.floating, NDArray[np.floating]]


@dataclass
class Track:
    slopes: NDArray[np.float64]  # 单位: %
    slope_intervals: NDArray[np.float64]  # 单位: m
    speed_limits: NDArray[np.float64]  # 单位: m/s
    speed_limit_intervals: NDArray[np.float64]  # 单位: m

    def __post_init__(self):
        self.slopes = np.asarray(self.slopes, dtype=np.float64)
        self.slope_intervals = np.asarray(self.slope_intervals, dtype=np.float64)
        self.speed_limits = np.asarray(self.speed_limits, dtype=np.float64)
        self.speed_limit_intervals = np.asarray(
            self.speed_limit_intervals, dtype=np.float64
        )


class TrackProfile:
    """
    上海磁浮示范线轨道纵断面特性类
    Methods:
        GetSlope: 获取当前位置的梯度
        GetNextSlopeAndDistance: 获取下个坡度变化区间的坡度和与当前位置的距离
        GetSpeedlimit: 获取当前位置的限速
        GetNextSpeedlimitAndDistance: 获取下个限速变化区间的限速和与当前位置的距离
    """

    def __init__(
        self,
        track: Track,
        *,
        dtype: DTypeLike = np.float64,
    ):
        # 将输入数据转换为指定的 numpy dtype（默认 float32）以节省内存并满足调用方要求
        self.slopes = np.asarray(track.slopes, dtype=dtype)
        self.slope_intervals = np.asarray(track.slope_intervals, dtype=dtype)
        self.speed_limits = np.asarray(track.speed_limits, dtype=dtype)
        self.speed_limit_intervals = np.asarray(
            track.speed_limit_intervals, dtype=dtype
        )
        self.z = np.zeros_like(self.slope_intervals, dtype=dtype)
        # 根据坡度计算高程并用三次样条曲线插值
        for i in range(len(self.slope_intervals) - 1):
            ds = self.slope_intervals[i + 1] - self.slope_intervals[i]
            self.z[i + 1] = self.z[i] + (self.slopes[i] / 100.0) * ds
        self.slope_spline = CubicSpline(
            self.slope_intervals, self.z, bc_type="not-a-knot"
        )

    @overload
    def GetSlope(
        self,
        pos: float | np.floating,
        interpolate: bool = False,
        *,
        dtype: DTypeLike = np.float32,
    ) -> np.floating: ...

    @overload
    def GetSlope(
        self,
        pos: NDArray[np.floating],
        interpolate: bool = False,
        *,
        dtype: DTypeLike = np.float32,
    ) -> NDArray[np.floating]: ...

    def GetSlope(
        self,
        pos: Numeric,
        interpolate: bool = False,
        *,
        dtype: DTypeLike = np.float32,
    ) -> np.floating | NDArray[np.floating]:
        """
        根据当前位置计算相应的坡度值。

        Args:
            pos: 当前位置
            interpolate: 若为True，则使用插值法计算坡度；否则，根据位置查找所在区间的坡度
        Returns:
            当前位置对应的坡度（百分位）
        """
        if interpolate:
            # 坡度=高程对位置的导数*100
            slope = self.slope_spline.derivative()(pos) * 100.0
        else:
            slope = self.slopes[
                np.clip(
                    GetIntervalIndex(pos, self.slope_intervals), 0, len(self.slopes) - 1
                )
            ]
        return slope.astype(dtype=dtype)

    @overload
    def GetNextSlopeAndDistance(
        self, pos: float | np.floating, direction: int, *, dtype: DTypeLike = np.float32
    ) -> tuple[np.floating, np.floating]: ...

    @overload
    def GetNextSlopeAndDistance(
        self,
        pos: NDArray[np.floating],
        direction: int,
        *,
        dtype: DTypeLike = np.float32,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

    def GetNextSlopeAndDistance(
        self, pos: Numeric, direction: int, *, dtype: DTypeLike = np.float32
    ) -> (
        tuple[np.floating, np.floating]
        | tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        """
        根据当前位置和运动方向返回当前坡度和下一区间坡度的变化量和当前位置与下一坡度区间的距离

        Args:
            pos: 当前位置
            direction: 运动方向

        Returns:
            下个坡度区间的坡度(百分位)，当前位置与下一坡度区间的距离(m)
        """
        current_interval_index = GetIntervalIndex(pos, self.slope_intervals)
        next_interval_index = np.clip(
            current_interval_index + direction, 0, len(self.slopes) - 1
        )

        next_slope = self.slopes[next_interval_index]
        distance = (
            self.slope_intervals[
                np.clip(
                    current_interval_index + (direction + 1) // 2,
                    0,
                    len(self.slope_intervals) - 1,
                )
            ]
            - pos
        )

        return next_slope.astype(dtype=dtype), distance.astype(dtype=dtype)

    @overload
    def GetSpeedlimit(
        self, pos: float | np.floating, *, dtype: DTypeLike = np.float32
    ) -> np.floating: ...

    @overload
    def GetSpeedlimit(
        self, pos: NDArray[np.floating], *, dtype: DTypeLike = np.float32
    ) -> NDArray[np.floating]: ...

    def GetSpeedlimit(
        self, pos: Numeric, *, dtype: DTypeLike = np.float32
    ) -> np.floating | NDArray[np.floating]:
        """
        根据当前位置返回对应的限速值。
        Args:
            pos: 当前位置
        Returns:
            当前位置对应的限速值(m/s)
        """
        speed_limit = self.speed_limits[
            np.clip(
                GetIntervalIndex(pos, self.speed_limit_intervals),
                0,
                len(self.speed_limits) - 1,
            )
        ]
        return speed_limit.astype(dtype=dtype)

    @overload
    def GetNextSpeedlimitAndDistance(
        self, pos: float | np.floating, direction: int, *, dtype: DTypeLike = np.float32
    ) -> tuple[np.floating, np.floating]: ...

    @overload
    def GetNextSpeedlimitAndDistance(
        self,
        pos: NDArray[np.floating],
        direction: int,
        *,
        dtype: DTypeLike = np.float32,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

    def GetNextSpeedlimitAndDistance(
        self, pos: Numeric, direction: int, *, dtype: DTypeLike = np.float32
    ) -> (
        tuple[np.floating, np.floating]
        | tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        """
        根据当前位置和运动方向返回对应的限速值变化量和当前位置与下一限速区间的距离

        Args:
            pos: 当前位置
            direction: 运动方向

        Returns:
            下个限速区间的限速值(m/s)，当前位置与下一限速区间的距离(m)
        """
        current_interval_index = GetIntervalIndex(pos, self.speed_limit_intervals)
        next_interval_index = np.clip(
            current_interval_index + direction, 0, len(self.speed_limits) - 1
        )

        next_speed_limit = self.speed_limits[next_interval_index]

        distance = (
            self.speed_limit_intervals[
                np.clip(
                    current_interval_index + (direction + 1) // 2,
                    0,
                    len(self.speed_limit_intervals) - 1,
                )
            ]
            - pos
        )

        return next_speed_limit.astype(dtype=dtype), distance.astype(dtype=dtype)
