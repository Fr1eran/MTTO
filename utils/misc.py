import numpy as np
from numpy.typing import NDArray
from typing import Union, overload

Numeric = Union[float, np.number, NDArray]


@overload
def GetIntervalIndex(pos: float | np.number, interval_points: NDArray) -> np.intp: ...


@overload
def GetIntervalIndex(pos: NDArray, interval_points: NDArray) -> NDArray[np.intp]: ...


def GetIntervalIndex(
    pos: Numeric, interval_points: NDArray
) -> np.intp | NDArray[np.intp]:
    """
    获得当前位置所处区间的索引
    Args:
        pos: 当前位置
        intervals: 区间端点
    """
    pos = np.asarray(pos, dtype=np.float64)
    interval_points = np.asarray(interval_points, dtype=np.float64)
    idx = np.searchsorted(interval_points, pos, side="right") - 1
    return idx


def FindSpeedRiseEntryAndFallExit(
    speed_limits,
    interval_points,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    返回所有限速开始上升的入口点列表和所有限速下降的出口点列表。
    """
    sl = np.asarray(speed_limits)
    pts = np.asarray(interval_points)

    diff = np.diff(sl)

    # diff 长度为 len(sl)-1，对应边界索引 0..len(sl)-2，边界位置为 pts[1..-2]
    n = sl.size
    if n < 2:
        return [], []

    # 默认索引范围为全部区间 [0, n-1]
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1

    diff_indices = np.arange(n - 1)
    idx_mask = (diff_indices >= int(start_idx)) & (diff_indices < int(end_idx))

    # 上升入口点：diff > 0 且在 idx_mask 范围内 -> boundary at pts[i+1]
    inc_idxs = np.where((diff > 0) & idx_mask)[0]
    ascend_begin_pos = [(float(pts[i + 1]), float(speed_limits[i])) for i in inc_idxs]

    # 下降出口点：diff < 0 且在 idx_mask 范围内
    dec_idxs = np.where((diff < 0) & idx_mask)[0]
    descend_end_pos = [
        (float(pts[j + 1]), float(speed_limits[j + 1])) for j in dec_idxs
    ]

    return ascend_begin_pos, descend_end_pos


if __name__ == "__main__":
    speed_limits = [60, 100, 150, 120, 80]
    interval_points = [0, 100, 200, 300, 400, 500]
    asc, desc = FindSpeedRiseEntryAndFallExit(
        speed_limits=speed_limits,
        interval_points=interval_points,
        start_idx=1,
        end_idx=4,
    )
    print(f"asc: {asc}")
    print(f"desc: {desc}")
    # speed_limits = [60, 100]
    # interval_points = [0, 100, 200]
    # asc, desc = FindSpeedRiseEntryAndFallExit(
    #     speed_limits=speed_limits,
    #     interval_points=interval_points,
    #     start_idx=0,
    #     end_idx=2,
    # )
    # print(f"asc: {asc}")
    # print(f"desc: {desc}")
