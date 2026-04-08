import numpy as np
from numpy.typing import NDArray
from typing import Union, overload


Numeric = Union[float, np.number, NDArray]


@overload
def get_interval_index(pos: float | np.number, interval_points: NDArray) -> np.intp: ...


@overload
def get_interval_index(pos: NDArray, interval_points: NDArray) -> NDArray[np.intp]: ...


def get_interval_index(
    pos: Numeric, interval_points: NDArray
) -> np.intp | NDArray[np.intp]:
    """Return the index of the interval containing the given position(s)."""
    pos = np.asarray(pos, dtype=np.float64)
    interval_points = np.asarray(interval_points, dtype=np.float64)
    idx = np.searchsorted(interval_points, pos, side="right") - 1
    return idx


def find_speed_rise_entry_and_fall(
    speed_limits,
    interval_points,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return speed-rise entry points and speed-fall exit points."""
    sl = np.asarray(speed_limits)
    pts = np.asarray(interval_points)

    diff = np.diff(sl)

    n = sl.size
    if n < 2:
        return [], []

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1

    diff_indices = np.arange(n - 1)
    idx_mask = (diff_indices >= int(start_idx)) & (diff_indices < int(end_idx))

    inc_idxs = np.where((diff > 0) & idx_mask)[0]
    ascend_begin_pos = [(float(pts[i + 1]), float(speed_limits[i])) for i in inc_idxs]

    dec_idxs = np.where((diff < 0) & idx_mask)[0]
    descend_end_pos = [
        (float(pts[j + 1]), float(speed_limits[j + 1])) for j in dec_idxs
    ]

    return ascend_begin_pos, descend_end_pos
