from typing import NamedTuple, TypedDict

import numpy as np
from numba import (
    njit,
)
from numpy.typing import NDArray

from model.common.energy_consumption_calculator import ECC
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.indexing_utils import (
    find_speed_rise_entry_and_fall,
    get_interval_index_scalar_numba,
)


class GeneralOperation(NamedTuple):
    acc: float
    operation_time: float


class ForwardBeginPoint(NamedTuple):
    begin_pos: float
    begin_speed: float
    begin_interval: int


class BackwardEndPoint(NamedTuple):
    end_pos: float
    end_speed: float
    end_interval: int


class AscendOperation(TypedDict):
    ascend_begin_pos: float
    ascend_begin_speed: float
    ascend_operation_time: float
    ascend_end_pos: float
    ascend_end_interval: int
    ascend_begin_interval: int


class DescendOperation(TypedDict):
    descend_end_pos: float
    descend_end_speed: float
    descend_operation_time: float
    descend_begin_pos: float
    descend_begin_interval: int
    descend_end_interval: int


@njit(cache=True)
def _get_speed_limits_interval_index_numba(
    speed_limit_intervals: NDArray[np.float64], pos: float, ascend: bool
) -> int:
    return get_interval_index_scalar_numba(pos, speed_limit_intervals, ascend)


@njit(cache=True)
def _calc_mb_descend_operation_numba(
    end_pos: float,
    end_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_dec_abs: float,
    tol: float,
) -> tuple[float, float, int]:
    n_limits = speed_limits.size
    begin_interval = _get_speed_limits_interval_index_numba(
        speed_limit_intervals, end_pos, False
    )
    begin_pos = end_pos
    operation_time = 0.0

    while begin_interval >= 0:
        mark_pos = speed_limit_intervals[begin_interval]
        begin_speed = speed_limits[begin_interval] * factor
        operation_time = (begin_speed - end_speed) / max_dec_abs
        begin_pos = end_pos - (begin_speed * begin_speed - end_speed * end_speed) / (
            2.0 * max_dec_abs
        )

        if begin_pos > mark_pos or np.abs(begin_pos - mark_pos) <= tol:
            break

        distance = end_pos - mark_pos
        edge_speed_2 = end_speed * end_speed + 2.0 * max_dec_abs * distance
        if edge_speed_2 < 0.0:
            edge_speed_2 = 0.0
        edge_speed = np.sqrt(edge_speed_2)

        next_idx = begin_interval - 1
        if next_idx < 0:
            next_idx = 0
        next_interval_speed_limit = speed_limits[next_idx] * factor

        if (
            edge_speed < next_interval_speed_limit
            or np.abs(edge_speed - next_interval_speed_limit) <= tol
        ):
            begin_interval -= 1
        else:
            break

    if n_limits <= 0:
        clipped_idx = 0
    else:
        if begin_interval < 0:
            clipped_idx = 0
        elif begin_interval >= n_limits:
            clipped_idx = n_limits - 1
        else:
            clipped_idx = begin_interval

    return begin_pos, operation_time, clipped_idx


@njit(cache=True)
def _calc_ma_ascend_operation_numba(
    begin_pos: float,
    begin_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_acc: float,
    tol: float,
) -> tuple[float, float, int]:
    n_limits = speed_limits.size
    end_interval = _get_speed_limits_interval_index_numba(
        speed_limit_intervals, begin_pos, True
    )
    end_pos = begin_pos
    operation_time = 0.0

    while end_interval <= n_limits - 1:
        mark_pos = speed_limit_intervals[end_interval + 1]
        end_speed = speed_limits[end_interval] * factor
        operation_time = (end_speed - begin_speed) / max_acc
        end_pos = begin_pos + (end_speed * end_speed - begin_speed * begin_speed) / (
            2.0 * max_acc
        )

        if end_pos < mark_pos or np.abs(end_pos - mark_pos) <= tol:
            break

        distance = mark_pos - begin_pos
        edge_speed_2 = begin_speed * begin_speed + 2.0 * max_acc * distance
        if edge_speed_2 < 0.0:
            edge_speed_2 = 0.0
        edge_speed = np.sqrt(edge_speed_2)

        next_idx = end_interval + 1
        if next_idx < 0:
            next_idx = 0
        elif next_idx >= n_limits:
            next_idx = n_limits - 1
        next_interval_speed_limit = speed_limits[next_idx] * factor

        if (
            edge_speed < next_interval_speed_limit
            or np.abs(edge_speed - next_interval_speed_limit) <= tol
        ):
            end_interval += 1
        else:
            break

    if n_limits <= 0:
        clipped_idx = 0
    else:
        if end_interval < 0:
            clipped_idx = 0
        elif end_interval >= n_limits:
            clipped_idx = n_limits - 1
        else:
            clipped_idx = end_interval

    return end_pos, operation_time, clipped_idx


@njit(cache=True)
def _append_operation_numba(
    acc_arr: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    count: int,
    acc: float,
    operation_time: float,
) -> int:
    acc_arr[count] = acc
    time_arr[count] = operation_time
    return count + 1


@njit(cache=True)
def _nocruise_scenario_numba(
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
    max_acc: float,
    max_dec: float,
    max_dec_abs: float,
    acc_arr: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    count: int,
) -> tuple[int, float, bool]:
    """Append the no-cruise operations and report the sceptical position."""
    min_brake_distance = (begin_speed**2 - end_speed**2) / (2.0 * max_dec_abs)
    sceptical_pos = 0.0
    has_sceptical_pos = False
    if min_brake_distance > (end_pos - begin_pos):
        sceptical_pos = begin_pos - (begin_speed**2 - end_speed**2) / (
            2.0 * max_dec_abs
        )
        has_sceptical_pos = True
        operation_time = (begin_speed - end_speed) / max_dec_abs
        count = _append_operation_numba(
            acc_arr, time_arr, count, max_dec, operation_time
        )
    else:
        speed_peak_2 = (
            2.0 * max_acc * max_dec_abs * (end_pos - begin_pos)
            + max_dec_abs * begin_speed**2
            + max_acc * end_speed**2
        ) / (max_acc + max_dec_abs)
        speed_peak = np.sqrt(speed_peak_2)
        forward_time = (speed_peak - begin_speed) / max_acc
        backward_time = (speed_peak - end_speed) / max_dec_abs
        count = _append_operation_numba(acc_arr, time_arr, count, max_acc, forward_time)
        count = _append_operation_numba(
            acc_arr, time_arr, count, max_dec, backward_time
        )
    return count, sceptical_pos, has_sceptical_pos


@njit(cache=True)
def _cruise_scenario_numba(
    cruise_begin_pos: float,
    cruise_end_pos: float,
    cruise_interval: int,
    ma_time: float,
    mb_time: float,
    speed_limits: NDArray[np.float64],
    factor: float,
    max_acc: float,
    max_dec: float,
    acc_arr: NDArray[np.float64],
    time_arr: NDArray[np.float64],
    count: int,
) -> int:
    """Append max-acceleration, cruise, and max-braking operations."""
    cruise_speed = speed_limits[cruise_interval] * factor
    cruise_time = (cruise_end_pos - cruise_begin_pos) / cruise_speed
    count = _append_operation_numba(acc_arr, time_arr, count, max_acc, ma_time)
    count = _append_operation_numba(acc_arr, time_arr, count, 0.0, cruise_time)
    count = _append_operation_numba(acc_arr, time_arr, count, max_dec, mb_time)
    return count


@njit(cache=True)
def min_runtime_operations_numba(
    current_pos: float,
    current_speed: float,
    end_pos: float,
    end_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_acc: float,
    max_dec: float,
    max_dec_abs: float,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the minimum-runtime operation sequence in nopython mode.

    This is the jitted equivalent of the reference implementation. The
    result is two index-aligned float64 arrays of accelerations and durations.
    """
    n_limits = speed_limits.size
    speed_limits_scaled = speed_limits * factor

    tow_end_pos, tow_operation_time, tow_end_interval = _calc_ma_ascend_operation_numba(
        current_pos,
        current_speed,
        speed_limits,
        speed_limit_intervals,
        factor,
        max_acc,
        tol,
    )
    brake_begin_pos, brake_operation_time, brake_begin_interval = (
        _calc_mb_descend_operation_numba(
            end_pos,
            end_speed,
            speed_limits,
            speed_limit_intervals,
            factor,
            max_dec_abs,
            tol,
        )
    )

    capacity = 3 * n_limits + 8
    acc_arr = np.empty(capacity, dtype=np.float64)
    time_arr = np.empty(capacity, dtype=np.float64)
    count = 0

    if tow_end_interval >= brake_begin_interval:
        if tow_end_pos > brake_begin_pos:
            count, _, _ = _nocruise_scenario_numba(
                current_pos,
                current_speed,
                end_pos,
                end_speed,
                max_acc,
                max_dec,
                max_dec_abs,
                acc_arr,
                time_arr,
                count,
            )
        else:
            count = _cruise_scenario_numba(
                tow_end_pos,
                brake_begin_pos,
                brake_begin_interval,
                tow_operation_time,
                brake_operation_time,
                speed_limits,
                factor,
                max_acc,
                max_dec,
                acc_arr,
                time_arr,
                count,
            )
        return acc_arr[:count], time_arr[:count]

    table_size = n_limits + 4
    asc_begin_pos = np.empty(table_size)
    asc_begin_speed = np.empty(table_size)
    asc_operation_time = np.empty(table_size)
    asc_end_pos = np.empty(table_size)
    asc_end_interval = np.empty(table_size, dtype=np.int64)

    desc_end_pos = np.empty(table_size)
    desc_end_speed = np.empty(table_size)
    desc_operation_time = np.empty(table_size)
    desc_begin_pos = np.empty(table_size)
    desc_begin_interval = np.empty(table_size, dtype=np.int64)
    desc_end_interval = np.empty(table_size, dtype=np.int64)

    n_ascend = 1
    asc_begin_pos[0] = current_pos
    asc_begin_speed[0] = current_speed
    asc_operation_time[0] = tow_operation_time
    asc_end_pos[0] = tow_end_pos
    asc_end_interval[0] = tow_end_interval

    prev_ascend_end_interval = tow_end_interval
    for i in range(tow_end_interval, brake_begin_interval):
        if i >= n_limits - 1:
            continue
        if speed_limits[i + 1] <= speed_limits[i]:
            continue
        next_interval = i + 1
        if next_interval > prev_ascend_end_interval:
            end_pos_i, operation_time_i, end_interval_i = (
                _calc_ma_ascend_operation_numba(
                    speed_limit_intervals[next_interval],
                    speed_limits_scaled[i],
                    speed_limits,
                    speed_limit_intervals,
                    factor,
                    max_acc,
                    tol,
                )
            )
            asc_begin_pos[n_ascend] = speed_limit_intervals[next_interval]
            asc_begin_speed[n_ascend] = speed_limits_scaled[i]
            asc_operation_time[n_ascend] = operation_time_i
            asc_end_pos[n_ascend] = end_pos_i
            asc_end_interval[n_ascend] = end_interval_i
            n_ascend += 1
            prev_ascend_end_interval = end_interval_i

    n_descend = 1
    desc_end_pos[0] = end_pos
    desc_end_speed[0] = end_speed
    desc_operation_time[0] = brake_operation_time
    desc_begin_pos[0] = brake_begin_pos
    desc_begin_interval[0] = brake_begin_interval
    desc_end_interval[0] = _get_speed_limits_interval_index_numba(
        speed_limit_intervals, end_pos, False
    )

    prev_descend_begin_interval = brake_begin_interval
    for j in range(brake_begin_interval - 1, tow_end_interval - 1, -1):
        if j < 0:
            continue
        if speed_limits[j + 1] >= speed_limits[j]:
            continue
        if j < prev_descend_begin_interval:
            begin_pos_j, operation_time_j, begin_interval_j = (
                _calc_mb_descend_operation_numba(
                    speed_limit_intervals[j + 1],
                    speed_limits_scaled[j + 1],
                    speed_limits,
                    speed_limit_intervals,
                    factor,
                    max_dec_abs,
                    tol,
                )
            )
            desc_end_pos[n_descend] = speed_limit_intervals[j + 1]
            desc_end_speed[n_descend] = speed_limits_scaled[j + 1]
            desc_operation_time[n_descend] = operation_time_j
            desc_begin_pos[n_descend] = begin_pos_j
            desc_begin_interval[n_descend] = begin_interval_j
            desc_end_interval[n_descend] = j
            n_descend += 1
            prev_descend_begin_interval = j

    sceptical_pos = 0.0
    has_sceptical_pos = False
    current_interval = tow_end_interval
    while current_interval <= brake_begin_interval:
        ascend_idx = -1
        for k in range(n_ascend):
            if asc_end_interval[k] >= current_interval:
                ascend_idx = k
                break
        descend_idx = -1
        for k in range(n_descend):
            if desc_begin_interval[k] <= current_interval:
                descend_idx = k
                break

        if ascend_idx >= 0 and descend_idx >= 0:
            if asc_end_pos[ascend_idx] > desc_begin_pos[descend_idx]:
                count, sceptical_pos, has_sceptical_pos = _nocruise_scenario_numba(
                    asc_begin_pos[ascend_idx],
                    asc_begin_speed[ascend_idx],
                    desc_end_pos[descend_idx],
                    desc_end_speed[descend_idx],
                    max_acc,
                    max_dec,
                    max_dec_abs,
                    acc_arr,
                    time_arr,
                    count,
                )
            else:
                count = _cruise_scenario_numba(
                    asc_end_pos[ascend_idx],
                    desc_begin_pos[descend_idx],
                    current_interval,
                    asc_operation_time[ascend_idx],
                    desc_operation_time[descend_idx],
                    speed_limits,
                    factor,
                    max_acc,
                    max_dec,
                    acc_arr,
                    time_arr,
                    count,
                )
            current_interval = desc_end_interval[descend_idx] + 1
        elif ascend_idx >= 0:
            count = _append_operation_numba(
                acc_arr,
                time_arr,
                count,
                max_acc,
                asc_operation_time[ascend_idx],
            )
            cruise_speed = speed_limits[current_interval] * factor
            cruise_distance = (
                speed_limit_intervals[asc_end_interval[ascend_idx] + 1]
                - asc_end_pos[ascend_idx]
            )
            count = _append_operation_numba(
                acc_arr, time_arr, count, 0.0, cruise_distance / cruise_speed
            )
            current_interval = asc_end_interval[ascend_idx] + 1
        elif descend_idx >= 0:
            cruise_start_pos = (
                sceptical_pos
                if has_sceptical_pos
                else speed_limit_intervals[current_interval]
            )
            cruise_distance = desc_begin_pos[descend_idx] - cruise_start_pos
            descend_time = desc_operation_time[descend_idx]
            if cruise_distance < 0.0:
                count = _append_operation_numba(
                    acc_arr, time_arr, count, max_dec, descend_time
                )
                sceptical_pos += (
                    speed_limits[current_interval] * factor * descend_time
                    + 0.5 * max_dec * descend_time**2
                )
                has_sceptical_pos = True
            else:
                cruise_speed = speed_limits[current_interval] * factor
                count = _append_operation_numba(
                    acc_arr, time_arr, count, 0.0, cruise_distance / cruise_speed
                )
                count = _append_operation_numba(
                    acc_arr, time_arr, count, max_dec, descend_time
                )
                sceptical_pos = 0.0
                has_sceptical_pos = False
            current_interval = desc_end_interval[descend_idx] + 1
        else:
            cruise_speed = speed_limits[current_interval] * factor
            cruise_distance = (
                speed_limit_intervals[current_interval + 1]
                - speed_limit_intervals[current_interval]
            )
            count = _append_operation_numba(
                acc_arr, time_arr, count, 0.0, cruise_distance / cruise_speed
            )
            current_interval += 1

    return acc_arr[:count], time_arr[:count]


@njit(cache=True)
def min_operation_time_numba(
    current_pos: float,
    current_speed: float,
    end_pos: float,
    end_speed: float,
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    factor: float,
    max_acc: float,
    max_dec: float,
    max_dec_abs: float,
    tol: float,
) -> float:
    """Return only the total minimum operation time (nopython fast path)."""
    _, time_arr = min_runtime_operations_numba(
        current_pos,
        current_speed,
        end_pos,
        end_speed,
        speed_limits,
        speed_limit_intervals,
        factor,
        max_acc,
        max_dec,
        max_dec_abs,
        tol,
    )
    return float(np.sum(time_arr))


def _get_speed_limits_interval_index_reference(
    speed_limit_intervals: NDArray[np.float64],
    pos: float,
    *,
    ascend: bool = True,
) -> int:
    """Return the interval index containing ``pos`` (side depends on direction)."""
    side = "right" if ascend else "left"
    return int(np.searchsorted(speed_limit_intervals, pos, side=side) - 1)


def _find_speed_rise_entry_and_fall_reference(
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    *,
    factor: float,
    start_idx: int,
    end_idx: int,
) -> tuple[list[ForwardBeginPoint], list[BackwardEndPoint]]:
    """Return scaled rise entries and fall exits in ``[start_idx, end_idx)``."""
    rise_entries, fall_exits = find_speed_rise_entry_and_fall(
        speed_limits=speed_limits,
        interval_points=speed_limit_intervals,
        start_idx=start_idx,
        end_idx=end_idx,
        speed_factor=factor,
    )
    ascend_begin_points = [
        ForwardBeginPoint(
            float(entry.boundary_pos),
            float(entry.left_speed_scaled),
            int(entry.next_interval),
        )
        for entry in rise_entries
    ]
    descend_end_points = [
        BackwardEndPoint(
            float(entry.boundary_pos),
            float(entry.right_speed_scaled),
            int(entry.prev_interval),
        )
        for entry in fall_exits
    ]
    return ascend_begin_points, descend_end_points


def _calc_mb_descend_operation_reference(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    end_pos: float,
    end_speed: float,
) -> tuple[float, float, int]:
    """Backward max-braking operation for the reference implementation."""
    begin_pos, operation_time, begin_interval = _calc_mb_descend_operation_numba(
        end_pos=float(end_pos),
        end_speed=float(end_speed),
        speed_limits=track.speed_limits,
        speed_limit_intervals=track.speed_limit_intervals,
        factor=float(factor),
        max_dec_abs=float(vehicle.max_dec_abs),
        tol=1e-9,
    )
    return float(begin_pos), float(operation_time), int(begin_interval)


def _calc_ma_ascend_operation_reference(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    begin_pos: float,
    begin_speed: float,
) -> tuple[float, float, int]:
    """Forward max-acceleration operation for the reference implementation."""
    end_pos, operation_time, end_interval = _calc_ma_ascend_operation_numba(
        begin_pos=float(begin_pos),
        begin_speed=float(begin_speed),
        speed_limits=track.speed_limits,
        speed_limit_intervals=track.speed_limit_intervals,
        factor=float(factor),
        max_acc=float(vehicle.max_acc),
        tol=1e-9,
    )
    return float(end_pos), float(operation_time), int(end_interval)


def _calc_withnocruise_scenario_reference(
    vehicle: VehicleInfo,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> tuple[list[GeneralOperation], float | None]:
    """Return the no-cruise operation sequence for the reference implementation."""
    operation: list[GeneralOperation] = []
    max_dec_abs = float(vehicle.max_dec_abs)
    min_brake_distance = (begin_speed**2 - end_speed**2) / (2.0 * max_dec_abs)
    sceptical_pos: float | None = None
    if min_brake_distance > (end_pos - begin_pos):
        sceptical_pos = begin_pos - (begin_speed**2 - end_speed**2) / (
            2.0 * max_dec_abs
        )
        operation_time = (begin_speed - end_speed) / max_dec_abs
        operation.append(GeneralOperation(vehicle.max_dec, operation_time))
    else:
        speed_peak_2 = (
            2.0 * vehicle.max_acc * max_dec_abs * (end_pos - begin_pos)
            + max_dec_abs * begin_speed**2
            + vehicle.max_acc * end_speed**2
        ) / (vehicle.max_acc + max_dec_abs)
        speed_peak = np.sqrt(speed_peak_2)
        forward_operation_time = (speed_peak - begin_speed) / vehicle.max_acc
        backward_operation_time = (speed_peak - end_speed) / max_dec_abs
        operation.append(GeneralOperation(vehicle.max_acc, forward_operation_time))
        operation.append(GeneralOperation(vehicle.max_dec, backward_operation_time))
    return operation, sceptical_pos


def _calc_withcruise_scenario_reference(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    cruise_begin_pos: float,
    cruise_end_pos: float,
    cruise_interval: int,
    ma_time: float,
    mb_time: float,
) -> list[GeneralOperation]:
    """Return the max-accel/cruise/max-brake sequence (reference implementation)."""
    cruise_speed = track.speed_limits[cruise_interval] * factor
    cruise_time = (cruise_end_pos - cruise_begin_pos) / cruise_speed
    return [
        GeneralOperation(vehicle.max_acc, ma_time),
        GeneralOperation(0.0, cruise_time),
        GeneralOperation(vehicle.max_dec, mb_time),
    ]


def _calc_min_runtime_operation_reference(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    current_pos: float,
    current_speed: float,
    end_pos: float,
    end_speed: float,
) -> list[GeneralOperation]:
    """Pure-Python reference for the minimum-runtime operation sequence."""
    tow_begin_speed = current_speed
    tow_begin_pos = current_pos
    brake_end_speed = end_speed
    brake_end_pos = end_pos

    (
        tow_end_pos,
        tow_operation_time,
        tow_end_interval,
    ) = _calc_ma_ascend_operation_reference(
        vehicle, track, factor, tow_begin_pos, tow_begin_speed
    )

    (
        brake_begin_pos,
        brake_operation_time,
        brake_begin_interval,
    ) = _calc_mb_descend_operation_reference(
        vehicle, track, factor, brake_end_pos, brake_end_speed
    )
    operations: list[GeneralOperation] = []

    if tow_end_interval < brake_begin_interval:
        ascend_begin_points, descend_end_points = (
            _find_speed_rise_entry_and_fall_reference(
                track.speed_limits,
                track.speed_limit_intervals,
                factor=factor,
                start_idx=tow_end_interval,
                end_idx=brake_begin_interval,
            )
        )

        ascend_operations: list[AscendOperation] = [
            {
                "ascend_begin_pos": tow_begin_pos,
                "ascend_begin_speed": tow_begin_speed,
                "ascend_operation_time": tow_operation_time,
                "ascend_end_pos": tow_end_pos,
                "ascend_end_interval": tow_end_interval,
                "ascend_begin_interval": _get_speed_limits_interval_index_reference(
                    track.speed_limit_intervals,
                    tow_begin_pos,
                    ascend=True,
                ),
            }
        ]

        descend_operations: list[DescendOperation] = [
            {
                "descend_end_pos": brake_end_pos,
                "descend_end_speed": brake_end_speed,
                "descend_operation_time": brake_operation_time,
                "descend_begin_pos": brake_begin_pos,
                "descend_begin_interval": brake_begin_interval,
                "descend_end_interval": _get_speed_limits_interval_index_reference(
                    track.speed_limit_intervals,
                    brake_end_pos,
                    ascend=False,
                ),
            }
        ]
        prev_ascend_end_interval = tow_end_interval
        for (
            ascend_begin_pos,
            ascend_begin_speed,
            ascend_begin_interval,
        ) in ascend_begin_points:
            if ascend_begin_interval > prev_ascend_end_interval:
                (
                    ascend_end_pos,
                    ascend_operation_time,
                    ascend_end_interval,
                ) = _calc_ma_ascend_operation_reference(
                    vehicle,
                    track,
                    factor,
                    ascend_begin_pos,
                    ascend_begin_speed,
                )
                ascend_operations.append(
                    {
                        "ascend_begin_pos": ascend_begin_pos,
                        "ascend_begin_speed": ascend_begin_speed,
                        "ascend_operation_time": ascend_operation_time,
                        "ascend_end_pos": ascend_end_pos,
                        "ascend_end_interval": ascend_end_interval,
                        "ascend_begin_interval": ascend_begin_interval,
                    }
                )
                prev_ascend_end_interval = ascend_end_interval

        prev_descend_begin_interval = brake_begin_interval
        for (
            descend_end_pos,
            descend_end_speed,
            descend_end_interval,
        ) in reversed(descend_end_points):
            if descend_end_interval < prev_descend_begin_interval:
                (
                    descend_begin_pos,
                    descend_operation_time,
                    descend_begin_interval,
                ) = _calc_mb_descend_operation_reference(
                    vehicle,
                    track,
                    factor,
                    descend_end_pos,
                    descend_end_speed,
                )
                descend_operations.append(
                    {
                        "descend_end_pos": descend_end_pos,
                        "descend_end_speed": descend_end_speed,
                        "descend_operation_time": descend_operation_time,
                        "descend_begin_pos": descend_begin_pos,
                        "descend_begin_interval": descend_begin_interval,
                        "descend_end_interval": descend_end_interval,
                    }
                )

        sceptical_pos: float | None = None
        current_interval = tow_end_interval
        while current_interval <= brake_begin_interval:
            ascend_operation_idx = next(
                (
                    i
                    for i, fo in enumerate(ascend_operations)
                    if fo["ascend_end_interval"] >= current_interval
                ),
                None,
            )
            descend_operation_idx = next(
                (
                    i
                    for i, bo in enumerate(descend_operations)
                    if bo["descend_begin_interval"] <= current_interval
                ),
                None,
            )
            if (ascend_operation_idx is not None) and (
                descend_operation_idx is not None
            ):
                ascend_op = ascend_operations[ascend_operation_idx]
                descend_op = descend_operations[descend_operation_idx]
                if ascend_op["ascend_end_pos"] > descend_op["descend_begin_pos"]:
                    middle_operations, sceptical_pos = (
                        _calc_withnocruise_scenario_reference(
                            vehicle,
                            ascend_op["ascend_begin_pos"],
                            ascend_op["ascend_begin_speed"],
                            descend_op["descend_end_pos"],
                            descend_op["descend_end_speed"],
                        )
                    )
                else:
                    middle_operations = _calc_withcruise_scenario_reference(
                        vehicle,
                        track,
                        factor,
                        ascend_op["ascend_end_pos"],
                        descend_op["descend_begin_pos"],
                        current_interval,
                        ascend_op["ascend_operation_time"],
                        descend_op["descend_operation_time"],
                    )
                operations += middle_operations
                current_interval = descend_op["descend_end_interval"] + 1
            elif (ascend_operation_idx is not None) and (descend_operation_idx is None):
                ascend_op = ascend_operations[ascend_operation_idx]
                operations.append(
                    GeneralOperation(
                        vehicle.max_acc, ascend_op["ascend_operation_time"]
                    )
                )
                cruise_speed = track.speed_limits[current_interval] * factor
                cruise_distance = (
                    track.speed_limit_intervals[ascend_op["ascend_end_interval"] + 1]
                    - ascend_op["ascend_end_pos"]
                )
                operations.append(GeneralOperation(0.0, cruise_distance / cruise_speed))
                current_interval = ascend_op["ascend_end_interval"] + 1
            elif (ascend_operation_idx is None) and (descend_operation_idx is not None):
                descend_op = descend_operations[descend_operation_idx]
                cruise_distance = descend_op["descend_begin_pos"] - (
                    sceptical_pos
                    if sceptical_pos is not None
                    else track.speed_limit_intervals[current_interval]
                )
                dot = descend_op["descend_operation_time"]
                if cruise_distance < 0.0:
                    operations.append(GeneralOperation(vehicle.max_dec, dot))
                    sceptical_pos += (
                        track.speed_limits[current_interval] * factor * dot
                        + 0.5 * vehicle.max_dec * dot**2
                    )
                else:
                    cruise_speed = track.speed_limits[current_interval] * factor
                    operations.append(
                        GeneralOperation(0.0, cruise_distance / cruise_speed)
                    )
                    operations.append(GeneralOperation(vehicle.max_dec, dot))
                    sceptical_pos = None
                current_interval = descend_op["descend_end_interval"] + 1
            else:
                cruise_speed = track.speed_limits[current_interval] * factor
                cruise_distance = (
                    track.speed_limit_intervals[current_interval + 1]
                    - track.speed_limit_intervals[current_interval]
                )
                operations.append(GeneralOperation(0.0, cruise_distance / cruise_speed))
                current_interval += 1
    else:
        if tow_end_pos > brake_begin_pos:
            operations, _ = _calc_withnocruise_scenario_reference(
                vehicle,
                tow_begin_pos,
                tow_begin_speed,
                brake_end_pos,
                brake_end_speed,
            )
        else:
            operations = _calc_withcruise_scenario_reference(
                vehicle,
                track,
                factor,
                tow_end_pos,
                brake_begin_pos,
                brake_begin_interval,
                tow_operation_time,
                brake_operation_time,
            )

    return operations


def min_operation_time(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> float:
    """Return the minimum operation time from ``begin`` to ``end``."""
    return float(
        min_operation_time_numba(
            current_pos=begin_pos,
            current_speed=begin_speed,
            end_pos=end_pos,
            end_speed=end_speed,
            speed_limits=track.speed_limits,
            speed_limit_intervals=track.speed_limit_intervals,
            factor=float(factor),
            max_acc=float(vehicle.max_acc),
            max_dec=float(vehicle.max_dec),
            max_dec_abs=float(vehicle.max_dec_abs),
            tol=1e-9,
        )
    )


def min_operation_time_curve(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the minimum-time speed curve as sampled (position, speed) arrays."""
    acc_arr, time_arr = min_runtime_operations_numba(
        current_pos=begin_pos,
        current_speed=begin_speed,
        end_pos=end_pos,
        end_speed=end_speed,
        speed_limits=track.speed_limits,
        speed_limit_intervals=track.speed_limit_intervals,
        factor=float(factor),
        max_acc=float(vehicle.max_acc),
        max_dec=float(vehicle.max_dec),
        max_dec_abs=float(vehicle.max_dec_abs),
        tol=1e-9,
    )
    curve_pos_array = np.array([begin_pos], dtype=np.float64)
    curve_speed_array = np.array([begin_speed], dtype=np.float64)
    for acc, operation_time in zip(acc_arr, time_arr, strict=True):
        operation_time_value = float(operation_time)
        if operation_time_value <= 0:
            continue
        dt = 0.1
        n_steps = max(int(np.floor(operation_time_value / dt)), 2)
        t_samples = np.linspace(
            0.0, operation_time_value, n_steps, endpoint=True, dtype=np.float64
        )
        acc_value = float(acc)
        speeds = begin_speed + acc_value * t_samples
        positions = begin_pos + begin_speed * t_samples + 0.5 * acc_value * t_samples**2
        curve_pos_array = np.concatenate((curve_pos_array[:-1], positions))
        curve_speed_array = np.concatenate((curve_speed_array[:-1], speeds))
        begin_pos = curve_pos_array[-1]
        begin_speed = curve_speed_array[-1]

    if curve_pos_array.size > 1:
        keep_mask = np.empty(curve_pos_array.size, dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = np.diff(curve_pos_array) != 0.0
        curve_pos_array = curve_pos_array[keep_mask]
        curve_speed_array = curve_speed_array[keep_mask]

    return curve_pos_array, curve_speed_array


def max_energy_and_min_operation_time(
    vehicle: VehicleInfo,
    track: TrackInfo,
    factor: float,
    energy_con_calc: ECC,
    begin_pos: float,
    begin_speed: float,
    end_pos: float,
    end_speed: float,
    distance: float,
) -> tuple[float, float, float]:
    """Return reference max-energy consumption and the minimum operation time."""
    acc_arr, time_arr = min_runtime_operations_numba(
        current_pos=begin_pos,
        current_speed=begin_speed,
        end_pos=end_pos,
        end_speed=end_speed,
        speed_limits=track.speed_limits,
        speed_limit_intervals=track.speed_limit_intervals,
        factor=float(factor),
        max_acc=float(vehicle.max_acc),
        max_dec=float(vehicle.max_dec),
        max_dec_abs=float(vehicle.max_dec_abs),
        tol=1e-9,
    )

    ref_mec = 0.0
    ref_lec = 0.0
    ref_operation_time = 0.0
    accumulated_distance = 0.0

    current_pos_i = float(begin_pos)
    current_speed_i = float(begin_speed)

    for acc, operation_time in zip(acc_arr, time_arr, strict=True):
        acc_value = float(acc)
        operation_time_value = float(operation_time)
        segment_distance = (
            current_speed_i * operation_time_value
            + 0.5 * acc_value * operation_time_value**2
        )
        if accumulated_distance + segment_distance >= float(distance):
            remaining_displacement = float(distance) - accumulated_distance
            if np.abs(acc_value) < 1e-9:
                actual_time = remaining_displacement / np.maximum(current_speed_i, 1e-6)
            else:
                discriminant = (
                    current_speed_i**2 + 2 * acc_value * remaining_displacement
                )
                discriminant = max(discriminant, 0)
                actual_time = (np.sqrt(discriminant) - current_speed_i) / acc_value
            pec, lec = energy_con_calc.calc_energy(
                begin_pos=current_pos_i,
                begin_speed=current_speed_i,
                acc=acc_value,
                distance=remaining_displacement,
                direction=1,
                operation_time=actual_time,
                vehicle=vehicle,
                track=track,
            )
            ref_mec += pec
            ref_lec += lec
            ref_operation_time += actual_time
            break
        pec, lec = energy_con_calc.calc_energy(
            begin_pos=current_pos_i,
            begin_speed=current_speed_i,
            acc=acc_value,
            distance=segment_distance,
            direction=1,
            operation_time=operation_time_value,
            vehicle=vehicle,
            track=track,
        )
        ref_mec += pec
        ref_lec += lec
        ref_operation_time += operation_time_value
        accumulated_distance += segment_distance
        current_pos_i += segment_distance
        current_speed_i += acc_value * operation_time_value

    return float(ref_mec), float(ref_lec), float(ref_operation_time)
