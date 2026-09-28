"""Graph construction, stage division, and transition computations for DP."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import signal
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import EnergyParams, segment_energy
from mtto.domain.kinematics import accel_between
from mtto.domain.line import Line
from mtto.domain.safeguard import (
    Safeguard,
    detect_any_danger,
    intersecting_danger_points,
)
from mtto.domain.srtsp import SrtspCurve, interp_upper_speed

__all__ = [
    "DP_UPPER_SPEED_ENVELOPE_VERSION",
    "ParallelPrecomputeExitedError",
    "STATIC_REGION_SAMPLE_STEP_M",
    "SparseTransitionEntry",
    "SparseTransitionRows",
    "TransitionBatchResult",
    "TransitionGraph",
    "TransitionPayload",
    "build_speed_states",
    "build_transaction_batch",
    "build_transition_graph",
    "build_transition_graph_parallel",
    "build_transition_graph_serial",
    "cancel_parallel_futures",
    "generate_stages",
    "generate_uniform_spacing_stages",
    "generate_variable_spacing_stages",
    "get_stage_speed_upper_indices",
    "make_task_ranges",
    "merge_transition_batch",
    "resolve_parallel_config",
    "speed_tolerance",
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

STATIC_REGION_SAMPLE_STEP_M = 10.0
DP_UPPER_SPEED_ENVELOPE_VERSION = 1


class ParallelPrecomputeExitedError(RuntimeError):
    """Raised when parallel precompute exits and DP should stop immediately."""


TransitionPayload = tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]
SparseTransitionEntry = tuple[
    int,
    NDArray[np.int_],
    NDArray[np.float64],
    NDArray[np.float64],
]
SparseTransitionRows = list[tuple[int, list[SparseTransitionEntry]]]
TransitionBatchResult = tuple[SparseTransitionRows, int, int]


class TransitionGraph(TypedDict):
    stages: NDArray[np.float64]
    speed_states: NDArray[np.float64]
    stage_speed_upper_idx: NDArray[np.int_]
    transitions: list[list[TransitionPayload | None]]
    total_valid_edges: int


@dataclass(frozen=True)
class _TransitionBuildContext:
    stages: NDArray[np.float64]
    speed_states: NDArray[np.float64]
    stage_speed_upper_idx: NDArray[np.int_]
    vehicle: Vehicle
    safeguard: Safeguard
    energy: EnergyParams
    track: Line
    srtsp_curve: SrtspCurve


class _CancellationEvent(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


_dp_parallel_context: _TransitionBuildContext | None = None
_dp_cancel_event: _CancellationEvent | None = None


def _calculate_transition_with_context(
    *,
    pos_k: float,
    speed_k: float,
    displacement: float,
    speed_k_1: float,
    vehicle: Vehicle,
    safeguard: Safeguard,
    energy: EnergyParams,
    track: Line,
    srtsp_curve: SrtspCurve,
) -> tuple[float, float] | None:
    if math.isclose(displacement, 0.0):
        return None

    if math.isclose(speed_k + speed_k_1, 0.0):
        return None

    acc, time = accel_between(
        speed_k,
        speed_k_1,
        displacement,
    )

    acc_tol = 1e-9
    if acc > vehicle.max_acc + acc_tol or acc < vehicle.max_dec - acc_tol:
        return None

    sample_count = max(
        2, math.ceil(abs(displacement) / STATIC_REGION_SAMPLE_STEP_M) + 1
    )
    distance_sample = np.linspace(0.0, displacement, sample_count, dtype=np.float64)
    pos_sample = distance_sample + pos_k

    # 由匀变速公式采样速度，保证与端点速度一致
    speed_sq_sample = 2.0 * acc * distance_sample + speed_k**2
    speed_sample = np.sqrt(np.maximum(speed_sq_sample, 0.0))

    upper_speed_sample = interp_upper_speed(
        srtsp_curve,
        pos_sample,
    )
    if np.any(speed_sample > upper_speed_sample):
        return None

    # 检查是否进入危险速度域
    if detect_any_danger(safeguard, pos=pos_sample, speed=speed_sample):
        return None

    propulsion_energy, leviation_energy = segment_energy(
        energy,
        vehicle,
        track,
        begin_pos=pos_k,
        begin_speed=speed_k,
        acc=acc,
        distance=abs(displacement),
        direction=1 if displacement > 0 else -1,
        operation_time=time,
    )

    total_energy = propulsion_energy + leviation_energy
    if not math.isfinite(float(time)) or time <= 0.0:
        return None
    if not math.isfinite(float(total_energy)):
        return None

    return total_energy, time


def build_transaction_batch(
    *,
    context: _TransitionBuildContext,
    k_start: int,
    k_end: int,
    cancel_event: _CancellationEvent | None = None,
) -> TransitionBatchResult:
    """Build one transition-graph batch for either serial or parallel dispatch."""
    total_steps = len(context.stages) - 1
    if not (0 <= k_start <= k_end <= total_steps):
        raise ValueError("transition batch range is outside the stage graph")

    batch_rows: SparseTransitionRows = []
    total_valid_edges = 0

    for k_idx in range(k_start, k_end):
        if cancel_event is not None and cancel_event.is_set():
            raise ParallelPrecomputeExitedError("并行预计算被主进程取消")
        pos_k = float(context.stages[k_idx])
        delta_pos = float(context.stages[k_idx + 1] - context.stages[k_idx])
        abs_delta_pos = abs(delta_pos)
        current_upper = int(context.stage_speed_upper_idx[k_idx])
        next_upper = int(context.stage_speed_upper_idx[k_idx + 1])

        if current_upper < 0 or next_upper < 0:
            continue

        row_entries: list[SparseTransitionEntry] = []

        for i in range(current_upper + 1):
            speed_k = float(context.speed_states[i])

            # 基于加减速度物理边界的下一阶段速度索引剪枝
            v2_min = max(
                speed_k**2 + 2.0 * context.vehicle.max_dec * abs_delta_pos,
                0.0,
            )
            v2_max = max(
                speed_k**2 + 2.0 * context.vehicle.max_acc * abs_delta_pos,
                0.0,
            )
            v_next_min = math.sqrt(v2_min)
            v_next_max = math.sqrt(v2_max)

            j_min = int(np.searchsorted(context.speed_states, v_next_min, side="left"))
            j_max = int(
                np.searchsorted(context.speed_states, v_next_max, side="right") - 1
            )
            j_max = min(j_max, next_upper)

            if j_min > j_max:
                continue

            next_indices: list[int] = []
            delta_energy_list: list[float] = []
            delta_time_list: list[float] = []

            for j in range(j_min, j_max + 1):
                speed_next = float(context.speed_states[j])
                transition = _calculate_transition_with_context(
                    pos_k=pos_k,
                    speed_k=speed_k,
                    displacement=delta_pos,
                    speed_k_1=speed_next,
                    vehicle=context.vehicle,
                    safeguard=context.safeguard,
                    energy=context.energy,
                    track=context.track,
                    srtsp_curve=context.srtsp_curve,
                )
                if transition is None:
                    continue
                delta_energy, delta_time = transition

                next_indices.append(j)
                delta_energy_list.append(delta_energy)
                delta_time_list.append(delta_time)

            if not next_indices:
                continue

            row_entries.append(
                (
                    i,
                    np.asarray(next_indices, dtype=np.int_),
                    np.asarray(delta_energy_list, dtype=np.float64),
                    np.asarray(delta_time_list, dtype=np.float64),
                )
            )
            total_valid_edges += len(next_indices)

        if row_entries:
            batch_rows.append((k_idx, row_entries))

    return batch_rows, total_valid_edges, k_end - k_start


def _init_transition_worker(
    context: _TransitionBuildContext,
    cancel_event: _CancellationEvent | None = None,
) -> None:
    global _dp_parallel_context
    global _dp_cancel_event
    try:
        _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
        sigbreak = getattr(signal, "SIGBREAK", None)
        if sigbreak is not None:
            _ = signal.signal(cast(int, sigbreak), signal.SIG_IGN)
    except ValueError, AttributeError:
        pass
    _dp_parallel_context = context
    _dp_cancel_event = cancel_event


def _compute_transition_batch_worker(k_start: int, k_end: int) -> TransitionBatchResult:
    if _dp_parallel_context is None:
        raise RuntimeError("worker context is not initialized")

    return build_transaction_batch(
        context=_dp_parallel_context,
        k_start=k_start,
        k_end=k_end,
        cancel_event=_dp_cancel_event,
    )


def speed_tolerance(speed: float) -> float:
    """Return proportional speed tolerance for grid quantization."""
    return max(1e-9, abs(float(speed)) * 1e-9)


def build_speed_states(
    speed_grid_upper_mps: float, delta_speed: float
) -> NDArray[np.float64]:
    """Build a stable grid bounded by the reachable route speed."""
    state_count = int(math.floor(speed_grid_upper_mps / delta_speed + 1e-12)) + 1
    speed_states = np.arange(state_count, dtype=np.float64) * delta_speed
    speed_states = speed_states[
        speed_states <= speed_grid_upper_mps + speed_tolerance(speed_grid_upper_mps)
    ]
    if speed_states.size == 0:
        raise ValueError("speed grid must contain at least the zero state")
    speed_states[0] = 0.0
    return speed_states


def generate_variable_spacing_stages(
    *,
    safeguard: Safeguard,
    sub_stage_count: int,
    start_position: float,
    target_position: float,
) -> NDArray[np.float64]:
    """Generate monotonic stages split at dangerous-region intersections."""
    low = min(start_position, target_position)
    high = max(start_position, target_position)
    dangerous_points = np.asarray(
        intersecting_danger_points(safeguard),
        dtype=np.float64,
    ).reshape(-1)
    dangerous_points = dangerous_points[np.isfinite(dangerous_points)]
    dangerous_points = np.unique(
        dangerous_points[(dangerous_points > low) & (dangerous_points < high)]
    )
    if target_position < start_position:
        dangerous_points = dangerous_points[::-1]

    critical_points = np.concatenate(
        (
            np.asarray([start_position], dtype=np.float64),
            dangerous_points,
            np.asarray([target_position], dtype=np.float64),
        )
    )
    stages: list[float] = []
    for index in range(len(critical_points) - 1):
        interval_start = float(critical_points[index])
        interval_end = float(critical_points[index + 1])
        if math.isclose(interval_start, interval_end, abs_tol=1e-12):
            continue
        partition = np.linspace(
            interval_start,
            interval_end,
            sub_stage_count + 1,
            dtype=np.float64,
        )
        if not stages:
            stages.extend(float(value) for value in partition)
        else:
            stages.extend(float(value) for value in partition[1:])

    result = np.asarray(stages, dtype=np.float64)
    if result.size < 2:
        raise ValueError("stage generation produced fewer than two stages")
    return result


def generate_uniform_spacing_stages(
    *,
    uniform_step_size: float,
    start_position: float,
    target_position: float,
) -> NDArray[np.float64]:
    """Generate monotonic stages with no interval longer than the step size."""
    num_steps = max(
        1,
        int(math.ceil(abs(target_position - start_position) / uniform_step_size)),
    )
    return np.linspace(
        start_position,
        target_position,
        num_steps + 1,
        dtype=np.float64,
    )


def generate_stages(
    *,
    stage_division: Literal["variable", "uniform"],
    sub_stage_count: int,
    uniform_step_size: float,
    start_position: float,
    target_position: float,
    safeguard: Safeguard,
) -> NDArray[np.float64]:
    """Dispatch stage generation according to the configured division mode."""
    if stage_division == "uniform":
        return generate_uniform_spacing_stages(
            uniform_step_size=uniform_step_size,
            start_position=start_position,
            target_position=target_position,
        )
    return generate_variable_spacing_stages(
        safeguard=safeguard,
        sub_stage_count=sub_stage_count,
        start_position=start_position,
        target_position=target_position,
    )


def get_stage_speed_upper_indices(
    *,
    safeguard: Safeguard,
    srtsp_curve: SrtspCurve,
    speed_grid_upper_mps: float,
    vehicle_max_speed: float,
    stages: NDArray[np.float64],
    speed_states: NDArray[np.float64],
) -> NDArray[np.int_]:
    """根据线路限速与任务相关最短运行时间包络生成阶段速度上界。"""
    speed_limits = safeguard.speed_limits
    speed_limit_intervals = safeguard.speed_limit_intervals
    if speed_limits.size == 0 or speed_limit_intervals.size == 0:
        raise ValueError("safeguard must provide speed limits")

    interval_indices = (
        np.searchsorted(
            speed_limit_intervals,
            stages,
            side="right",
        )
        - 1
    )
    interval_indices = np.clip(interval_indices, 0, len(speed_limits) - 1)
    stage_speed_upper = np.minimum(
        min(speed_grid_upper_mps, vehicle_max_speed),
        speed_limits[interval_indices] * float(safeguard.params.factor),
    )
    task_upper_speed = interp_upper_speed(srtsp_curve, stages)
    stage_speed_upper = np.minimum(stage_speed_upper, task_upper_speed)
    stage_speed_upper = np.maximum(stage_speed_upper, 0.0)
    upper_idx = np.searchsorted(speed_states, stage_speed_upper, side="right") - 1
    return np.clip(upper_idx, -1, len(speed_states) - 1).astype(np.int_)


def cancel_parallel_futures(
    executor: ProcessPoolExecutor | None,
    futures: list[Future[TransitionBatchResult]],
) -> None:
    """Cancel all pending parallel futures and shut down the executor."""
    for future in futures:
        _ = future.cancel()

    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


def make_task_ranges(total_steps: int, chunk_size: int) -> list[tuple[int, int]]:
    """Partition step indices into contiguous ranges of chunk_size."""
    if total_steps <= 0:
        return []
    return [
        (k_start, min(k_start + chunk_size, total_steps))
        for k_start in range(0, total_steps, chunk_size)
    ]


def merge_transition_batch(
    transitions: list[list[TransitionPayload | None]],
    batch_rows: SparseTransitionRows,
) -> None:
    """Merge sparse batch transition rows into the main transition graph."""
    for k_idx, row_entries in batch_rows:
        for i, next_idx, delta_energy, delta_time in row_entries:
            transitions[k_idx][i] = (next_idx, delta_energy, delta_time)


def resolve_parallel_config(
    total_steps: int,
    precompute_workers: int | None,
    precompute_chunk_size: int | None,
) -> tuple[int, int]:
    """Resolve the worker count and task chunk size for parallel precompute."""
    workers = precompute_workers
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    chunk_size = precompute_chunk_size
    if chunk_size is None:
        chunk_size = max(1, (total_steps + workers * 4 - 1) // (workers * 4))

    return workers, chunk_size


def build_transition_graph_serial(
    *,
    context: _TransitionBuildContext,
    precompute_chunk_size: int | None = None,
    show_precompute_progress: bool = True,
    precompute_progress_desc: str = "状态转移图预计算",
) -> tuple[list[list[TransitionPayload | None]], int]:
    """Build transition graph sequentially stage by stage."""
    total_steps = len(context.stages) - 1
    num_speed_states = len(context.speed_states)
    transitions: list[list[TransitionPayload | None]] = [
        [None for _ in range(num_speed_states)] for _ in range(total_steps)
    ]
    total_valid_edges = 0

    chunk_size = precompute_chunk_size or max(1, total_steps)
    task_ranges = make_task_ranges(total_steps, chunk_size)
    progress_bar = None
    if show_precompute_progress and tqdm is not None:
        progress_bar = tqdm(
            total=total_steps,
            desc=precompute_progress_desc,
            dynamic_ncols=True,
            unit="stage",
            mininterval=0.2,
        )

    try:
        for k_start, k_end in task_ranges:
            batch = build_transaction_batch(
                context=context,
                k_start=k_start,
                k_end=k_end,
            )
            batch_rows, batch_valid_edges, batch_steps = batch
            merge_transition_batch(transitions, batch_rows)
            total_valid_edges += batch_valid_edges
            if progress_bar is not None:
                _ = progress_bar.update(batch_steps)
    except KeyboardInterrupt:
        logger.info("检测到 Ctrl+C，正在终止串行预计算任务...")
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return transitions, total_valid_edges


def build_transition_graph_parallel(
    *,
    context: _TransitionBuildContext,
    precompute_workers: int | None = None,
    precompute_chunk_size: int | None = None,
    mp_start_method: str | None = None,
    show_precompute_progress: bool = True,
    precompute_progress_desc: str = "状态转移图预计算",
) -> tuple[list[list[TransitionPayload | None]], int]:
    """Build transition graph concurrently across process workers."""
    total_steps = len(context.stages) - 1
    num_speed_states = len(context.speed_states)
    workers, chunk_size = resolve_parallel_config(
        total_steps, precompute_workers, precompute_chunk_size
    )
    task_ranges = make_task_ranges(total_steps, chunk_size)
    if workers <= 1 or total_steps < 2:
        logger.info("并行预计算条件不满足，自动回退串行模式。")
        return build_transition_graph_serial(
            context=context,
            precompute_chunk_size=chunk_size,
            show_precompute_progress=show_precompute_progress,
            precompute_progress_desc=precompute_progress_desc,
        )
    if len(task_ranges) <= 1:
        logger.info("并行预计算任务过少，自动回退串行模式。")
        return build_transition_graph_serial(
            context=context,
            precompute_chunk_size=chunk_size,
            show_precompute_progress=show_precompute_progress,
            precompute_progress_desc=precompute_progress_desc,
        )

    logger.info(
        "并行预计算配置: workers=%s, chunk_size=%s, tasks=%s",
        workers,
        chunk_size,
        len(task_ranges),
    )

    transitions: list[list[TransitionPayload | None]] = [
        [None for _ in range(num_speed_states)] for _ in range(total_steps)
    ]
    total_valid_edges = 0

    progress_bar = None
    if show_precompute_progress and tqdm is not None:
        progress_bar = tqdm(
            total=total_steps,
            desc=f"{precompute_progress_desc}(并行)",
            dynamic_ncols=True,
            unit="stage",
            mininterval=0.2,
        )

    executor: ProcessPoolExecutor | None = None
    futures: list[Future[TransitionBatchResult]] = []
    shutdown_called = False
    manager = None
    cancel_event: _CancellationEvent | None = None

    try:
        mp_context = (
            mp.get_context(mp_start_method)
            if mp_start_method is not None
            else mp.get_context()
        )
        manager = mp_context.Manager()
        cancel_event = manager.Event()
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=_init_transition_worker,
            initargs=(context, cancel_event),
        )
        futures = [
            executor.submit(_compute_transition_batch_worker, k_start, k_end)
            for k_start, k_end in task_ranges
        ]
        for future in as_completed(futures):
            batch_rows, batch_valid_edges, batch_steps = future.result()
            total_valid_edges += batch_valid_edges

            merge_transition_batch(transitions, batch_rows)

            if progress_bar is not None:
                _ = progress_bar.update(batch_steps)
    except KeyboardInterrupt:
        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None
        logger.info("检测到 Ctrl+C，正在终止并行预计算任务...")
        if cancel_event is not None:
            cancel_event.set()
        cancel_parallel_futures(executor=executor, futures=futures)
        shutdown_called = True
        raise
    except Exception as exc:
        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None
        logger.exception("并行预计算失败，准备终止动态规划。")
        if cancel_event is not None:
            cancel_event.set()
        cancel_parallel_futures(executor=executor, futures=futures)
        shutdown_called = True
        raise ParallelPrecomputeExitedError(f"并行预计算异常退出: {exc}") from exc
    finally:
        if progress_bar is not None:
            progress_bar.close()
        try:
            if executor is not None and not shutdown_called:
                executor.shutdown(wait=True, cancel_futures=False)
        finally:
            if manager is not None:
                manager.shutdown()

    return transitions, total_valid_edges


def build_transition_graph(
    *,
    stages: NDArray[np.float64],
    speed_states: NDArray[np.float64],
    stage_speed_upper_idx: NDArray[np.int_],
    vehicle: Vehicle,
    safeguard: Safeguard,
    energy: EnergyParams,
    track: Line,
    srtsp_curve: SrtspCurve,
    precompute_mode: Literal["serial", "parallel"] = "serial",
    precompute_workers: int | None = None,
    precompute_chunk_size: int | None = None,
    mp_start_method: str | None = None,
    show_precompute_progress: bool = True,
    precompute_progress_desc: str = "状态转移图预计算",
) -> TransitionGraph:
    """预计算状态转移图（可行性/能耗/时间）, 供外层不同 lambda 复用。"""
    context = _TransitionBuildContext(
        stages=stages,
        speed_states=speed_states,
        stage_speed_upper_idx=stage_speed_upper_idx,
        vehicle=vehicle,
        safeguard=safeguard,
        energy=energy,
        track=track,
        srtsp_curve=srtsp_curve,
    )

    if precompute_mode == "parallel":
        transitions, total_valid_edges = build_transition_graph_parallel(
            context=context,
            precompute_workers=precompute_workers,
            precompute_chunk_size=precompute_chunk_size,
            mp_start_method=mp_start_method,
            show_precompute_progress=show_precompute_progress,
            precompute_progress_desc=precompute_progress_desc,
        )
    else:
        transitions, total_valid_edges = build_transition_graph_serial(
            context=context,
            precompute_chunk_size=precompute_chunk_size,
            show_precompute_progress=show_precompute_progress,
            precompute_progress_desc=precompute_progress_desc,
        )

    return {
        "stages": stages,
        "speed_states": speed_states,
        "stage_speed_upper_idx": stage_speed_upper_idx,
        "transitions": transitions,
        "total_valid_edges": total_valid_edges,
    }
