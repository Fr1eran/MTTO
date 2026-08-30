from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import signal
import tempfile
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from model.common import (
    ECC,
    calc_transition_to_speed_scalar_numba,
    min_operation_time_curve,
)
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.io_utils import format_float_token

__all__ = [
    "OptimalSpeedProfile",
    "ParallelPrecomputeExitedError",
    "DP_UPPER_SPEED_ENVELOPE_VERSION",
    "VariableSpacingDPOptimizer",
    "build_transaction_batch",
]


logger = logging.getLogger(__name__)
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

DP_UPPER_SPEED_ENVELOPE_VERSION = 1
TRANSITION_CACHE_SCHEMA_VERSION = 3
TRANSITION_CACHE_ALGORITHM_VERSION = "task-upper-speed-envelope-v1"
_CACHE_ENDPOINT_SPEED_MPS = 0.0


class OptimalSpeedProfile(TypedDict):
    pos: Sequence[float] | NDArray[np.float64]
    speed: Sequence[float] | NDArray[np.float64]
    cum_time_s: Sequence[float] | NDArray[np.float64]
    total_time: float
    total_energy: float


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


class _TransitionGraphCache(TypedDict):
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
    vehicle: VehicleInfo
    safeguard_utility: SafeGuardUtility
    ecc: ECC
    track: TrackInfo
    upper_curve_pos: NDArray[np.float64]
    upper_curve_speed: NDArray[np.float64]


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
    vehicle: VehicleInfo,
    safeguard_utility: SafeGuardUtility,
    ecc: ECC,
    track: TrackInfo,
    upper_curve_pos: NDArray[np.float64],
    upper_curve_speed: NDArray[np.float64],
) -> tuple[float, float] | None:
    if math.isclose(displacement, 0.0):
        return None

    if math.isclose(speed_k + speed_k_1, 0.0):
        return None

    acc, time = calc_transition_to_speed_scalar_numba(
        speed_k,
        speed_k_1,
        displacement,
    )

    acc_tol = 1e-9
    if acc > vehicle.max_acc + acc_tol or acc < vehicle.max_dec - acc_tol:
        return None

    sample_count = max(2, math.ceil(abs(displacement) / 10.0) + 1)
    distance_sample = np.linspace(0.0, displacement, sample_count, dtype=np.float64)
    pos_sample = distance_sample + pos_k

    # 由匀变速公式采样速度，保证与端点速度一致
    speed_sq_sample = 2.0 * acc * distance_sample + speed_k**2
    speed_sample = np.sqrt(np.maximum(speed_sq_sample, 0.0))

    upper_speed_sample = np.interp(
        pos_sample,
        upper_curve_pos,
        upper_curve_speed,
    )
    if np.any(speed_sample > upper_speed_sample):
        return None

    # 检查是否进入危险速度域
    if safeguard_utility.detect_any_danger(pos=pos_sample, speed=speed_sample):
        return None

    propulsion_energy, leviation_energy = ecc.calc_energy(
        begin_pos=pos_k,
        begin_speed=speed_k,
        acc=acc,
        distance=abs(displacement),
        direction=1 if displacement > 0 else -1,
        operation_time=time,
        vehicle=vehicle,
        track=track,
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
                    safeguard_utility=context.safeguard_utility,
                    ecc=context.ecc,
                    track=context.track,
                    upper_curve_pos=context.upper_curve_pos,
                    upper_curve_speed=context.upper_curve_speed,
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
        # Windows平台使用SIGBREAK
        sigbreak = getattr(signal, "SIGBREAK", None)
        if sigbreak is not None:
            _ = signal.signal(cast(int, sigbreak), signal.SIG_IGN)
    except ValueError, AttributeError:
        # 非主进程/不支持的平台上忽略注册错误
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


class VariableSpacingDPOptimizer:
    """
    采用动态规划算法计算磁浮列车最优运行速度曲线

    1.内层动态规划
    _solve_dp_inner 接收运行时间的拉格朗日乘子, 执行一次二维变间距动态规划,
    并返回此时的最优解。

    2.外层二分法
    在动态规划算法的外层引入二分搜索循环, 根据内层计算出的实际最优运行时间,
    动态调整运行时间乘子, 直到运行时间逼近设定的规划运行时间。

    参考文献：
    [1] 赖晴鹰, 刘军, 赵若愚, 等. 基于变间距动态规划的中高速磁悬浮列车速度曲线优化[J].
    吉林大学学报（工学版）, 2019, 49(3): 749-756.
    [2] Lai Q, Liu J, Haghani A, et al. Optimal Energy Speed Profile of Medium-Speed
    Maglev Trains Integrating the Power Supply System and Train Control System[J].
    Transportation Research Record, 2020, 2674(Compendex): 729-738.
    [3] Fu C, Sun P, Wang Q, et al. Modeling and energy-saving operation optimization
    of high-speed maglev trains[J]. Journal of Cleaner Production, 2025, 519.


    """

    _CACHE_BASE_DIR: str = "output/_dp_transition_graph_cache"
    _INITIAL_LAMBDA_TIME: float = 1e3
    _MAX_LAMBDA_TIME: float = 1e8
    _LAMBDA_EXPANSION_FACTOR: float = 2.0

    @staticmethod
    def _cancel_parallel_futures(
        executor: ProcessPoolExecutor | None,
        futures: list[Future[TransitionBatchResult]],
    ) -> None:
        for future in futures:
            _ = future.cancel()

        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __init__(
        self,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        delta_speed: float = 0.1,
        max_outer_iterations: int = 100,
        show_precompute_progress: bool = True,
        precompute_progress_desc: str = "状态转移图预计算",
        precompute_mode: Literal["serial", "parallel"] = "serial",
        precompute_workers: int | None = None,
        precompute_chunk_size: int | None = None,
        mp_start_method: str | None = None,
        stage_division: Literal["variable", "uniform"] = "uniform",
        uniform_step_size: float = 30.0,
        sub_stage_count: int = 30,
        skip_disk_cache: bool = False,
    ) -> None:
        resolved_delta_speed = float(delta_speed)
        if not math.isfinite(resolved_delta_speed) or resolved_delta_speed <= 0.0:
            raise ValueError("delta_speed must be a finite positive number")
        if max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be >= 1")
        if precompute_mode not in ("serial", "parallel"):
            raise ValueError("precompute_mode must be 'serial' or 'parallel'")
        if precompute_workers is not None and precompute_workers < 1:
            raise ValueError("precompute_workers must be >= 1")
        if precompute_chunk_size is not None and precompute_chunk_size < 1:
            raise ValueError("precompute_chunk_size must be >= 1")
        if mp_start_method is None and os.name == "nt":
            mp_start_method = "spawn"
        if stage_division not in ("variable", "uniform"):
            raise ValueError("stage_division must be 'variable' or 'uniform'")
        if uniform_step_size <= 0.0:
            raise ValueError("uniform_step_size must be > 0")
        if sub_stage_count < 1:
            raise ValueError("sub_stage_count must be >= 1")

        self.vehicle: VehicleInfo = vehicle
        self.track: TrackInfo = track
        self.safeguard_utility: SafeGuardUtility = safeguard_utility
        self.train_service: TrainService = train_service
        self.delta_speed: float = resolved_delta_speed
        self.max_outer_iterations: int = int(max_outer_iterations)
        self.show_precompute_progress: bool = show_precompute_progress
        self.precompute_progress_desc: str = precompute_progress_desc
        self.precompute_mode: Literal["serial", "parallel"] = precompute_mode
        self.precompute_workers: int | None = precompute_workers
        self.precompute_chunk_size: int | None = precompute_chunk_size
        self.mp_start_method: str | None = mp_start_method
        self.stage_division: Literal["variable", "uniform"] = stage_division
        self.uniform_step_size: float = uniform_step_size
        self.sub_stage_count: int = sub_stage_count
        self.skip_disk_cache: bool = skip_disk_cache
        self.ecc: ECC = ECC(
            R_m=0.2796,
            L_d=0.0002,
            R_k=50.0,
            L_k=0.000142,
            Tau=0.258,
            Psi_fd=3.9629,
            k_c=0.8,
        )
        upper_curve_pos, upper_curve_speed = min_operation_time_curve(
            vehicle=self.vehicle,
            track=self.track,
            factor=float(self.safeguard_utility.gamma),
            begin_pos=float(self.train_service.start_position),
            begin_speed=0.0,
            end_pos=float(self.train_service.target_position)
            + float(self.train_service.max_stop_error) * 20.0,
            end_speed=0.0,
        )
        upper_curve_pos = np.asarray(upper_curve_pos, dtype=np.float64)
        upper_curve_speed = np.maximum(
            np.asarray(upper_curve_speed, dtype=np.float64), 0.0
        )
        if upper_curve_pos.size == 0 or upper_curve_speed.size != upper_curve_pos.size:
            raise ValueError("minimum-operation-time upper curve is invalid")
        if upper_curve_pos[0] > upper_curve_pos[-1]:
            upper_curve_pos = upper_curve_pos[::-1].copy()
            upper_curve_speed = upper_curve_speed[::-1].copy()
        if np.any(np.diff(upper_curve_pos) <= 0.0):
            raise ValueError(
                "minimum-operation-time upper curve positions must be "
                + "strictly increasing"
            )
        upper_curve_pos.flags.writeable = False
        upper_curve_speed.flags.writeable = False
        self.upper_curve_pos: NDArray[np.float64] = upper_curve_pos
        self.upper_curve_speed: NDArray[np.float64] = upper_curve_speed
        speed_limits = np.asarray(
            self.safeguard_utility.speed_limits, dtype=np.float64
        )
        if speed_limits.size == 0 or not np.all(np.isfinite(speed_limits)):
            raise ValueError("safeguard speed limits must be finite and non-empty")
        self.speed_grid_upper_mps: float = min(
            float(self.vehicle.max_speed),
            float(np.max(speed_limits)) * float(self.safeguard_utility.gamma),
            float(np.max(self.upper_curve_speed)),
        )
        if (
            not math.isfinite(self.speed_grid_upper_mps)
            or self.speed_grid_upper_mps <= 0.0
        ):
            raise ValueError("derived DP speed-grid upper bound must be positive")
        self._graph_cache_signature: tuple[object, ...] | None = None
        self._graph_cache: _TransitionGraphCache | None = None

    def _get_stage_speed_upper_indices(
        self, stages: NDArray[np.float64], speed_states: NDArray[np.float64]
    ) -> NDArray[np.int_]:
        """根据线路限速与任务相关最短运行时间包络生成阶段速度上界。"""
        speed_limits = self.safeguard_utility.speed_limits
        speed_limit_intervals = self.safeguard_utility.speed_limit_intervals
        if speed_limits.size == 0 or speed_limit_intervals.size == 0:
            raise ValueError("safeguard utility must provide speed limits")

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
            min(self.speed_grid_upper_mps, float(self.vehicle.max_speed)),
            speed_limits[interval_indices] * float(self.safeguard_utility.gamma),
        )
        task_upper_speed = np.interp(
            stages,
            self.upper_curve_pos,
            self.upper_curve_speed,
        )
        stage_speed_upper = np.minimum(stage_speed_upper, task_upper_speed)
        stage_speed_upper = np.maximum(stage_speed_upper, 0.0)
        upper_idx = np.searchsorted(speed_states, stage_speed_upper, side="right") - 1
        return np.clip(upper_idx, -1, len(speed_states) - 1).astype(np.int_)

    def _build_transition_graph(
        self,
        *,
        stages: NDArray[np.float64],
        speed_states: NDArray[np.float64],
        stage_speed_upper_idx: NDArray[np.int_],
    ) -> _TransitionGraphCache:
        """
        预计算状态转移图（可行性/能耗/时间）, 供外层不同lambda复用。
        """
        context = _TransitionBuildContext(
            stages=stages,
            speed_states=speed_states,
            stage_speed_upper_idx=stage_speed_upper_idx,
            vehicle=self.vehicle,
            safeguard_utility=self.safeguard_utility,
            ecc=self.ecc,
            track=self.track,
            upper_curve_pos=self.upper_curve_pos,
            upper_curve_speed=self.upper_curve_speed,
        )

        if self.precompute_mode == "parallel":
            transitions, total_valid_edges = self._build_transition_graph_parallel(
                context=context
            )
        else:
            transitions, total_valid_edges = self._build_transition_graph_serial(
                context=context
            )

        return {
            "stages": stages,
            "speed_states": speed_states,
            "stage_speed_upper_idx": stage_speed_upper_idx,
            "transitions": transitions,
            "total_valid_edges": total_valid_edges,
        }

    def _build_transition_graph_serial(
        self,
        *,
        context: _TransitionBuildContext,
    ) -> tuple[list[list[TransitionPayload | None]], int]:
        total_steps = len(context.stages) - 1
        num_speed_states = len(context.speed_states)
        transitions: list[list[TransitionPayload | None]] = [
            [None for _ in range(num_speed_states)] for _ in range(total_steps)
        ]
        total_valid_edges = 0

        chunk_size = self.precompute_chunk_size or max(1, total_steps)
        task_ranges = self._make_task_ranges(total_steps, chunk_size)
        progress_bar = None
        if self.show_precompute_progress and tqdm is not None:
            progress_bar = tqdm(
                total=total_steps,
                desc=self.precompute_progress_desc,
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
                self._merge_transition_batch(transitions, batch_rows)
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

    @staticmethod
    def _make_task_ranges(total_steps: int, chunk_size: int) -> list[tuple[int, int]]:
        if total_steps <= 0:
            return []
        return [
            (k_start, min(k_start + chunk_size, total_steps))
            for k_start in range(0, total_steps, chunk_size)
        ]

    @staticmethod
    def _merge_transition_batch(
        transitions: list[list[TransitionPayload | None]],
        batch_rows: SparseTransitionRows,
    ) -> None:
        for k_idx, row_entries in batch_rows:
            for i, next_idx, delta_energy, delta_time in row_entries:
                transitions[k_idx][i] = (next_idx, delta_energy, delta_time)

    def _resolve_parallel_config(self, total_steps: int) -> tuple[int, int]:
        workers = self.precompute_workers
        if workers is None:
            workers = max(1, (os.cpu_count() or 1) - 1)

        chunk_size = self.precompute_chunk_size
        if chunk_size is None:
            chunk_size = max(1, (total_steps + workers * 4 - 1) // (workers * 4))

        return workers, chunk_size

    def _build_transition_graph_parallel(
        self,
        *,
        context: _TransitionBuildContext,
    ) -> tuple[list[list[TransitionPayload | None]], int]:
        total_steps = len(context.stages) - 1
        num_speed_states = len(context.speed_states)
        workers, chunk_size = self._resolve_parallel_config(total_steps)
        task_ranges = self._make_task_ranges(total_steps, chunk_size)
        if workers <= 1 or total_steps < 2:
            logger.info("并行预计算条件不满足，自动回退串行模式。")
            return self._build_transition_graph_serial(context=context)
        if len(task_ranges) <= 1:
            logger.info("并行预计算任务过少，自动回退串行模式。")
            return self._build_transition_graph_serial(context=context)

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
        if self.show_precompute_progress and tqdm is not None:
            progress_bar = tqdm(
                total=total_steps,
                desc=f"{self.precompute_progress_desc}(并行)",
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
                mp.get_context(self.mp_start_method)
                if self.mp_start_method is not None
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

                self._merge_transition_batch(transitions, batch_rows)

                if progress_bar is not None:
                    _ = progress_bar.update(batch_steps)
        except KeyboardInterrupt:
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            logger.info("检测到 Ctrl+C，正在终止并行预计算任务...")
            if cancel_event is not None:
                cancel_event.set()
            self._cancel_parallel_futures(
                executor=executor,
                futures=futures,
            )
            shutdown_called = True
            raise
        except Exception as exc:
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            logger.exception("并行预计算失败，准备终止动态规划。")
            if cancel_event is not None:
                cancel_event.set()
            self._cancel_parallel_futures(
                executor=executor,
                futures=futures,
            )
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

    def _prepare_transition_graph_cache(
        self, start_position: float, target_position: float
    ) -> _TransitionGraphCache:
        """Prepare or reuse the graph for one position interval.

        The graph deliberately has fixed zero-speed cache endpoints.  Endpoint
        speeds are selected by ``optimize`` from the same grid and therefore do
        not change the graph topology or its cache key.
        """
        stages = np.asarray(
            self._generate_stages(start_position, target_position), dtype=np.float64
        )
        speed_states = self._build_speed_states()
        stage_speed_upper_idx = self._get_stage_speed_upper_indices(
            stages, speed_states
        )
        content_hash = self._compute_cache_input_hash(
            stages=stages,
            speed_states=speed_states,
            stage_speed_upper_idx=stage_speed_upper_idx,
            start_position=start_position,
            target_position=target_position,
        )
        cache_signature = (
            float(start_position),
            float(target_position),
            self.speed_grid_upper_mps,
            self.delta_speed,
            self.stage_division,
            self.sub_stage_count,
            self.uniform_step_size,
            content_hash,
        )

        if self._graph_cache_signature == cache_signature:
            assert self._graph_cache is not None
            return self._graph_cache

        if not self.skip_disk_cache:
            cached = self._load_transition_graph_from_disk(
                content_hash=content_hash,
                expected_stages=stages,
                expected_speed_states=speed_states,
                expected_stage_speed_upper_idx=stage_speed_upper_idx,
                start_position=start_position,
                target_position=target_position,
            )
            if cached is not None:
                self._graph_cache = cached
                self._graph_cache_signature = cache_signature
                return cached

        logger.info("正在预计算状态转移图（仅首次或参数变化时执行）...")
        if self.show_precompute_progress and tqdm is None:
            logger.info("未检测到 tqdm，已回退为普通循环输出。")
        logger.info("预计算执行模式: %s", self.precompute_mode)
        try:
            graph_cache = self._build_transition_graph(
                stages=stages,
                speed_states=speed_states,
                stage_speed_upper_idx=stage_speed_upper_idx,
            )
        except KeyboardInterrupt:
            logger.info("检测到 Ctrl+C，预计算已终止。")
            raise
        except ParallelPrecomputeExitedError:
            logger.info("并行预计算流程已退出，动态规划将终止。")
            raise

        self._graph_cache = graph_cache
        self._graph_cache_signature = cache_signature
        logger.info(
            "转移图预计算完成: 可行转移边数量 %s",
            graph_cache["total_valid_edges"],
        )

        if not self.skip_disk_cache:
            self._save_transition_graph_to_disk(
                graph_cache=graph_cache,
                start_position=start_position,
                target_position=target_position,
                content_hash=content_hash,
            )

        return graph_cache

    def _build_speed_states(self) -> NDArray[np.float64]:
        """Build a stable grid bounded by the reachable route speed."""
        state_count = (
            int(
                math.floor(
                    self.speed_grid_upper_mps / self.delta_speed + 1e-12
                )
            )
            + 1
        )
        speed_states = np.arange(state_count, dtype=np.float64) * self.delta_speed
        speed_states = speed_states[
            speed_states
            <= self.speed_grid_upper_mps
            + self._speed_tolerance(self.speed_grid_upper_mps)
        ]
        if speed_states.size == 0:
            raise ValueError("speed grid must contain at least the zero state")
        speed_states[0] = 0.0
        return speed_states

    @staticmethod
    def _speed_tolerance(speed: float) -> float:
        return max(1e-9, abs(float(speed)) * 1e-9)

    def _validate_task_parameters(
        self,
        *,
        start_position: float,
        start_speed: float,
        target_position: float,
        target_speed: float,
        schedule_time: float,
    ) -> tuple[float, float, float, float, float]:
        try:
            values = tuple(
                float(value)
                for value in (
                    start_position,
                    start_speed,
                    target_position,
                    target_speed,
                    schedule_time,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("optimization task parameters must be numeric") from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError("optimization task parameters must be finite")

        start_pos, start_v, target_pos, target_v, target_time = values
        if start_v < 0.0 or target_v < 0.0:
            raise ValueError("endpoint speeds must be non-negative")
        if target_time <= 0.0:
            raise ValueError("schedule_time must be positive")
        if math.isclose(start_pos, target_pos, abs_tol=1e-9, rel_tol=0.0):
            raise ValueError("start_position and target_position must differ")

        return start_pos, start_v, target_pos, target_v, target_time

    def _resolve_speed_state_index(
        self,
        speed: float,
        speed_states: NDArray[np.float64],
        *,
        name: str,
    ) -> int:
        tolerance = self._speed_tolerance(self.speed_grid_upper_mps)
        if speed < -tolerance or speed > self.speed_grid_upper_mps + tolerance:
            raise ValueError(
                f"{name}={speed:g} is outside the configured speed grid range"
            )

        insertion_idx = int(np.searchsorted(speed_states, speed, side="left"))
        candidates = {
            max(0, min(insertion_idx, len(speed_states) - 1)),
            max(0, min(insertion_idx - 1, len(speed_states) - 1)),
        }
        for candidate in candidates:
            if abs(float(speed_states[candidate]) - speed) <= tolerance:
                return candidate

        raise ValueError(
            f"{name}={speed:g} is not representable on the configured speed grid "
            f"(delta_speed={self.delta_speed:g})"
        )

    def _generate_variable_spacing_stages(
        self, start_position: float, target_position: float
    ) -> NDArray[np.float64]:
        """Generate monotonic stages split at dangerous-region intersections."""
        low = min(start_position, target_position)
        high = max(start_position, target_position)
        dangerous_points = np.asarray(
            self.safeguard_utility.get_intersecting_dangerous_point(),
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
                self.sub_stage_count + 1,
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

    def _generate_uniform_spacing_stages(
        self, start_position: float, target_position: float
    ) -> NDArray[np.float64]:
        """Generate monotonic stages with no interval longer than the step size."""
        num_steps = max(
            1,
            int(
                math.ceil(
                    abs(target_position - start_position) / self.uniform_step_size
                )
            ),
        )
        return np.linspace(
            start_position,
            target_position,
            num_steps + 1,
            dtype=np.float64,
        )

    def _generate_stages(
        self, start_position: float, target_position: float
    ) -> NDArray[np.float64]:
        """Dispatch stage generation according to the configured division mode."""
        if self.stage_division == "uniform":
            return self._generate_uniform_spacing_stages(
                start_position, target_position
            )
        return self._generate_variable_spacing_stages(start_position, target_position)

    @staticmethod
    def _hash_value(hasher: hashlib._Hash, name: str, value: object) -> None:
        hasher.update(f"{name}={value!r}\0".encode())

    @staticmethod
    def _hash_array(
        hasher: hashlib._Hash,
        name: str,
        values: Sequence[object] | NDArray[np.generic],
    ) -> None:
        array = np.ascontiguousarray(np.asarray(values))
        hasher.update(f"{name}|dtype={array.dtype.str}|shape={array.shape}\0".encode())
        hasher.update(array.tobytes())
        hasher.update(b"\0")

    def _compute_cache_input_hash(
        self,
        *,
        stages: NDArray[np.float64],
        speed_states: NDArray[np.float64],
        stage_speed_upper_idx: NDArray[np.int_],
        start_position: float,
        target_position: float,
    ) -> str:
        """Hash every input that can change a transition graph."""
        hasher = hashlib.sha256()
        self._hash_value(
            hasher, "cache_schema_version", TRANSITION_CACHE_SCHEMA_VERSION
        )
        self._hash_value(
            hasher, "algorithm_version", TRANSITION_CACHE_ALGORITHM_VERSION
        )
        self._hash_value(hasher, "cache_start_speed", _CACHE_ENDPOINT_SPEED_MPS)
        self._hash_value(hasher, "cache_target_speed", _CACHE_ENDPOINT_SPEED_MPS)
        self._hash_value(hasher, "start_position", float(start_position))
        self._hash_value(hasher, "target_position", float(target_position))
        self._hash_value(
            hasher, "speed_grid_upper_mps", self.speed_grid_upper_mps
        )
        self._hash_value(hasher, "delta_speed", self.delta_speed)
        self._hash_value(hasher, "stage_division", self.stage_division)
        self._hash_value(hasher, "sub_stage_count", self.sub_stage_count)
        self._hash_value(hasher, "uniform_step_size", self.uniform_step_size)

        self._hash_array(hasher, "stages", stages)
        self._hash_array(hasher, "speed_states", speed_states)
        self._hash_array(hasher, "stage_speed_upper_idx", stage_speed_upper_idx)
        vehicle = self.vehicle
        for name in (
            "mass",
            "numoftrainsets",
            "length",
            "max_speed",
            "max_acc",
            "max_dec",
            "max_slope_capacity",
            "levi_power_per_mass",
        ):
            self._hash_value(hasher, f"vehicle.{name}", getattr(vehicle, name))

        ecc = self.ecc
        for name in (
            "R_m",
            "L_d",
            "R_k",
            "L_k",
            "Tau",
            "Psi_fd",
            "k_c",
            "Phi_1",
            "Phi_2",
        ):
            self._hash_value(hasher, f"ecc.{name}", getattr(ecc, name))

        safeguard = self.safeguard_utility
        self._hash_array(hasher, "safeguard.speed_limits", safeguard.speed_limits)
        self._hash_array(
            hasher,
            "safeguard.speed_limit_intervals",
            safeguard.speed_limit_intervals,
        )
        self._hash_value(hasher, "safeguard.gamma", safeguard.gamma)
        for list_name in (
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        ):
            curves = getattr(safeguard, list_name)
            self._hash_value(hasher, f"safeguard.{list_name}.count", len(curves))
            for index, curve in enumerate(curves):
                self._hash_array(hasher, f"safeguard.{list_name}.{index}", curve)

        self._hash_array(hasher, "track.slopes", self.track.slopes)
        self._hash_array(hasher, "track.slope_intervals", self.track.slope_intervals)
        self._hash_array(hasher, "track.speed_limits", self.track.speed_limits)
        self._hash_array(
            hasher,
            "track.speed_limit_intervals",
            self.track.speed_limit_intervals,
        )
        return hasher.hexdigest()

    def _make_cache_folder_name(self, content_hash: str) -> str:
        """Generate a versioned, readable cache directory name."""
        delta_token = format_float_token(self.delta_speed)
        speed_upper_token = format_float_token(self.speed_grid_upper_mps)
        hash_prefix = content_hash[:16]
        if self.stage_division == "uniform":
            div_token = f"uni{format_float_token(self.uniform_step_size)}"
        else:
            div_token = f"var{self.sub_stage_count}"
        return (
            f"v{TRANSITION_CACHE_SCHEMA_VERSION}_{div_token}_"
            f"{speed_upper_token}_{delta_token}_{hash_prefix}"
        )

    def _get_disk_cache_dir(self, content_hash: str) -> Path:
        return Path(self._CACHE_BASE_DIR) / self._make_cache_folder_name(content_hash)

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        """Replace a cache file atomically, leaving the old file on failure."""
        file_descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                dir=path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "wb") as handle:
                file_descriptor = None
                _ = handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def _save_transition_graph_to_disk(
        self,
        *,
        graph_cache: _TransitionGraphCache,
        start_position: float,
        target_position: float,
        content_hash: str,
    ) -> None:
        cache_dir = self._get_disk_cache_dir(content_hash)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("无法创建缓存目录 %s: %s", cache_dir, exc)
            return

        graph_path = cache_dir / "graph_data.pkl.gz"
        meta_path = cache_dir / "metadata.json"
        try:
            graph_bytes = gzip.compress(
                pickle.dumps(graph_cache, protocol=5), compresslevel=5
            )
            self._atomic_write_bytes(graph_path, graph_bytes)
        except Exception as exc:
            logger.warning("写入缓存文件失败 (%s): %s", graph_path, exc)
            return

        metadata = {
            "cache_schema_version": TRANSITION_CACHE_SCHEMA_VERSION,
            "algorithm_version": TRANSITION_CACHE_ALGORITHM_VERSION,
            "dp_upper_speed_envelope_version": DP_UPPER_SPEED_ENVELOPE_VERSION,
            "content_hash": content_hash,
            "file_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "stage_division": self.stage_division,
            "sub_stage_count": self.sub_stage_count,
            "uniform_step_size": self.uniform_step_size,
            "speed_grid_upper_mps": self.speed_grid_upper_mps,
            "delta_speed": self.delta_speed,
            "start_position": float(start_position),
            "target_position": float(target_position),
            "cache_start_speed": _CACHE_ENDPOINT_SPEED_MPS,
            "cache_target_speed": _CACHE_ENDPOINT_SPEED_MPS,
            "num_stages": int(len(graph_cache["stages"])),
            "num_speed_states": int(len(graph_cache["speed_states"])),
            "total_valid_edges": int(graph_cache["total_valid_edges"]),
            "created_at": datetime.now().isoformat(),
        }
        try:
            metadata_bytes = json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            self._atomic_write_bytes(meta_path, metadata_bytes)
        except Exception as exc:
            logger.warning("写入缓存元数据失败 (%s): %s", meta_path, exc)

    @staticmethod
    def _metadata_float_matches(
        metadata: dict[str, object], name: str, expected: float
    ) -> bool:
        value = metadata.get(name)
        if not isinstance(value, (int, float)):
            return False
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)

    @staticmethod
    def _validate_transition_graph(
        graph_cache: object,
        *,
        expected_stages: NDArray[np.float64],
        expected_speed_states: NDArray[np.float64],
        expected_stage_speed_upper_idx: NDArray[np.int_],
    ) -> tuple[bool, str]:
        """Validate cached graph shape, values, and sparse edge payloads."""
        try:
            if not isinstance(graph_cache, dict):
                return False, "root is not a dictionary"
            required_keys = {
                "stages",
                "speed_states",
                "stage_speed_upper_idx",
                "transitions",
                "total_valid_edges",
            }
            if not required_keys.issubset(graph_cache):
                return False, "required graph keys are missing"

            stages = graph_cache["stages"]
            speed_states = graph_cache["speed_states"]
            upper_idx = graph_cache["stage_speed_upper_idx"]
            transitions = graph_cache["transitions"]
            total_valid_edges = graph_cache["total_valid_edges"]
            if not all(
                isinstance(array, np.ndarray)
                for array in (stages, speed_states, upper_idx)
            ):
                return False, "grid fields are not numpy arrays"
            if stages.ndim != 1 or speed_states.ndim != 1 or upper_idx.ndim != 1:
                return False, "grid fields must be one-dimensional"
            if not np.issubdtype(upper_idx.dtype, np.integer):
                return False, "upper-bound indices are not integral"
            if not np.array_equal(stages, expected_stages):
                return False, "stage grid does not match the cache key"
            if not np.array_equal(speed_states, expected_speed_states):
                return False, "speed grid does not match the cache key"
            if not np.array_equal(upper_idx, expected_stage_speed_upper_idx):
                return False, "stage speed bounds do not match the cache key"
            if not np.all(np.isfinite(stages)) or not np.all(np.isfinite(speed_states)):
                return False, "grid contains non-finite values"
            if speed_states.size == 0 or not math.isclose(
                float(speed_states[0]), 0.0, abs_tol=1e-9
            ):
                return False, "speed grid does not start at zero"
            if speed_states.size > 1 and not np.all(np.diff(speed_states) > 0.0):
                return False, "speed grid is not strictly increasing"
            if stages.size < 2:
                return False, "stage grid has fewer than two points"
            stage_diff = np.diff(stages)
            if not (np.all(stage_diff > 0.0) or np.all(stage_diff < 0.0)):
                return False, "stage grid is not strictly monotonic"
            if np.any(upper_idx < -1) or np.any(upper_idx >= speed_states.size):
                return False, "stage speed bounds are outside the speed grid"
            if not isinstance(transitions, list) or len(transitions) != stages.size - 1:
                return False, "transition stage count does not match the grid"
            if not isinstance(total_valid_edges, (int, np.integer)):
                return False, "total_valid_edges is not integral"
            if int(total_valid_edges) < 0:
                return False, "total_valid_edges is negative"

            counted_edges = 0
            for stage_index, rows in enumerate(transitions):
                if not isinstance(rows, list) or len(rows) != speed_states.size:
                    return False, f"transition row {stage_index} has wrong width"
                for speed_index, transition in enumerate(rows):
                    if transition is None:
                        continue
                    if (
                        not isinstance(transition, (tuple, list))
                        or len(transition) != 3
                    ):
                        return False, "transition payload has wrong shape"
                    next_indices, delta_energy, delta_time = transition
                    if not all(
                        isinstance(array, np.ndarray)
                        and array.ndim == 1
                        and np.issubdtype(array.dtype, np.number)
                        for array in (next_indices, delta_energy, delta_time)
                    ):
                        return False, "transition payload arrays are invalid"
                    if not np.issubdtype(next_indices.dtype, np.integer):
                        return False, "transition indices are not integral"
                    if not (len(next_indices) == len(delta_energy) == len(delta_time)):
                        return False, "transition payload lengths differ"
                    if next_indices.size > 1 and not np.all(np.diff(next_indices) > 0):
                        return False, "transition indices are not strictly increasing"
                    if np.any(next_indices < 0) or np.any(
                        next_indices >= speed_states.size
                    ):
                        return False, "transition index is outside the speed grid"
                    if not np.all(np.isfinite(delta_energy)) or not np.all(
                        np.isfinite(delta_time)
                    ):
                        return False, "transition payload contains non-finite values"
                    if np.any(delta_time <= 0.0):
                        return False, "transition time must be positive"
                    if speed_index > int(upper_idx[stage_index]):
                        return False, "transition exists above its stage bound"
                    counted_edges += int(next_indices.size)

            if counted_edges != int(total_valid_edges):
                return False, "total_valid_edges does not match payloads"
        except Exception as exc:
            return False, f"validation raised {type(exc).__name__}: {exc}"

        return True, ""

    def _load_transition_graph_from_disk(
        self,
        *,
        content_hash: str,
        expected_stages: NDArray[np.float64],
        expected_speed_states: NDArray[np.float64],
        expected_stage_speed_upper_idx: NDArray[np.int_],
        start_position: float,
        target_position: float,
    ) -> _TransitionGraphCache | None:
        cache_dir = self._get_disk_cache_dir(content_hash)
        graph_path = cache_dir / "graph_data.pkl.gz"
        meta_path = cache_dir / "metadata.json"
        if not graph_path.is_file() or not meta_path.is_file():
            return None

        try:
            metadata_value = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(metadata_value, dict):
                raise ValueError("metadata root is not a dictionary")
            metadata: dict[str, object] = metadata_value
        except Exception as exc:
            logger.warning("缓存元数据损坏，将重新计算 (%s): %s", meta_path, exc)
            return None

        expected_ints = {
            "cache_schema_version": TRANSITION_CACHE_SCHEMA_VERSION,
            "sub_stage_count": self.sub_stage_count,
            "num_stages": len(expected_stages),
            "num_speed_states": len(expected_speed_states),
        }
        for name, expected in expected_ints.items():
            if metadata.get(name) != expected:
                logger.warning("缓存元数据 %s 不匹配，将重新计算 (%s)", name, meta_path)
                return None
        if metadata.get("algorithm_version") != TRANSITION_CACHE_ALGORITHM_VERSION:
            logger.warning("缓存算法版本不匹配，将重新计算 (%s)", meta_path)
            return None
        if metadata.get("content_hash") != content_hash:
            logger.warning("缓存参数签名不匹配，将重新计算 (%s)", meta_path)
            return None
        if metadata.get("stage_division") != self.stage_division:
            logger.warning("缓存阶段划分配置不匹配，将重新计算 (%s)", meta_path)
            return None
        if (
            not self._metadata_float_matches(
                metadata, "uniform_step_size", self.uniform_step_size
            )
            or not self._metadata_float_matches(
                metadata, "speed_grid_upper_mps", self.speed_grid_upper_mps
            )
            or not self._metadata_float_matches(
                metadata, "delta_speed", self.delta_speed
            )
            or not self._metadata_float_matches(
                metadata, "start_position", start_position
            )
            or not self._metadata_float_matches(
                metadata, "target_position", target_position
            )
            or not self._metadata_float_matches(
                metadata, "cache_start_speed", _CACHE_ENDPOINT_SPEED_MPS
            )
            or not self._metadata_float_matches(
                metadata, "cache_target_speed", _CACHE_ENDPOINT_SPEED_MPS
            )
        ):
            logger.warning("缓存任务参数不匹配，将重新计算 (%s)", meta_path)
            return None

        expected_file_hash = metadata.get("file_sha256")
        if not isinstance(expected_file_hash, str):
            logger.warning("缓存文件缺少完整性校验值，将重新计算 (%s)", meta_path)
            return None
        try:
            graph_bytes = graph_path.read_bytes()
        except OSError as exc:
            logger.warning("读取缓存文件失败 (%s): %s", graph_path, exc)
            return None
        if hashlib.sha256(graph_bytes).hexdigest() != expected_file_hash:
            logger.warning("缓存文件完整性校验失败 (%s)，将重新计算", graph_path)
            return None

        try:
            graph_cache: object = pickle.loads(gzip.decompress(graph_bytes))
        except Exception as exc:
            logger.warning("缓存文件反序列化失败 (%s): %s", graph_path, exc)
            return None

        valid, reason = self._validate_transition_graph(
            graph_cache,
            expected_stages=expected_stages,
            expected_speed_states=expected_speed_states,
            expected_stage_speed_upper_idx=expected_stage_speed_upper_idx,
        )
        if not valid:
            logger.warning("缓存结构校验失败，将重新计算 (%s): %s", graph_path, reason)
            return None
        typed_graph_cache = cast(_TransitionGraphCache, graph_cache)
        if metadata.get("total_valid_edges") != typed_graph_cache["total_valid_edges"]:
            logger.warning("缓存边数量元数据不匹配，将重新计算 (%s)", meta_path)
            return None

        logger.info(
            "从磁盘缓存加载状态转移图: %s (%s 条可行转移边)",
            cache_dir.name,
            typed_graph_cache["total_valid_edges"],
        )
        return typed_graph_cache

    def _solve_dp_inner(
        self,
        *,
        cache: _TransitionGraphCache,
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> OptimalSpeedProfile | None:
        """Solve one Lagrangian DP problem on an already-built graph."""
        if not math.isfinite(lambda_time) or lambda_time < 0.0:
            raise ValueError("lambda_time must be a finite non-negative number")

        stages = cache["stages"]
        speed_states = cache["speed_states"]
        stage_speed_upper_idx = cache["stage_speed_upper_idx"]
        transitions = cache["transitions"]
        total_steps = len(stages) - 1
        num_speed_states = len(speed_states)

        if not (0 <= start_state_idx < num_speed_states):
            raise ValueError("start_state_idx is outside the speed grid")
        if not (0 <= target_state_idx < num_speed_states):
            raise ValueError("target_state_idx is outside the speed grid")
        if start_state_idx > int(stage_speed_upper_idx[0]):
            return None
        if target_state_idx > int(stage_speed_upper_idx[-1]):
            return None

        # Only two value rows are needed.  The policy retains one next-state
        # index per stage/state so the selected trajectory can be reconstructed.
        next_cost = np.full(num_speed_states, np.inf, dtype=np.float64)
        next_time = np.full(num_speed_states, np.inf, dtype=np.float64)
        next_cost[target_state_idx] = 0.0
        next_time[target_state_idx] = 0.0
        policy = np.full((total_steps, num_speed_states), -1, dtype=np.int_)

        for stage_index in range(total_steps - 1, -1, -1):
            current_cost = np.full(num_speed_states, np.inf, dtype=np.float64)
            current_time = np.full(num_speed_states, np.inf, dtype=np.float64)
            current_upper = min(
                int(stage_speed_upper_idx[stage_index]), num_speed_states - 1
            )
            if current_upper < 0:
                next_cost, current_cost = current_cost, next_cost
                next_time, current_time = current_time, next_time
                continue

            for speed_index in range(current_upper + 1):
                transition = transitions[stage_index][speed_index]
                if transition is None:
                    continue

                next_indices, delta_energy, delta_time = transition
                successor_cost = next_cost[next_indices]
                finite_mask = np.isfinite(successor_cost)
                if not np.any(finite_mask):
                    continue

                valid_next_indices = next_indices[finite_mask]
                valid_delta_energy = delta_energy[finite_mask]
                valid_delta_time = delta_time[finite_mask]
                valid_successor_cost = successor_cost[finite_mask]
                candidate_cost = (
                    valid_delta_energy
                    + lambda_time * valid_delta_time
                    + valid_successor_cost
                )
                best_local_index = int(np.argmin(candidate_cost))
                best_next_index = int(valid_next_indices[best_local_index])
                current_cost[speed_index] = float(candidate_cost[best_local_index])
                current_time[speed_index] = float(
                    valid_delta_time[best_local_index] + next_time[best_next_index]
                )
                policy[stage_index, speed_index] = best_next_index

            next_cost, current_cost = current_cost, next_cost
            next_time, current_time = current_time, next_time

        if not math.isfinite(float(next_cost[start_state_idx])):
            return None

        optimal_speed_indices = np.empty(total_steps + 1, dtype=np.int_)
        optimal_speed_indices[0] = start_state_idx
        cum_time_s = np.zeros(total_steps + 1, dtype=np.float64)
        total_energy = 0.0
        current_speed_idx = start_state_idx
        for stage_index in range(total_steps):
            next_speed_idx = int(policy[stage_index, current_speed_idx])
            if next_speed_idx < 0:
                return None
            transition = transitions[stage_index][current_speed_idx]
            if transition is None:
                return None
            next_indices, delta_energy, delta_time = transition
            local_index = int(np.searchsorted(next_indices, next_speed_idx))
            if (
                local_index >= len(next_indices)
                or int(next_indices[local_index]) != next_speed_idx
            ):
                return None
            total_energy += float(delta_energy[local_index])
            cum_time_s[stage_index + 1] = cum_time_s[stage_index] + float(
                delta_time[local_index]
            )
            optimal_speed_indices[stage_index + 1] = next_speed_idx
            current_speed_idx = next_speed_idx

        if current_speed_idx != target_state_idx:
            return None
        total_time = float(cum_time_s[-1])
        return {
            "pos": stages.tolist(),
            "speed": speed_states[optimal_speed_indices].tolist(),
            "cum_time_s": cum_time_s.tolist(),
            "total_time": total_time,
            "total_energy": float(total_energy),
        }

    def optimize(
        self,
        start_pos: float,
        start_speed: float,
        target_pos: float,
        target_speed: float,
        schedule_time: float,
    ) -> OptimalSpeedProfile | None:
        """Find the minimum-energy trajectory within the schedule tolerance."""
        (
            start_position,
            initial_speed,
            target_position,
            final_speed,
            target_time,
        ) = self._validate_task_parameters(
            start_position=start_pos,
            start_speed=start_speed,
            target_position=target_pos,
            target_speed=target_speed,
            schedule_time=schedule_time,
        )
        speed_states = self._build_speed_states()
        start_state_idx = self._resolve_speed_state_index(
            initial_speed,
            speed_states,
            name="start_speed",
        )
        target_state_idx = self._resolve_speed_state_index(
            final_speed,
            speed_states,
            name="target_speed",
        )
        cache = self._prepare_transition_graph_cache(
            start_position=start_position,
            target_position=target_position,
        )
        time_tolerance_s = float(self.train_service.max_arr_time_error_s)

        if start_state_idx > int(cache["stage_speed_upper_idx"][0]):
            logger.warning("起点速度超过安全包络，无法构造可行轨迹。")
            return None
        if target_state_idx > int(cache["stage_speed_upper_idx"][-1]):
            logger.warning("终点速度超过安全包络，无法构造可行轨迹。")
            return None

        logger.info(
            "开始双层寻优: 目标时间 %.2fs, 时间误差阈值 %.2fs, 速度网格 %.3fm/s",
            target_time,
            time_tolerance_s,
            self.delta_speed,
        )

        best_result: OptimalSpeedProfile | None = None
        best_error = math.inf
        best_energy = math.inf

        def evaluate(lambda_value: float) -> OptimalSpeedProfile | None:
            nonlocal best_result, best_error, best_energy
            result = self._solve_dp_inner(
                cache=cache,
                lambda_time=lambda_value,
                start_state_idx=start_state_idx,
                target_state_idx=target_state_idx,
            )
            if result is None:
                logger.debug("lambda=%.6g 未找到可行轨迹", lambda_value)
                return None

            total_time = float(result["total_time"])
            total_energy = float(result["total_energy"])
            if not math.isfinite(total_time) or not math.isfinite(total_energy):
                logger.warning("lambda=%.6g 返回了非有限的 DP 结果。", lambda_value)
                return None
            time_error = abs(total_time - target_time)
            is_better = time_error < best_error - 1e-12 or (
                math.isclose(time_error, best_error, abs_tol=1e-12, rel_tol=0.0)
                and total_energy < best_energy
            )
            if is_better:
                best_result = result
                best_error = time_error
                best_energy = total_energy
            logger.debug(
                "lambda=%.6g, time=%.6fs, energy=%.6f, error=%.6fs",
                lambda_value,
                total_time,
                total_energy,
                time_error,
            )
            return result

        low_lambda = 0.0
        low_result = evaluate(low_lambda)
        if low_result is None:
            logger.warning("lambda=0 未找到可行轨迹。")
            return best_result
        low_time = float(low_result["total_time"])
        if best_error < time_tolerance_s:
            logger.info("lambda=0 已满足准点阈值，误差 %.6fs。", best_error)
            return best_result
        if low_time < target_time:
            logger.warning(
                "能耗最优轨迹已快于目标时间，无法用非负 lambda 覆盖目标区间；"
                "返回最小时间误差结果。"
            )
            return best_result

        high_lambda = min(self._INITIAL_LAMBDA_TIME, self._MAX_LAMBDA_TIME)
        high_result = evaluate(high_lambda)
        if high_result is None:
            logger.warning(
                "lambda=%.6g 未找到可行轨迹；返回最小时间误差结果。",
                high_lambda,
            )
            return best_result
        high_time = float(high_result["total_time"])

        while high_time > target_time and high_lambda < self._MAX_LAMBDA_TIME:
            next_lambda = min(
                self._MAX_LAMBDA_TIME,
                high_lambda * self._LAMBDA_EXPANSION_FACTOR,
            )
            if next_lambda <= high_lambda:
                break
            high_lambda = next_lambda
            next_result = evaluate(high_lambda)
            if next_result is None:
                break
            high_result = next_result
            high_time = float(high_result["total_time"])

        if not (low_time >= target_time and high_time <= target_time):
            logger.warning(
                "lambda 搜索未能形成跨越目标时间 %.6fs 的区间（当前 %.6f~%.6fs）；"
                "返回最小时间误差结果。",
                target_time,
                low_time,
                high_time,
            )
            return best_result

        for iteration in range(self.max_outer_iterations):
            if best_error < time_tolerance_s:
                break
            mid_lambda = (low_lambda + high_lambda) / 2.0
            if mid_lambda <= low_lambda or mid_lambda >= high_lambda:
                break
            result = evaluate(mid_lambda)
            if result is None:
                logger.warning("lambda=%.6g 未找到可行轨迹，停止二分。", mid_lambda)
                break

            mid_time = float(result["total_time"])
            logger.debug(
                "lambda 二分迭代 %s/%s: %.6g -> %.6fs",
                iteration + 1,
                self.max_outer_iterations,
                mid_lambda,
                mid_time,
            )
            if mid_time > target_time:
                low_lambda = mid_lambda
                low_time = mid_time
            else:
                high_lambda = mid_lambda
                high_time = mid_time

        if best_result is not None:
            if best_error < time_tolerance_s:
                logger.info("双层寻优收敛，最小时间误差 %.6fs。", best_error)
            else:
                logger.warning(
                    "双层寻优达到迭代/搜索边界，最小时间误差 %.6fs。",
                    best_error,
                )
        return best_result
