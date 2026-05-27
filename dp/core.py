from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import signal
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from model.common import ECC, ORS
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.io_utils import format_float_token

__all__ = [
    "OptimalSpeedProfile",
    "ParallelPrecomputeExitedError",
    "VariableSpacingDPOptimizer",
]


class OptimalSpeedProfile(TypedDict):
    pos: Sequence[float] | NDArray
    speed: Sequence[float] | NDArray
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

_DP_PARALLEL_CONTEXT: dict[str, Any] | None = None


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
) -> tuple[bool, float, float]:
    if math.isclose(displacement, 0.0):
        return False, np.inf, np.inf

    if math.isclose(speed_k + speed_k_1, 0.0):
        return False, np.inf, np.inf

    # 匀变速模型: a = (v1^2 - v0^2) / (2 * ds)
    acc = (speed_k_1**2 - speed_k**2) / (2.0 * displacement)

    acc_tol = 1e-9
    if acc > vehicle.max_acc + acc_tol or acc < vehicle.max_dec - acc_tol:
        return False, np.inf, np.inf

    sample_count = max(10, int(np.abs(displacement) / 10.0))
    distance_sample = np.linspace(0.0, displacement, sample_count, dtype=np.float64)
    pos_sample = distance_sample + pos_k

    # 由匀变速公式采样速度，保证与端点速度一致
    speed_sq_sample = 2.0 * acc * distance_sample + speed_k**2
    speed_sample = np.sqrt(np.maximum(speed_sq_sample, 0.0))

    # 检查是否进入危险速度域
    if safeguard_utility.detect_any_danger(pos=pos_sample, speed=speed_sample):
        return False, np.inf, np.inf

    if np.abs(acc) < 1e-9:
        time = np.abs(displacement) / speed_k
    else:
        time = (speed_k_1 - speed_k) / acc

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

    return True, propulsion_energy + leviation_energy, time


def _compute_transition_batch(
    *,
    k_start: int,
    k_end: int,
    stages: NDArray[np.float64],
    speed_states: NDArray[np.float64],
    stage_speed_upper_idx: NDArray[np.int_],
    vehicle: VehicleInfo,
    safeguard_utility: SafeGuardUtility,
    ecc: ECC,
    track: TrackInfo,
) -> TransitionBatchResult:
    batch_rows: SparseTransitionRows = []
    total_valid_edges = 0

    for k_idx in range(k_start, k_end):
        pos_k = float(stages[k_idx])
        delta_pos = float(stages[k_idx + 1] - stages[k_idx])
        abs_delta_pos = abs(delta_pos)
        current_upper = int(stage_speed_upper_idx[k_idx])
        next_upper = int(stage_speed_upper_idx[k_idx + 1])

        if current_upper < 0 or next_upper < 0:
            continue

        row_entries: list[SparseTransitionEntry] = []

        for i in range(current_upper + 1):
            speed_k = float(speed_states[i])

            # 基于加减速度物理边界的下一阶段速度索引剪枝
            v2_min = max(speed_k**2 + 2.0 * vehicle.max_dec * abs_delta_pos, 0.0)
            v2_max = max(speed_k**2 + 2.0 * vehicle.max_acc * abs_delta_pos, 0.0)
            v_next_min = np.sqrt(v2_min)
            v_next_max = np.sqrt(v2_max)

            j_min = int(np.searchsorted(speed_states, v_next_min, side="left"))
            j_max = int(np.searchsorted(speed_states, v_next_max, side="right") - 1)
            j_max = min(j_max, next_upper)

            if j_min > j_max:
                continue

            next_indices: list[int] = []
            delta_energy_list: list[float] = []
            delta_time_list: list[float] = []

            for j in range(j_min, j_max + 1):
                speed_next = float(speed_states[j])
                is_valid, delta_energy, delta_time = _calculate_transition_with_context(
                    pos_k=pos_k,
                    speed_k=speed_k,
                    displacement=delta_pos,
                    speed_k_1=speed_next,
                    vehicle=vehicle,
                    safeguard_utility=safeguard_utility,
                    ecc=ecc,
                    track=track,
                )
                if not is_valid:
                    continue

                next_indices.append(j)
                delta_energy_list.append(delta_energy)
                delta_time_list.append(delta_time)

            if not next_indices:
                continue

            row_entries.append((
                i,
                np.asarray(next_indices, dtype=np.int_),
                np.asarray(delta_energy_list, dtype=np.float64),
                np.asarray(delta_time_list, dtype=np.float64),
            ))
            total_valid_edges += len(next_indices)

        if row_entries:
            batch_rows.append((k_idx, row_entries))

    return batch_rows, total_valid_edges, k_end - k_start


def _init_transition_worker(context: dict[str, Any]) -> None:
    global _DP_PARALLEL_CONTEXT
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Windows平台使用SIGBREAK
        sigbreak = getattr(signal, "SIGBREAK", None)
        if sigbreak is not None:
            signal.signal(sigbreak, signal.SIG_IGN)
    except ValueError, AttributeError:
        # 非主进程/不支持的平台上忽略注册错误
        pass
    _DP_PARALLEL_CONTEXT = context


def _compute_transition_batch_worker(k_start: int, k_end: int) -> TransitionBatchResult:
    if _DP_PARALLEL_CONTEXT is None:
        raise RuntimeError("worker context is not initialized")

    return _compute_transition_batch(
        k_start=k_start,
        k_end=k_end,
        stages=_DP_PARALLEL_CONTEXT["stages"],
        speed_states=_DP_PARALLEL_CONTEXT["speed_states"],
        stage_speed_upper_idx=_DP_PARALLEL_CONTEXT["stage_speed_upper_idx"],
        vehicle=_DP_PARALLEL_CONTEXT["vehicle"],
        safeguard_utility=_DP_PARALLEL_CONTEXT["safeguard_utility"],
        ecc=_DP_PARALLEL_CONTEXT["ecc"],
        track=_DP_PARALLEL_CONTEXT["track"],
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

    @staticmethod
    def _cancel_parallel_futures(
        executor: ProcessPoolExecutor | None,
        futures: list[Any],
        *,
        force_terminate_running: bool = False,
    ) -> None:
        for future in futures:
            future.cancel()

        if executor is not None:
            if force_terminate_running:
                VariableSpacingDPOptimizer._force_terminate_executor_processes(executor)
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _force_terminate_executor_processes(executor: ProcessPoolExecutor) -> None:
        # ProcessPoolExecutor 没有公开 API 直接终止正在运行的任务，这里在 Ctrl+C 场景下
        # 使用私有进程句柄进行强制终止，确保并行预计算能尽快停止。
        processes = getattr(executor, "_processes", None)
        if not isinstance(processes, dict):
            return

        for process in list(processes.values()):
            if process is None:
                continue
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                continue

        for process in list(processes.values()):
            if process is None:
                continue
            try:
                process.join(timeout=0.5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=0.5)
            except Exception:
                continue

    def __init__(
        self,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        show_precompute_progress: bool = True,
        precompute_progress_desc: str = "状态转移图预计算",
        precompute_mode: Literal["serial", "parallel"] = "serial",
        precompute_workers: int | None = None,
        precompute_chunk_size: int | None = None,
        mp_start_method: str | None = None,
        stage_division: Literal["variable", "uniform"] = "variable",
        uniform_step_size: float = 10.0,
        sub_stage_count: int = 30,
        skip_disk_cache: bool = False,
    ) -> None:
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

        self.vehicle = vehicle
        self.track = track
        self.safeguard_utility = safeguard_utility
        self.train_service = train_service
        self.show_precompute_progress = show_precompute_progress
        self.precompute_progress_desc = precompute_progress_desc
        self.precompute_mode = precompute_mode
        self.precompute_workers = precompute_workers
        self.precompute_chunk_size = precompute_chunk_size
        self.mp_start_method = mp_start_method
        self.stage_division = stage_division
        self.uniform_step_size = uniform_step_size
        self.sub_stage_count = sub_stage_count
        self.skip_disk_cache = skip_disk_cache
        self.ecc = ECC(
            R_m=0.2796,
            L_d=0.0002,
            R_k=50.0,
            L_k=0.000142,
            Tau=0.258,
            Psi_fd=3.9629,
            k_c=0.8,
        )
        self.ors = ORS(
            vehicle=self.vehicle,
            track=self.track,
            factor=self.safeguard_utility.gamma,
        )
        # 计算最短运行时间参考曲线
        self.ref_curve_pos, self.ref_curve_speed = (
            self.ors.calc_min_operation_time_curve(
                begin_pos=self.train_service.start_position,
                begin_speed=self.train_service.start_speed,
                end_pos=self.train_service.target_position,
                end_speed=0.0,
            )
        )
        self._graph_cache_signature: (
            tuple[float, float, str, int, float, str] | None
        ) = None
        self._graph_cache: dict[str, Any] | None = None

    def _get_ref_speed(self, pos: float) -> float:
        return max(0.0, np.interp(pos, self.ref_curve_pos, self.ref_curve_speed))

    def _get_stage_speed_upper_indices(
        self, stages: NDArray[np.float64], speed_states: NDArray[np.float64]
    ) -> NDArray[np.int_]:
        """预插值参考速度上界, 避免在DP内层重复插值。"""
        ref_speed_at_stage = np.maximum(
            0.0, np.interp(stages, self.ref_curve_pos, self.ref_curve_speed)
        )
        upper_idx = np.searchsorted(speed_states, ref_speed_at_stage, side="right") - 1
        return np.clip(upper_idx, -1, len(speed_states) - 1).astype(np.int_)

    def _build_transition_graph(
        self, stages: NDArray[np.float64], speed_states: NDArray[np.float64]
    ) -> dict[str, Any]:
        """
        预计算状态转移图（可行性/能耗/时间）, 供外层不同lambda复用。
        """
        total_steps = len(stages) - 1
        num_speed_states = len(speed_states)
        stage_speed_upper_idx = self._get_stage_speed_upper_indices(
            stages, speed_states
        )

        if self.precompute_mode == "parallel":
            transitions, total_valid_edges = self._build_transition_graph_parallel(
                stages=stages,
                speed_states=speed_states,
                stage_speed_upper_idx=stage_speed_upper_idx,
                total_steps=total_steps,
                num_speed_states=num_speed_states,
            )
        else:
            transitions, total_valid_edges = self._build_transition_graph_serial(
                stages=stages,
                speed_states=speed_states,
                stage_speed_upper_idx=stage_speed_upper_idx,
                total_steps=total_steps,
                num_speed_states=num_speed_states,
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
        stages: NDArray[np.float64],
        speed_states: NDArray[np.float64],
        stage_speed_upper_idx: NDArray[np.int_],
        total_steps: int,
        num_speed_states: int,
    ) -> tuple[list[list[TransitionPayload | None]], int]:
        transitions: list[list[TransitionPayload | None]] = [
            [None for _ in range(num_speed_states)] for _ in range(total_steps)
        ]
        total_valid_edges = 0

        step_iter = self._iter_precompute_steps(total_steps)
        try:
            for k in step_iter:
                k_idx = int(k)
                pos_k = float(stages[k_idx])
                delta_pos = float(stages[k_idx + 1] - stages[k_idx])
                abs_delta_pos = abs(delta_pos)
                current_upper = int(stage_speed_upper_idx[k_idx])
                next_upper = int(stage_speed_upper_idx[k_idx + 1])

                if current_upper < 0 or next_upper < 0:
                    continue

                for i in range(current_upper + 1):
                    speed_k = float(speed_states[i])

                    # 基于加减速度物理边界的下一阶段速度索引剪枝
                    v2_min = max(
                        speed_k**2 + 2.0 * self.vehicle.max_dec * abs_delta_pos,
                        0.0,
                    )
                    v2_max = max(
                        speed_k**2 + 2.0 * self.vehicle.max_acc * abs_delta_pos,
                        0.0,
                    )
                    v_next_min = np.sqrt(v2_min)
                    v_next_max = np.sqrt(v2_max)

                    j_min = int(np.searchsorted(speed_states, v_next_min, side="left"))
                    j_max = int(
                        np.searchsorted(speed_states, v_next_max, side="right") - 1
                    )
                    j_max = min(j_max, next_upper)

                    if j_min > j_max:
                        continue

                    next_indices: list[int] = []
                    delta_energy_list: list[float] = []
                    delta_time_list: list[float] = []

                    for j in range(j_min, j_max + 1):
                        speed_next = float(speed_states[j])
                        is_valid, delta_energy, delta_time = (
                            _calculate_transition_with_context(
                                pos_k=pos_k,
                                speed_k=speed_k,
                                displacement=delta_pos,
                                speed_k_1=speed_next,
                                vehicle=self.vehicle,
                                safeguard_utility=self.safeguard_utility,
                                ecc=self.ecc,
                                track=self.track,
                            )
                        )
                        if not is_valid:
                            continue

                        next_indices.append(j)
                        delta_energy_list.append(delta_energy)
                        delta_time_list.append(delta_time)

                    if not next_indices:
                        continue

                    transitions[k_idx][i] = (
                        np.asarray(next_indices, dtype=np.int_),
                        np.asarray(delta_energy_list, dtype=np.float64),
                        np.asarray(delta_time_list, dtype=np.float64),
                    )
                    total_valid_edges += len(next_indices)
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在终止串行预计算任务...")
            raise
        finally:
            close_fn = getattr(step_iter, "close", None)
            if callable(close_fn):
                close_fn()

        return transitions, total_valid_edges

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
        stages: NDArray[np.float64],
        speed_states: NDArray[np.float64],
        stage_speed_upper_idx: NDArray[np.int_],
        total_steps: int,
        num_speed_states: int,
    ) -> tuple[list[list[TransitionPayload | None]], int]:
        workers, chunk_size = self._resolve_parallel_config(total_steps)
        if workers <= 1 or total_steps < 2:
            raise ParallelPrecomputeExitedError(
                "并行预计算条件不满足（workers<=1 或 total_steps<2）。"
            )

        task_ranges = [
            (k_start, min(k_start + chunk_size, total_steps))
            for k_start in range(0, total_steps, chunk_size)
        ]
        if len(task_ranges) <= 1:
            raise ParallelPrecomputeExitedError("并行预计算任务过少，无法有效并行。")

        print(
            "并行预计算配置: "
            f"workers={workers}, chunk_size={chunk_size}, tasks={len(task_ranges)}"
        )

        transitions: list[list[TransitionPayload | None]] = [
            [None for _ in range(num_speed_states)] for _ in range(total_steps)
        ]
        total_valid_edges = 0

        worker_context: dict[str, Any] = {
            "stages": stages,
            "speed_states": speed_states,
            "stage_speed_upper_idx": stage_speed_upper_idx,
            "vehicle": self.vehicle,
            "safeguard_utility": self.safeguard_utility,
            "ecc": self.ecc,
            "track": self.track,
        }

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
        futures: list[Any] = []
        shutdown_called = False

        try:
            mp_context = (
                mp.get_context(self.mp_start_method)
                if self.mp_start_method is not None
                else mp.get_context()
            )
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                initializer=_init_transition_worker,
                initargs=(worker_context,),
            )
            futures = [
                executor.submit(_compute_transition_batch_worker, k_start, k_end)
                for k_start, k_end in task_ranges
            ]
            for future in as_completed(futures):
                batch_rows, batch_valid_edges, batch_steps = future.result()
                total_valid_edges += batch_valid_edges

                for k_idx, row_entries in batch_rows:
                    for i, next_idx, delta_energy, delta_time in row_entries:
                        transitions[k_idx][i] = (
                            next_idx,
                            delta_energy,
                            delta_time,
                        )

                if progress_bar is not None:
                    progress_bar.update(batch_steps)
        except KeyboardInterrupt:
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            print("\n检测到 Ctrl+C，正在终止并行预计算任务...")
            self._cancel_parallel_futures(
                executor=executor,
                futures=futures,
                force_terminate_running=True,
            )
            shutdown_called = True
            raise
        except Exception as exc:
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            print(f"并行预计算失败，准备终止动态规划。原因: {exc}")
            self._cancel_parallel_futures(
                executor=executor,
                futures=futures,
                force_terminate_running=True,
            )
            shutdown_called = True
            raise ParallelPrecomputeExitedError(f"并行预计算异常退出: {exc}") from exc
        finally:
            if progress_bar is not None:
                progress_bar.close()
            if executor is not None and not shutdown_called:
                executor.shutdown(wait=True, cancel_futures=False)

        return transitions, total_valid_edges

    def _iter_precompute_steps(self, total_steps: int) -> Iterable[int]:
        if self.show_precompute_progress and tqdm is not None:
            return tqdm(
                range(total_steps),
                total=total_steps,
                desc=self.precompute_progress_desc,
                dynamic_ncols=True,
                unit="stage",
                mininterval=0.2,
            )
        return range(total_steps)

    def _prepare_transition_graph_cache(
        self, max_speed: float, delta_speed: float
    ) -> dict[str, Any]:
        stages = self._generate_stages().astype(np.float64)
        speed_states = np.arange(0, max_speed, delta_speed, dtype=np.float64)
        stage_speed_upper_idx = self._get_stage_speed_upper_indices(
            stages, speed_states
        )
        content_hash = self._compute_cache_input_hash(
            stages, speed_states, stage_speed_upper_idx
        )
        cache_signature = (
            float(max_speed),
            float(delta_speed),
            self.stage_division,
            self.sub_stage_count,
            self.uniform_step_size,
            content_hash,
        )

        if self._graph_cache_signature == cache_signature:
            assert self._graph_cache is not None
            return self._graph_cache

        # 1) 尝试从磁盘加载
        if not self.skip_disk_cache:
            cached = self._load_transition_graph_from_disk(delta_speed, content_hash)
            if cached is not None:
                self._graph_cache = cached
                self._graph_cache_signature = cache_signature
                return self._graph_cache

        # 2) 重新计算
        print("正在预计算状态转移图（仅首次或参数变化时执行）...")
        if self.show_precompute_progress and tqdm is None:
            print("未检测到 tqdm，已回退为普通循环输出。")
        print(f"预计算执行模式: {self.precompute_mode}")
        try:
            self._graph_cache = self._build_transition_graph(stages, speed_states)
        except KeyboardInterrupt:
            print("检测到 Ctrl+C，预计算已终止。")
            raise
        except ParallelPrecomputeExitedError:
            print("并行预计算流程已退出，动态规划将终止。")
            raise
        self._graph_cache_signature = cache_signature
        print(
            f"转移图预计算完成: 可行转移边数量 {self._graph_cache['total_valid_edges']}"
        )

        # 3) 写入磁盘缓存
        if not self.skip_disk_cache:
            self._save_transition_graph_to_disk(
                self._graph_cache, delta_speed, content_hash
            )

        assert self._graph_cache is not None
        return self._graph_cache

    def _generate_variable_spacing_stages(self) -> NDArray:
        """
        基于临界点的变间距阶段划分
        将线路按照临界点划分为大分区, 每个大分区等分为 sub_stage_count 个子阶段
        """
        critical_points_position_arr = np.concatenate((
            np.array([self.train_service.start_position]),
            self.safeguard_utility.get_intersecting_dangerous_point(),
            np.array([self.train_service.target_position]),
        ))
        stages = []

        for i in range(len(critical_points_position_arr) - 1):
            interval_start_pos = critical_points_position_arr[i]
            interval_end_pos = critical_points_position_arr[i + 1]
            partition_stages = np.linspace(
                interval_start_pos, interval_end_pos, self.sub_stage_count + 1
            )
            if i == 0:
                stages.extend(partition_stages)
            else:
                stages.extend(partition_stages[1:])

        return np.array(stages)

    def _generate_uniform_spacing_stages(self) -> NDArray:
        """基于等间距的阶段划分，从起点到终点按 uniform_step_size 等分。"""
        start = self.train_service.start_position
        end = self.train_service.target_position
        num_steps = max(1, int(np.ceil((end - start) / self.uniform_step_size)))
        return np.linspace(start, end, num_steps + 1)

    def _generate_stages(self) -> NDArray:
        """根据 stage_division 配置调度对应的阶段划分方法。"""
        if self.stage_division == "uniform":
            return self._generate_uniform_spacing_stages()
        return self._generate_variable_spacing_stages()

    def _compute_cache_input_hash(
        self,
        stages: NDArray[np.float64],
        speed_states: NDArray[np.float64],
        stage_speed_upper_idx: NDArray[np.int_],
    ) -> str:
        """计算影响状态转移图的所有输入的 SHA256 哈希，用作磁盘缓存键。"""
        hasher = hashlib.sha256()

        # 离散化网格
        hasher.update(stages.tobytes())
        hasher.update(speed_states.tobytes())
        hasher.update(stage_speed_upper_idx.tobytes())

        # 车辆参数（影响加速度约束与能耗）
        v = self.vehicle
        for val in (
            v.mass,
            v.numoftrainsets,
            v.length,
            v.max_speed,
            v.max_acc,
            v.max_dec,
            v.levi_power_per_mass,
        ):
            hasher.update(str(val).encode())

        # ECC 参数（影响能耗）
        ecc = self.ecc
        for val in (
            ecc.R_m,
            ecc.L_d,
            ecc.R_k,
            ecc.L_k,
            ecc.Tau,
            ecc.Psi_fd,
            ecc.k_c,
        ):
            hasher.update(str(val).encode())

        # SafeGuard 防护曲线与限速（影响危险域检测）
        su = self.safeguard_utility
        hasher.update(su.speed_limits.tobytes())
        hasher.update(su.speed_limit_intervals.tobytes())
        hasher.update(str(su.gamma).encode())
        for curve_list in (
            su.levi_curves_list,
            su.brake_curves_list,
            su.min_curves_list,
            su.max_curves_list,
        ):
            for curve in curve_list:
                hasher.update(curve.tobytes())

        # 线路坡度（影响能耗）
        hasher.update(self.track.slopes.tobytes())
        hasher.update(self.track.slope_intervals.tobytes())

        return hasher.hexdigest()

    def _make_cache_folder_name(self, delta_speed: float, content_hash: str) -> str:
        """生成可读的缓存文件夹名，包含关键参数与前缀哈希。"""
        delta_token = format_float_token(delta_speed)
        hash_prefix = content_hash[:12]
        if self.stage_division == "uniform":
            div_token = f"uni{format_float_token(self.uniform_step_size)}"
        else:
            div_token = f"var{self.sub_stage_count}"
        return f"{div_token}_{delta_token}_{hash_prefix}"

    def _get_disk_cache_dir(self, delta_speed: float, content_hash: str) -> Path:
        return Path(self._CACHE_BASE_DIR) / self._make_cache_folder_name(
            delta_speed, content_hash
        )

    def _save_transition_graph_to_disk(
        self,
        graph_cache: dict[str, Any],
        delta_speed: float,
        content_hash: str,
    ) -> None:
        cache_dir = self._get_disk_cache_dir(delta_speed, content_hash)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"无法创建缓存目录 {cache_dir}: {exc}")
            return

        graph_path = cache_dir / "graph_data.pkl.gz"
        meta_path = cache_dir / "metadata.json"

        graph_bytes: bytes | None = None
        try:
            graph_bytes = gzip.compress(
                pickle.dumps(graph_cache, protocol=5), compresslevel=5
            )
            graph_path.write_bytes(graph_bytes)
        except Exception as exc:
            print(f"写入缓存文件失败 ({graph_path}): {exc}")
            # 清理不完整的文件
            if graph_path.exists():
                try:
                    graph_path.unlink()
                except OSError:
                    pass
            return

        file_hash = hashlib.sha256(graph_bytes).hexdigest()
        max_speed_val = float(np.max(graph_cache["speed_states"]))
        metadata = {
            "content_hash": content_hash,
            "file_sha256": file_hash,
            "stage_division": self.stage_division,
            "sub_stage_count": self.sub_stage_count,
            "uniform_step_size": self.uniform_step_size,
            "max_speed": max_speed_val,
            "delta_speed": delta_speed,
            "num_stages": int(len(graph_cache["stages"])),
            "num_speed_states": int(len(graph_cache["speed_states"])),
            "total_valid_edges": int(graph_cache["total_valid_edges"]),
            "created_at": datetime.now().isoformat(),
        }
        try:
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        except OSError as exc:
            print(f"写入缓存元数据失败 ({meta_path}): {exc}")

    def _load_transition_graph_from_disk(
        self, delta_speed: float, content_hash: str
    ) -> dict[str, Any] | None:
        cache_dir = self._get_disk_cache_dir(delta_speed, content_hash)
        graph_path = cache_dir / "graph_data.pkl.gz"
        meta_path = cache_dir / "metadata.json"

        if not graph_path.is_file() or not meta_path.is_file():
            return None

        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"缓存元数据损坏，将重新计算 ({meta_path}): {exc}")
            return None

        if metadata.get("content_hash") != content_hash:
            print("缓存参数签名不匹配（数据可能已更新），将重新计算。")
            return None

        try:
            graph_bytes = graph_path.read_bytes()
        except OSError as exc:
            print(f"读取缓存文件失败 ({graph_path}): {exc}")
            return None

        expected_file_hash = metadata.get("file_sha256", "")
        actual_file_hash = hashlib.sha256(graph_bytes).hexdigest()
        if expected_file_hash and actual_file_hash != expected_file_hash:
            print(f"缓存文件完整性校验失败 ({graph_path})，将重新计算。")
            return None

        try:
            graph_cache = pickle.loads(gzip.decompress(graph_bytes))
        except Exception as exc:
            print(f"缓存文件反序列化失败 ({graph_path}): {exc}")
            return None

        if not isinstance(graph_cache, dict) or "transitions" not in graph_cache:
            print(f"缓存文件结构异常 ({graph_path})，将重新计算。")
            return None

        print(
            f"从磁盘缓存加载状态转移图: {cache_dir.name} "
            f"({graph_cache['total_valid_edges']} 条可行转移边)"
        )
        return graph_cache

    def _solve_dp_inner(
        self, lambda_time: float, max_speed: float, delta_speed: float
    ) -> OptimalSpeedProfile | None:
        """
        逆推法求解变间距动态规划, 结合目标函数及约束
        使用拉格朗日乘子法简化时间惩罚
        """
        cache = self._prepare_transition_graph_cache(
            max_speed=max_speed, delta_speed=delta_speed
        )
        stages = cache["stages"]
        speed_states = cache["speed_states"]
        stage_speed_upper_idx = cache["stage_speed_upper_idx"]
        transitions = cache["transitions"]

        total_steps = len(stages) - 1
        num_speed_states = len(speed_states)

        # 初始化DP状态值表
        # 存储形式为 dict, 键为[状态索引], 值为[从该状态到终点状态的最小代价]
        # 代价为包含能耗以及转化为惩罚的到站时刻误差
        dp_cost = np.full(
            (total_steps + 1, num_speed_states), np.inf
        )  # 相当于状态值, 以代价的形式
        dp_time_accum = np.full(
            (total_steps + 1, num_speed_states), np.inf
        )  # 记录对应的累计时间
        dp_policy = np.full(
            (total_steps + 1, num_speed_states), -1, dtype=int
        )  # 记录状态转移动作

        # 状态初始化
        dp_cost[total_steps, 0] = 0.0
        dp_time_accum[total_steps, 0] = 0.0

        print(f"开始使用逆推法求解，共{total_steps}个阶段")

        # 逆推法
        for k in range(total_steps - 1, -1, -1):
            current_upper = int(stage_speed_upper_idx[k])
            if current_upper < 0:
                continue

            for i in range(current_upper + 1):
                transition = transitions[k][i]
                if transition is None:
                    continue

                next_indices, delta_energy_arr, delta_time_arr = transition
                next_dp_cost = dp_cost[k + 1, next_indices]
                finite_mask = np.isfinite(next_dp_cost)
                if not np.any(finite_mask):
                    continue

                valid_next_indices = next_indices[finite_mask]
                valid_delta_energy = delta_energy_arr[finite_mask]
                valid_delta_time = delta_time_arr[finite_mask]
                valid_next_dp_cost = next_dp_cost[finite_mask]

                total_cost_arr = (
                    valid_delta_energy
                    + lambda_time * valid_delta_time
                    + valid_next_dp_cost
                )
                best_local_idx = int(np.argmin(total_cost_arr))
                best_next_j = int(valid_next_indices[best_local_idx])

                dp_cost[k, i] = float(total_cost_arr[best_local_idx])
                dp_policy[k, i] = best_next_j
                dp_time_accum[k, i] = float(
                    valid_delta_time[best_local_idx] + dp_time_accum[k + 1, best_next_j]
                )

        # 检验是否找到可行解
        if np.isinf(dp_cost[0, 0]):
            return None

        # 提取最优速度曲线
        optimal_speed_indices = [0]
        current_speed_idx = 0
        for k in range(total_steps):
            next_speed_idx = dp_policy[k, current_speed_idx]
            if next_speed_idx < 0:
                return None
            optimal_speed_indices.append(next_speed_idx)
            current_speed_idx = next_speed_idx

        optimal_speed_profile = speed_states[
            np.asarray(optimal_speed_indices, dtype=int)
        ]
        total_time = dp_time_accum[0, 0]
        total_energy = dp_cost[0, 0] - lambda_time * total_time  # 剥离时间惩罚

        return {
            "pos": stages.tolist(),
            "speed": optimal_speed_profile.tolist(),
            "total_time": total_time,
            "total_energy": total_energy,
        }

    def optimize(
        self, max_speed: float, delta_speed: float, max_iters: int = 100
    ) -> OptimalSpeedProfile | None:
        """
        二分法调整运行时间乘子, 从而使结果逼近规定的运行时间
        """
        target_time = self.train_service.schedule_time
        time_tolerance_ratio = self.train_service.max_arr_time_error_ratio
        print(
            f"开始双层寻优: 目标时间为{target_time:.2f}s, \
            时间误差容忍率为 {time_tolerance_ratio * 100}%, \
            速度步长为 {delta_speed:.2f}m/s"
        )

        lambda_time = 1e3
        lambda_min = 0.0
        lambda_max = 1e8
        best_result = None

        for iteration in range(max_iters):
            print(
                f"第{iteration + 1}次迭代: 测试 lambda = {lambda_time:.2f} ...",
                end="",
            )

            # 调用内层DP
            result = self._solve_dp_inner(
                lambda_time=lambda_time,
                max_speed=max_speed,
                delta_speed=delta_speed,
            )

            if result is None:
                print("未找到可行解, 请检查代码逻辑")
                break

            total_time = result["total_time"]
            total_energy = result["total_energy"]
            print(f"实际耗时为{total_time:.2f}s, 能耗为{total_energy:.2f}kJ")

            best_result = result

            # 判断时间误差率是否满足约束
            time_error_ratio = abs(total_time - target_time) / target_time
            if time_error_ratio <= time_tolerance_ratio:
                print(
                    f"成功收敛! 运行时间误差率为{time_error_ratio:.4f}%, 满足精度要求"
                )
                break

            # 二分查找
            if total_time > target_time:
                # 耗时太长, 增大时间惩罚
                lambda_min = lambda_time
            else:
                # 耗时太短, 减小时间惩罚
                lambda_max = lambda_time

            lambda_time = (lambda_min + lambda_max) / 2.0

        if best_result is not None:
            print("最终优化结果")
            print(f"目标运行时间: {target_time:.2f}s")
            print(f"实际规划时间: {best_result['total_time']:.2f}s")
            print(f"最低运行能耗: {best_result['total_energy']:.2f}kJ")

        return best_result
