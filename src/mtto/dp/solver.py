"""Dynamic programming solver and lambda search for MTTO speed profiles."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import segment_energy
from mtto.domain.kinematics import accel_between
from mtto.domain.line import Line
from mtto.domain.safeguard import Safeguard
from mtto.domain.scenario import Scenario, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.domain.srtsp import (
    SrtspCurve,
    build_srtsp_curve,
    min_operation_time_curve,
)
from mtto.dp.cache import (
    compute_cache_input_hash,
    load_transition_graph_from_disk,
    save_transition_graph_to_disk,
)
from mtto.dp.graph import (
    TransitionGraph,
    build_speed_states,
    build_transition_graph,
    generate_stages,
    get_stage_speed_upper_indices,
    speed_tolerance,
)

__all__ = [
    "VariableSpacingDPOptimizer",
    "_DPSolution",
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass(frozen=True, slots=True)
class _DPSolution:
    pos: NDArray[np.float64]
    speed: NDArray[np.float64]
    cum_time_s: NDArray[np.float64]
    total_energy: float


class VariableSpacingDPOptimizer:
    """采用动态规划算法计算磁浮列车最优运行速度曲线。

    1.内层动态规划
    _solve_dp_inner 接收运行时间的拉格朗日乘子, 执行一次二维变间距动态规划,
    并返回此时的最优解。

    2.外层二分法
    在动态规划算法的外层引入二分搜索循环, 根据内层计算出的实际最优运行时间,
    动态调整运行时间乘子, 直到运行时间逼近设定的规划运行时间。
    """

    _INITIAL_LAMBDA_TIME: float = 1e3
    _MAX_LAMBDA_TIME: float = 1e8
    _LAMBDA_EXPANSION_FACTOR: float = 2.0

    def __init__(
        self,
        scenario: Scenario,
        task: Task,
        cache_dir: str | Path | None,
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
    ) -> None:
        if task.schedule_time_s is None:
            raise ValueError("task.schedule_time_s must not be None")
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
        if stage_division not in ("variable", "uniform"):
            raise ValueError("stage_division must be 'variable' or 'uniform'")
        if uniform_step_size <= 0.0:
            raise ValueError("uniform_step_size must be > 0")
        if sub_stage_count < 1:
            raise ValueError("sub_stage_count must be >= 1")

        self.scenario: Scenario = scenario
        self.vehicle: Vehicle = scenario.vehicle
        self.track: Line = scenario.line
        self.safeguard: Safeguard = scenario.safeguard
        self.task: Task = task
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
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

        raw_pos, raw_speed = min_operation_time_curve(
            vehicle=self.vehicle,
            track=self.track,
            factor=float(self.safeguard.params.factor),
            begin_pos=float(self.task.start_position_m),
            begin_speed=0.0,
            end_pos=float(self.task.target_position_m)
            + float(self.task.max_stop_error_m) * 20.0,
            end_speed=0.0,
        )
        self.srtsp_curve: SrtspCurve = build_srtsp_curve(raw_pos, raw_speed)
        speed_limits = np.asarray(self.safeguard.speed_limits, dtype=np.float64)
        if speed_limits.size == 0 or not np.all(np.isfinite(speed_limits)):
            raise ValueError("safeguard speed limits must be finite and non-empty")
        self.speed_grid_upper_mps: float = min(
            float(self.vehicle.max_speed),
            float(np.max(speed_limits)) * float(self.safeguard.params.factor),
            float(np.max(self.srtsp_curve.speed_mps)),
        )
        if (
            not math.isfinite(self.speed_grid_upper_mps)
            or self.speed_grid_upper_mps <= 0.0
        ):
            raise ValueError("derived DP speed-grid upper bound must be positive")
        self._graph_cache_signature: tuple[object, ...] | None = None
        self._graph_cache: TransitionGraph | None = None

    def _graph_inputs(
        self, start_position: float, target_position: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_], str]:
        """Stages, speed grid, stage upper indices and cache key for one interval."""
        stages = np.asarray(
            generate_stages(
                stage_division=self.stage_division,
                sub_stage_count=self.sub_stage_count,
                uniform_step_size=self.uniform_step_size,
                start_position=start_position,
                target_position=target_position,
                safeguard=self.safeguard,
            ),
            dtype=np.float64,
        )
        speed_states = build_speed_states(
            speed_grid_upper_mps=self.speed_grid_upper_mps,
            delta_speed=self.delta_speed,
        )
        stage_speed_upper_idx = get_stage_speed_upper_indices(
            safeguard=self.safeguard,
            srtsp_curve=self.srtsp_curve,
            speed_grid_upper_mps=self.speed_grid_upper_mps,
            vehicle_max_speed=float(self.vehicle.max_speed),
            stages=stages,
            speed_states=speed_states,
        )
        content_hash = compute_cache_input_hash(
            stages=stages,
            speed_states=speed_states,
            stage_speed_upper_idx=stage_speed_upper_idx,
            start_position=start_position,
            target_position=target_position,
            speed_grid_upper_mps=self.speed_grid_upper_mps,
            delta_speed=self.delta_speed,
            stage_division=self.stage_division,
            sub_stage_count=self.sub_stage_count,
            uniform_step_size=self.uniform_step_size,
            vehicle=self.vehicle,
            energy=self.scenario.energy,
            safeguard=self.safeguard,
            track=self.track,
            scenario_hash=self.scenario.scenario_hash,
            task_max_stop_error_m=float(self.task.max_stop_error_m),
        )
        return stages, speed_states, stage_speed_upper_idx, content_hash

    def _prepare_transition_graph_cache(
        self, start_position: float, target_position: float
    ) -> TransitionGraph:
        stages, speed_states, stage_speed_upper_idx, content_hash = self._graph_inputs(
            start_position, target_position
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

        cache_folder = dict(
            stage_division=self.stage_division,
            sub_stage_count=self.sub_stage_count,
            uniform_step_size=self.uniform_step_size,
            speed_grid_upper_mps=self.speed_grid_upper_mps,
            delta_speed=self.delta_speed,
        )
        if self.cache_dir is not None:
            cached = load_transition_graph_from_disk(
                cache_base_dir=self.cache_dir,
                content_hash=content_hash,
                expected_stages=stages,
                expected_speed_states=speed_states,
                expected_stage_speed_upper_idx=stage_speed_upper_idx,
                start_position=start_position,
                target_position=target_position,
                **cache_folder,
            )
            if cached is not None:
                self._graph_cache = cached
                self._graph_cache_signature = cache_signature
                return cached

        logger.info("正在预计算状态转移图（仅首次或参数变化时执行）...")
        if self.show_precompute_progress and tqdm is None:
            logger.info("未检测到 tqdm，已回退为普通循环输出。")
        logger.info("预计算执行模式: %s", self.precompute_mode)

        graph_cache = build_transition_graph(
            stages=stages,
            speed_states=speed_states,
            stage_speed_upper_idx=stage_speed_upper_idx,
            vehicle=self.vehicle,
            safeguard=self.safeguard,
            energy=self.scenario.energy,
            track=self.track,
            srtsp_curve=self.srtsp_curve,
            precompute_mode=self.precompute_mode,
            precompute_workers=self.precompute_workers,
            precompute_chunk_size=self.precompute_chunk_size,
            mp_start_method=self.mp_start_method,
            show_precompute_progress=self.show_precompute_progress,
            precompute_progress_desc=self.precompute_progress_desc,
        )

        self._graph_cache = graph_cache
        self._graph_cache_signature = cache_signature
        logger.info(
            "转移图预计算完成: 可行转移边数量 %s",
            graph_cache["total_valid_edges"],
        )

        if self.cache_dir is not None:
            save_transition_graph_to_disk(
                cache_base_dir=self.cache_dir,
                graph_cache=graph_cache,
                start_position=start_position,
                target_position=target_position,
                content_hash=content_hash,
                **cache_folder,
            )

        return graph_cache

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
        tolerance = speed_tolerance(self.speed_grid_upper_mps)
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

    def _solve_dp_inner(
        self,
        *,
        cache: TransitionGraph,
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> _DPSolution | None:
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
        return _DPSolution(
            pos=stages,
            speed=speed_states[optimal_speed_indices],
            cum_time_s=cum_time_s,
            total_energy=float(total_energy),
        )

    def _to_speed_profile(self, solution: _DPSolution) -> SpeedProfile:
        position = solution.pos
        speed = solution.speed
        time = solution.cum_time_s

        n_nodes = position.size
        prop_deltas = np.empty(n_nodes - 1, dtype=np.float64)
        levi_deltas = np.empty(n_nodes - 1, dtype=np.float64)
        for k in range(n_nodes - 1):
            displacement = float(position[k + 1] - position[k])
            acc, duration = accel_between(
                float(speed[k]), float(speed[k + 1]), displacement
            )
            prop, levi = segment_energy(
                self.scenario.energy,
                self.vehicle,
                self.track,
                begin_pos=float(position[k]),
                begin_speed=float(speed[k]),
                acc=acc,
                distance=abs(displacement),
                direction=1 if displacement > 0 else -1,
                operation_time=duration,
            )
            prop_deltas[k] = prop
            levi_deltas[k] = levi

        propulsion_energy = np.concatenate(([0.0], np.cumsum(prop_deltas)))
        levitation_energy = np.concatenate(([0.0], np.cumsum(levi_deltas)))

        return SpeedProfile.from_arrays(
            position_m=position,
            speed_mps=speed,
            time_s=time,
            propulsion_energy_kj=propulsion_energy,
            levitation_energy_kj=levitation_energy,
        )

    def _search_optimal_solution(
        self,
        start_pos: float,
        start_speed: float,
        target_pos: float,
        target_speed: float,
        schedule_time: float,
    ) -> _DPSolution | None:
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
        speed_states = build_speed_states(
            speed_grid_upper_mps=self.speed_grid_upper_mps,
            delta_speed=self.delta_speed,
        )
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
        time_tolerance_s = float(self.task.max_arr_time_error_s)

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

        best_result: _DPSolution | None = None
        best_error = math.inf
        best_energy = math.inf

        def evaluate(lambda_value: float) -> _DPSolution | None:
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

            total_time = result.cum_time_s[-1]
            total_energy = float(result.total_energy)
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
        low_time = low_result.cum_time_s[-1]
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
        high_time = high_result.cum_time_s[-1]

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
            high_time = high_result.cum_time_s[-1]

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

            mid_time = result.cum_time_s[-1]
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

    def optimize(
        self,
        start_pos: float,
        start_speed: float,
        target_pos: float,
        target_speed: float,
        schedule_time: float,
    ) -> SpeedProfile | None:
        """Find the minimum-energy trajectory within the schedule tolerance."""
        solution = self._search_optimal_solution(
            start_pos=start_pos,
            start_speed=start_speed,
            target_pos=target_pos,
            target_speed=target_speed,
            schedule_time=schedule_time,
        )
        if solution is None:
            return None
        return self._to_speed_profile(solution)
