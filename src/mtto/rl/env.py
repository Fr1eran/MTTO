"""Time-step RL environment and Gym adapter."""

import math
from dataclasses import dataclass
from typing import Any, final, override

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import segment_energy
from mtto.domain.kinematics import Motion, run_time
from mtto.domain.line import Line, get_slope_scalar_numba
from mtto.domain.safeguard import (
    SPS,
    Safeguard,
    SPSState,
    ViolationKind,
    dynamic_limit_violation,
    dynamic_limits,
)
from mtto.domain.scenario import Scenario, StopState, Task
from mtto.domain.srtsp import SrtspLookup, lookup_upper_speed, min_remaining_time_s
from mtto.rl.diagnostics import (
    RewardDiagnosticsAccumulator,
    RewardDiagnosticsBatch,
    SafetyTruncationBatch,
    SafetyTruncationBuffer,
)
from mtto.rl.observation import ObservationBuilder
from mtto.rl.rewards import (
    RewardCalculator,
    RewardConfig,
    RewardNormalization,
    braking_reserve_steps,
    traction_reserve_steps,
)
from mtto.rl.state import State, StepResult, TerminationReason


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Termination state returned by one environment transition."""

    termination_reason: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {"termination_reason": self.termination_reason}


@dataclass(frozen=True, slots=True)
class EpisodeInfo:
    """Canonical episode snapshot with units encoded in field names."""

    position_m: float
    speed_mps: float
    stopping_point_index: int
    operation_time_s: float
    redundant_operation_time_s: float
    energy_consumption_j: float
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "position_m": float(self.position_m),
            "speed_mps": float(self.speed_mps),
            "stopping_point_index": self.stopping_point_index,
            "operation_time_s": float(self.operation_time_s),
            "redundant_operation_time_s": float(self.redundant_operation_time_s),
            "energy_consumption_j": float(self.energy_consumption_j),
            "comfort_tav": float(self.comfort_tav),
            "comfort_er_pct": float(self.comfort_er_pct),
            "comfort_rms": float(self.comfort_rms),
        }


@final
class MTTOEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        scenario: Scenario,
        task: Task,
        gamma: float,
        step_time_s: float,
        srtsp_lookup: SrtspLookup,
        normalization: RewardNormalization,
        compact_training_info: bool = False,
        enable_trajectory_tracking: bool = False,
        reward_config: RewardConfig | None = None,
        safety_truncation_buffer: SafetyTruncationBuffer | None = None,
        reward_diagnostics_accumulator: RewardDiagnosticsAccumulator | None = None,
    ) -> None:
        super().__init__()
        if task.schedule_time_s is None:
            raise ValueError("task.schedule_time_s must not be None")
        self.scenario = scenario
        self.vehicle: Vehicle = scenario.vehicle
        self.track: Line = scenario.line
        self.safeguard: Safeguard = scenario.safeguard
        self.task = task
        self.gamma = gamma
        self.step_time_s = float(step_time_s)
        if not math.isfinite(self.step_time_s) or self.step_time_s <= 0.0:
            raise ValueError("step_time_s must be finite and positive")
        self.srtsp_lookup = srtsp_lookup
        self.normalization = normalization
        self.sps = SPS(
            safeguard=self.safeguard,
            accessible_positions_m=self.track.accessible_points_m,
            danger_positions_m=self.track.danger_points_m,
            step_delay_s=self.safeguard.params.step_delay_s,
        )
        self.observation_builder = ObservationBuilder(
            vehicle=self.vehicle,
            track=self.track,
            task=task,
            whole_distance_m=task.target_position_m - task.start_position_m,
            initial_min_operation_time_s=normalization.initial_min_operation_time_s,
        )
        self.reward_calculator = RewardCalculator(
            normalization,
            gamma=gamma,
            reward_config=reward_config,
            step_time_s=self.step_time_s,
        )
        self.reward_config = self.reward_calculator.reward_config
        self.safety_truncation_buffer = safety_truncation_buffer
        self.reward_diagnostics_accumulator = reward_diagnostics_accumulator
        self.state = self.initial_state()
        self.observation_space = gym.spaces.Box(
            low=np.asarray(ObservationBuilder.LOW, dtype=np.float32),
            high=np.ones(ObservationBuilder.OBSERVATION_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._observation_buffer: NDArray[np.float32] = np.empty(
            self.observation_space.shape, dtype=np.float32
        )
        self.compact_training_info = bool(compact_training_info)
        self.episode_info: EpisodeInfo | None = None
        self.outcome = EpisodeOutcome(termination_reason=None)
        self._comfort_tav = 0.0
        self._comfort_sum_sq_delta_acc = 0.0
        self._comfort_exceedance_count = 0
        self.enable_trajectory_tracking = enable_trajectory_tracking
        self.trajectory_pos: list[float] | None = None
        self.trajectory_speed_mps: list[float] | None = None

    def _build_state(
        self,
        *,
        s_m: float,
        v_mps: float,
        commanded_acceleration_mps2: float,
        t_s: float,
        propulsion_energy_kj: float,
        levitation_energy_kj: float,
        step: int,
        sps: SPSState,
        schedule_time_s: float,
        schedule_changed: bool,
    ) -> State:
        slope = float(
            get_slope_scalar_numba(s_m, self.track.slopes, self.track.slope_intervals)
        )
        lower, upper = dynamic_limits(
            self.safeguard, s_m, sps.target_stopping_point_index
        )
        upper_limit = float(upper)
        srtsp_limit = lookup_upper_speed(self.srtsp_lookup, s_m)
        # Farthest position reachable within one control period.
        dt = self.step_time_s
        ahead_m = s_m + v_mps * dt + 0.5 * self.vehicle.max_acc * dt**2
        lower_ahead, upper_ahead = dynamic_limits(
            self.safeguard, ahead_m, sps.target_stopping_point_index
        )
        max_speed_ahead = min(
            lookup_upper_speed(self.srtsp_lookup, ahead_m), float(upper_ahead)
        )
        min_remaining = min_remaining_time_s(
            self.vehicle,
            self.track,
            self.safeguard.params.factor,
            s_m,
            v_mps,
            self.task.target_position_m,
        )
        return State(
            s_m=float(s_m),
            v_mps=float(v_mps),
            commanded_acceleration_mps2=float(commanded_acceleration_mps2),
            t_s=float(t_s),
            propulsion_energy_kj=float(propulsion_energy_kj),
            levitation_energy_kj=float(levitation_energy_kj),
            sps=sps,
            schedule_time_s=float(schedule_time_s),
            step=step,
            schedule_changed=bool(schedule_changed),
            slope_pct=slope,
            stop_error_m=abs(self.task.target_position_m - s_m),
            lower_limit_mps=float(lower),
            upper_limit_mps=upper_limit,
            srtsp_limit_mps=srtsp_limit,
            slack_time_s=schedule_time_s - t_s - min_remaining,
            braking_reserve_steps=braking_reserve_steps(
                v_mps,
                min(srtsp_limit, upper_limit),
                max_speed_ahead,
                self.vehicle.max_dec_abs,
                self.step_time_s,
            ),
            traction_reserve_steps=traction_reserve_steps(
                v_mps,
                float(lower),
                float(lower_ahead),
                self.vehicle.max_acc,
                self.step_time_s,
            ),
        )

    def initial_state(self) -> State:
        assert self.task.schedule_time_s is not None
        schedule_time_s = self.task.schedule_time_s
        schedule_changed = False
        if (
            self.task.schedule_change is not None
            and self.task.schedule_change.trigger_position_m
            == self.task.start_position_m
        ):
            schedule_time_s = self.task.schedule_change.new_schedule_time_s
            schedule_changed = True
        return self._build_state(
            s_m=self.task.start_position_m,
            v_mps=0.0,
            commanded_acceleration_mps2=0.0,
            t_s=0.0,
            propulsion_energy_kj=0.0,
            levitation_energy_kj=0.0,
            step=0,
            sps=self.sps.initial_state(),
            schedule_time_s=schedule_time_s,
            schedule_changed=schedule_changed,
        )

    def transition(
        self, state: State, commanded_acceleration_mps2: float
    ) -> StepResult:
        motion = Motion(
            *run_time(state.v_mps, float(commanded_acceleration_mps2), self.step_time_s)
        )
        propulsion, levitation = segment_energy(
            self.scenario.energy,
            self.vehicle,
            self.track,
            begin_pos=state.s_m,
            begin_speed=state.v_mps,
            acc=float(commanded_acceleration_mps2),
            distance=motion.distance_m,
            direction=1,
            operation_time=motion.duration_s,
        )
        position = state.s_m + motion.distance_m
        operation_time = state.t_s + motion.duration_s
        sps = self.sps.advance(
            state.sps,
            position_m=position,
            speed_mps=motion.v1_mps,
            time_s=operation_time,
        )
        step_end = self._build_state(
            s_m=position,
            v_mps=motion.v1_mps,
            commanded_acceleration_mps2=float(commanded_acceleration_mps2),
            t_s=operation_time,
            propulsion_energy_kj=state.propulsion_energy_kj + propulsion,
            levitation_energy_kj=state.levitation_energy_kj + levitation,
            step=state.step + 1,
            sps=sps,
            schedule_time_s=state.schedule_time_s,
            schedule_changed=state.schedule_changed,
        )
        violation = dynamic_limit_violation(
            step_end.s_m,
            step_end.v_mps,
            step_end.lower_limit_mps,
            step_end.upper_limit_mps,
        )
        stop_state = self.task.stop_state(step_end.s_m, step_end.v_mps)
        if stop_state == StopState.STOPPED_IN_ZONE:
            reason = TerminationReason.STOPPED_IN_ZONE
        elif stop_state == StopState.STOPPED_SHORT:
            reason = TerminationReason.STOPPED_SHORT
        elif stop_state == StopState.OVERRAN:
            reason = TerminationReason.OVERRAN
        elif violation is not None:
            reason = (
                TerminationReason.UNDER_LOWER_LIMIT
                if violation.kind == ViolationKind.UNDER_LOWER_LIMIT
                else TerminationReason.OVER_UPPER_LIMIT
            )
        elif step_end.v_mps > step_end.srtsp_limit_mps:
            reason = TerminationReason.OVER_SRTSP
        else:
            reason = None
        next_state = step_end
        if (
            reason is None
            and self.task.schedule_change is not None
            and not state.schedule_changed
            and state.s_m < self.task.schedule_change.trigger_position_m <= step_end.s_m
        ):
            next_state = self._build_state(
                s_m=step_end.s_m,
                v_mps=step_end.v_mps,
                commanded_acceleration_mps2=step_end.commanded_acceleration_mps2,
                t_s=step_end.t_s,
                propulsion_energy_kj=step_end.propulsion_energy_kj,
                levitation_energy_kj=step_end.levitation_energy_kj,
                step=step_end.step,
                sps=step_end.sps,
                schedule_time_s=self.task.schedule_change.new_schedule_time_s,
                schedule_changed=True,
            )
        return StepResult(
            step_end_state=step_end,
            next_state=next_state,
            commanded_acceleration_mps2=float(commanded_acceleration_mps2),
            motion=motion,
            propulsion_delta_kj=float(propulsion),
            levitation_delta_kj=float(levitation),
            termination_reason=reason,
            violation=violation,
        )

    def drain_safety_truncations(self) -> SafetyTruncationBatch:
        buffer = self.safety_truncation_buffer
        if buffer is None:
            raise RuntimeError("safety truncation tracking is not configured")
        return buffer.drain()

    def drain_reward_diagnostics(
        self, *, finalize: bool = False
    ) -> RewardDiagnosticsBatch:
        accumulator = self.reward_diagnostics_accumulator
        if accumulator is None:
            raise RuntimeError("reward diagnostics are not configured")
        return accumulator.drain(finalize=finalize)

    def _reset_trajectory(self) -> None:
        if not self.enable_trajectory_tracking:
            self.trajectory_pos = self.trajectory_speed_mps = None
            return
        self.trajectory_pos = [self.state.s_m]
        self.trajectory_speed_mps = [abs(self.state.v_mps)]

    def _record_trajectory(self) -> None:
        if self.enable_trajectory_tracking:
            assert (
                self.trajectory_pos is not None
                and self.trajectory_speed_mps is not None
            )
            self.trajectory_pos.append(self.state.s_m)
            self.trajectory_speed_mps.append(abs(self.state.v_mps))

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        _ = super().reset(seed=seed, options=options)
        self.state = self.initial_state()
        self.episode_info = None
        self.outcome = EpisodeOutcome(termination_reason=None)
        self._comfort_tav = self._comfort_sum_sq_delta_acc = 0.0
        self._comfort_exceedance_count = 0
        self._reset_trajectory()
        observation = self.observation_builder.build(
            self.state, out=self._observation_buffer
        )
        return observation.copy(), {}

    @override
    def step(
        self,
        action: Any,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        acceleration = self.observation_builder.denormalize_action(float(action[0]))
        previous = self.state
        result = self.transition(previous, acceleration)
        reward = self.reward_calculator.calculate(previous, result, self.task)
        self.state = result.next_state
        next_observation = self.observation_builder.build(
            self.state, out=self._observation_buffer
        )
        reason_str = (
            result.termination_reason.name
            if result.termination_reason is not None
            else None
        )
        self.outcome = EpisodeOutcome(
            termination_reason=reason_str,
        )
        if self.safety_truncation_buffer is not None:
            self.safety_truncation_buffer.record(
                position_m=self.state.s_m,
                termination_reason=result.termination_reason,
            )
        if self.reward_diagnostics_accumulator is not None:
            self.reward_diagnostics_accumulator.record(
                reward,
                termination_reason=result.termination_reason,
            )
        if not self.compact_training_info:
            delta_acc = abs(
                result.commanded_acceleration_mps2
                - previous.commanded_acceleration_mps2
            )
            self._comfort_tav += delta_acc
            self._comfort_sum_sq_delta_acc += delta_acc**2
            if delta_acc / self.step_time_s > self.task.max_jerk_mps3:
                self._comfort_exceedance_count += 1
        self._record_trajectory()
        info: dict[str, object]
        if self.compact_training_info:
            info = {
                "termination_reason": reason_str,
            }
        else:
            self._record_episode_info()
            assert self.episode_info is not None
            info = {
                "episode": self.episode_info.to_mapping(),
                "outcome": self.outcome.to_mapping(),
                "safety_margin_mps": min(
                    result.step_end_state.max_speed_mps - result.step_end_state.v_mps,
                    result.step_end_state.v_mps - result.step_end_state.lower_limit_mps,
                ),
                "termination_reason": reason_str,
            }
        return (
            # See reset(): VecEnv may retain a terminal observation after this
            # method returns, so it must not alias the reusable scratch buffer.
            next_observation.copy(),
            reward.total,
            result.termination_reason is not None,
            False,
            info,
        )

    def _record_episode_info(self) -> None:
        steps = max(self.state.step, 1)
        self.episode_info = EpisodeInfo(
            energy_consumption_j=self.state.total_energy_kj * 1000.0,
            operation_time_s=self.state.t_s,
            redundant_operation_time_s=self.state.slack_time_s,
            position_m=self.state.s_m,
            speed_mps=self.state.v_mps,
            stopping_point_index=self.state.sps.target_stopping_point_index,
            comfort_tav=self._comfort_tav,
            comfort_er_pct=self._comfort_exceedance_count / steps * 100.0,
            comfort_rms=math.sqrt(self._comfort_sum_sq_delta_acc / steps),
        )


def make_env(
    scenario: Scenario,
    task: Task,
    gamma: float,
    step_time_s: float,
    srtsp_lookup: SrtspLookup,
    normalization: RewardNormalization,
    compact_training_info: bool = False,
    enable_trajectory_tracking: bool = False,
    reward_config: RewardConfig | None = None,
    enable_safety_truncation_tracking: bool = False,
    reward_diagnostics_worker_rank: int | None = None,
    reward_diagnostics_rollout_capacity: int | None = None,
) -> MTTOEnv:
    if (reward_diagnostics_worker_rank is None) != (
        reward_diagnostics_rollout_capacity is None
    ):
        raise ValueError(
            "reward diagnostics worker rank and rollout capacity must be set together"
        )
    return MTTOEnv(
        scenario=scenario,
        task=task,
        gamma=gamma,
        step_time_s=step_time_s,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        compact_training_info=compact_training_info,
        enable_trajectory_tracking=enable_trajectory_tracking,
        reward_config=reward_config,
        safety_truncation_buffer=(
            SafetyTruncationBuffer() if enable_safety_truncation_tracking else None
        ),
        reward_diagnostics_accumulator=(
            RewardDiagnosticsAccumulator(
                worker_rank=reward_diagnostics_worker_rank,
                rollout_capacity=reward_diagnostics_rollout_capacity,
            )
            if reward_diagnostics_worker_rank is not None
            and reward_diagnostics_rollout_capacity is not None
            else None
        ),
    )
