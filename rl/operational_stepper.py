"""Operational state transition used by the Gym environment and rollouts."""

import math

import numpy as np
from numpy.typing import NDArray

from model.common import ECC, ORS, calc_transition_from_acc_scalar_numba
from model.ocs import SPS, SafeGuardUtility, SPSState, TrainService
from model.track import TrackInfo, get_slope_scalar_numba
from model.vehicle import VehicleInfo
from rl.operational_state import OperationalState, OperationalTransition, ViolationCode


class OperationalStepper:
    """Owns static operating dependencies and advances explicit runtime state."""

    def __init__(
        self,
        *,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        max_step_distance_m: float,
    ) -> None:
        self.vehicle: VehicleInfo = vehicle
        self.track: TrackInfo = track
        self.safeguard_utility: SafeGuardUtility = safeguard_utility
        self.train_service: TrainService = train_service
        self.max_step_distance_m: float = float(max_step_distance_m)
        self.whole_distance_m: float = abs(
            train_service.target_position - train_service.start_position
        )
        self.direction: int = (
            1 if train_service.start_position < train_service.target_position else -1
        )
        # The final transition is available to complete the task.  A
        # non-terminal transition at this boundary is truncated in advance().
        self.required_episode_steps: int = math.ceil(
            self.whole_distance_m / self.max_step_distance_m
        )
        # Compatibility name retained for reward normalization and callers.
        self.max_episode_steps: int = self.required_episode_steps
        self.ecc: ECC = ECC(
            R_m=0.2796,
            L_d=0.0002,
            R_k=50.0,
            L_k=0.000142,
            Tau=0.258,
            Psi_fd=3.9629,
            k_c=0.8,
        )
        self.ors: ORS = ORS(
            vehicle=vehicle, track=track, factor=safeguard_utility.gamma
        )
        self.sps: SPS = SPS(
            sgu=safeguard_utility,
            ASA_ap_list=track.ASA_aps,
            ASA_dp_list=track.ASA_dps,
            T_s=2.0,
        )
        profile_pos, profile_speed = self.ors.calc_min_operation_time_curve(
            begin_pos=train_service.start_position,
            begin_speed=train_service.start_speed,
            end_pos=train_service.target_position + train_service.max_stop_error * 20,
            end_speed=0.0,
        )
        self._upper_speed_lut_pos_min: float
        self._upper_speed_lut_step: float
        self._upper_speed_lut_speed_arr: NDArray[np.float64]
        (
            self._upper_speed_lut_pos_min,
            self._upper_speed_lut_step,
            self._upper_speed_lut_speed_arr,
        ) = self._build_upper_speed_lookup_table(profile_pos, profile_speed)
        self.min_operation_time_s: float
        mec, lec, self.min_operation_time_s = (
            self.ors.calc_max_energy_and_min_operation_time(
                begin_pos=train_service.start_position,
                begin_speed=train_service.start_speed,
                end_pos=train_service.target_position,
                end_speed=0.0,
                distance=train_service.target_position - train_service.start_position,
                energy_con_calc=self.ecc,
            )
        )
        self.max_energy_consumption_kj: float = float(mec + lec)

    @staticmethod
    def _build_upper_speed_lookup_table(
        pos_arr: np.ndarray, speed_arr: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        pos = np.asarray(pos_arr, dtype=np.float64)
        speed = np.asarray(speed_arr, dtype=np.float64)
        if pos.size == 0:
            return 0.0, 1.0, np.zeros(1, dtype=np.float32)
        if pos.size == 1:
            return (
                float(pos[0]),
                1.0,
                np.asarray([max(0.0, float(speed[0]))], dtype=np.float32),
            )
        if pos[0] > pos[-1]:
            pos, speed = pos[::-1], speed[::-1]
        step = 10.0
        positions = (
            float(pos[0])
            + np.arange(int(np.ceil((pos[-1] - pos[0]) / step)) + 1, dtype=np.float64)
            * step
        )
        values = np.interp(
            positions,
            pos,
            speed,
            left=float(speed[0]),
            right=float(speed[-1]),
        )
        return float(pos[0]), step, np.maximum(values, 0.0).astype(np.float32)

    def get_upper_speed(self, position_m: float) -> float:
        values = self._upper_speed_lut_speed_arr
        if values.size == 0:
            return 0.0
        index = (
            float(position_m) - self._upper_speed_lut_pos_min
        ) / self._upper_speed_lut_step
        if index <= 0:
            return float(values[0])
        if index >= values.size - 1:
            return float(values[-1])
        index0 = int(index)
        ratio = index - index0
        return float(values[index0] + (values[index0 + 1] - values[index0]) * ratio)

    def get_upper_speed_or_zero(self, position_m: float) -> float:
        pos_max = self._upper_speed_lut_pos_min + self._upper_speed_lut_step * (
            self._upper_speed_lut_speed_arr.size - 1
        )
        if position_m < self._upper_speed_lut_pos_min or position_m > pos_max:
            return 0.0
        return self.get_upper_speed(position_m)

    def _calc_redundant_operation_time(
        self, position_m: float, speed_mps: float, operation_time_s: float
    ) -> float:
        min_remaining = self.ors.calc_min_operation_time(
            begin_pos=position_m,
            begin_speed=speed_mps,
            end_pos=self.train_service.target_position,
            end_speed=0.0,
        )
        return self.train_service.schedule_time - operation_time_s - min_remaining

    def _build_state(
        self,
        *,
        position_m: float,
        speed_mps: float,
        acceleration_mps2: float,
        operation_time_s: float,
        energy_consumption_kj: float,
        step_count: int,
        sps_state: SPSState,
    ) -> OperationalState:
        slope = float(
            get_slope_scalar_numba(
                position_m, self.track.slopes, self.track.slope_intervals
            )
        )
        min_speed, guard_max = self.safeguard_utility.get_min_and_max_speed(
            current_pos=position_m, current_sp=sps_state.target_stopping_point_index
        )
        return OperationalState(
            position_m=float(position_m),
            speed_mps=float(speed_mps),
            acceleration_mps2=float(acceleration_mps2),
            operation_time_s=float(operation_time_s),
            redundant_operation_time_s=self._calc_redundant_operation_time(
                position_m, speed_mps, operation_time_s
            ),
            energy_consumption_kj=float(energy_consumption_kj),
            slope_permille=slope,
            min_speed_mps=float(min_speed),
            max_speed_mps=min(self.get_upper_speed(position_m), float(guard_max)),
            stop_error_m=abs(self.train_service.target_position - position_m),
            sps_state=sps_state,
            step_count=step_count,
        )

    def reset(self) -> OperationalState:
        return self._build_state(
            position_m=self.train_service.start_position,
            speed_mps=self.train_service.start_speed,
            acceleration_mps2=0.0,
            operation_time_s=0.0,
            energy_consumption_kj=0.0,
            step_count=0,
            sps_state=self.sps.initial_state(),
        )

    def refresh_schedule_time(self, state: OperationalState) -> OperationalState:
        return self._build_state(
            position_m=state.position_m,
            speed_mps=state.speed_mps,
            acceleration_mps2=state.acceleration_mps2,
            operation_time_s=state.operation_time_s,
            energy_consumption_kj=state.energy_consumption_kj,
            step_count=state.step_count,
            sps_state=state.sps_state,
        )

    def advance(
        self,
        state: OperationalState,
        acceleration_mps2: float,
        *,
        requested_distance_m: float | None = None,
    ) -> OperationalTransition:
        """Advance one constant-acceleration transition.

        ``requested_distance_m`` is primarily intended for replaying a
        reference trajectory whose final segment is shorter than the regular
        RL step.  Normal environment interaction leaves it as ``None`` and
        therefore preserves the configured fixed-step behaviour.
        """
        if requested_distance_m is None:
            step_distance_m = self.max_step_distance_m
        else:
            step_distance_m = float(requested_distance_m)
            if (
                not math.isfinite(step_distance_m)
                or step_distance_m <= 0.0
                or step_distance_m > self.max_step_distance_m
            ):
                raise ValueError(
                    "requested_distance_m must be finite, positive, and no greater "
                    + "than max_step_distance_m"
                )
        next_speed, distance, duration = calc_transition_from_acc_scalar_numba(
            state.speed_mps, float(acceleration_mps2), step_distance_m
        )
        energy_mec, energy_lec = self.ecc.calc_energy(
            begin_pos=state.position_m,
            begin_speed=state.speed_mps,
            acc=float(acceleration_mps2),
            distance=distance,
            direction=self.direction,
            operation_time=duration,
            vehicle=self.vehicle,
            track=self.track,
        )
        energy_delta = float(energy_mec + energy_lec)
        position = state.position_m + distance * self.direction
        operation_time = state.operation_time_s + duration
        sps_state = self.sps.advance(
            state.sps_state,
            current_pos=position,
            current_speed=next_speed,
            current_time=operation_time,
        )
        next_state = self._build_state(
            position_m=position,
            speed_mps=next_speed,
            acceleration_mps2=float(acceleration_mps2),
            operation_time_s=operation_time,
            energy_consumption_kj=state.energy_consumption_kj + energy_delta,
            step_count=state.step_count + 1,
            sps_state=sps_state,
        )
        stopped = math.isclose(next_state.speed_mps, 0.0, abs_tol=0.01)
        terminated = stopped and next_state.stop_error_m <= 9.0
        low = next_state.speed_mps < next_state.min_speed_mps
        high = next_state.speed_mps > next_state.max_speed_mps
        # On the last permitted transition successful completion takes
        # precedence, so Gymnasium never sees termination and truncation
        # together.
        step_limit = (
            not terminated and next_state.step_count >= self.required_episode_steps
        )
        failed_stop = stopped and not terminated
        truncated = not terminated and (low or high or step_limit or failed_stop)
        if terminated:
            code = ViolationCode.ONGOING
        elif failed_stop:
            code = ViolationCode.FAILED_STOP
        elif low:
            code = ViolationCode.SPEED_LOW
        elif high:
            code = ViolationCode.SPEED_HIGH
        elif step_limit:
            code = ViolationCode.STEP_LIMIT
        else:
            code = ViolationCode.ONGOING
        return OperationalTransition(
            state,
            next_state,
            float(acceleration_mps2),
            float(distance),
            float(duration),
            energy_delta,
            terminated,
            truncated,
            code,
        )
