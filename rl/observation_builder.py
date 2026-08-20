"""Pure conversion from an operational state to the agent observation."""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from model.ocs import TrainService
from model.track import TrackInfo, get_slope_scalar_numba
from model.vehicle import VehicleInfo, calc_levi_deceleration_scalar_numba
from rl.operational_state import OperationalState


class ObservationBuilder:
    target_attraction_domain_radius_m: float = 3000.0
    lookahead_distance_m: float = 1000.0
    lookahead_num_samples: int = 10
    OBSERVATION_DIM: int = 12

    def __init__(
        self,
        *,
        vehicle: VehicleInfo,
        track: TrackInfo,
        train_service: TrainService,
        step_distance_m: float,
        direction: int,
        whole_distance_m: float,
        get_upper_speed_or_zero: Callable[[float], float],
    ) -> None:
        self.vehicle: VehicleInfo = vehicle
        self.track: TrackInfo = track
        self.train_service: TrainService = train_service
        self.step_distance_m: float = float(step_distance_m)
        if not math.isfinite(self.step_distance_m) or self.step_distance_m <= 0.0:
            raise ValueError("step_distance_m must be finite and positive")
        self.direction: int = int(direction)
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        self.whole_distance_m: float = max(whole_distance_m, 1e-12)
        self._get_upper_speed_or_zero: Callable[[float], float] = (
            get_upper_speed_or_zero
        )
        (
            self._lookahead_avg_slope_by_step,
            self._lookahead_avg_upper_speed_by_step,
        ) = self._build_lookahead_feature_cache()
        self._obs_buffer: NDArray[np.float32] = np.empty(
            self.OBSERVATION_DIM, dtype=np.float32
        )

    def build(
        self,
        state: OperationalState,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        target = self._obs_buffer if out is None else out
        distance = self.train_service.target_position - state.position_m
        suggested = self.calc_coasting_acc(state)
        if abs(distance) <= self.target_attraction_domain_radius_m:
            suggested = -(state.speed_mps**2) / (2.0 * max(abs(distance), 1e-6))

        target[0] = max(-1.0, min(1.0, distance / self.whole_distance_m))
        target[1] = max(-1.0, min(1.0, state.speed_mps / self.vehicle.max_speed))
        target[2] = self.normalize_acc_to_action(state.acceleration_mps2)
        target[3] = self.normalize_acc_to_action(suggested)
        target[4] = max(
            -1.0,
            min(
                1.0,
                (self.train_service.schedule_time - state.operation_time_s)
                / self.train_service.schedule_time,
            ),
        )
        target[5] = max(
            -1.0,
            min(
                1.0,
                state.redundant_operation_time_s / self.train_service.schedule_time,
            ),
        )
        target[6] = max(-1.0, min(1.0, state.max_speed_mps / self.vehicle.max_speed))
        target[7] = max(-1.0, min(1.0, state.min_speed_mps / self.vehicle.max_speed))
        target[8] = max(
            -1.0, min(1.0, state.slope_permille / self.vehicle.max_slope_capacity)
        )
        target[9] = max(
            -1.0,
            min(
                1.0,
                self.get_lookahead_avg_slope(state.step_count)
                / self.vehicle.max_slope_capacity,
            ),
        )
        target[10] = max(
            -1.0,
            min(
                1.0,
                self.get_lookahead_avg_upper_speed(state.step_count)
                / self.vehicle.max_speed,
            ),
        )
        target[11] = self.calc_approach_progress(distance)
        return target

    def normalize_acc_to_action(self, acc: float) -> float:
        value = (
            2.0
            * (float(acc) - self.vehicle.max_dec)
            / (self.vehicle.max_acc - self.vehicle.max_dec)
            - 1.0
        )
        return max(-1.0, min(1.0, value))

    def denormalize_action(self, action: float) -> float:
        value = (self.vehicle.max_acc + self.vehicle.max_dec) / 2.0 + float(action) * (
            self.vehicle.max_acc - self.vehicle.max_dec
        ) / 2.0
        return float(value)

    def calc_coasting_acc(self, state: OperationalState) -> float:
        return -float(
            calc_levi_deceleration_scalar_numba(
                state.speed_mps,
                state.slope_permille,
                self.vehicle.mass,
                self.vehicle.numoftrainsets,
            )
        )

    def calc_approach_progress(self, distance_m: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                1.0 - abs(float(distance_m)) / self.target_attraction_domain_radius_m,
            ),
        )

    def _build_lookahead_feature_cache(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        offsets = np.linspace(
            self.step_distance_m,
            self.lookahead_distance_m,
            self.lookahead_num_samples,
            dtype=np.float64,
        )
        node_count = int(math.ceil(self.whole_distance_m / self.step_distance_m)) + 1
        slope_cache = np.empty(node_count, dtype=np.float64)
        upper_speed_cache = np.empty(node_count, dtype=np.float64)
        start_position_m = float(self.train_service.start_position)
        for step_index in range(node_count):
            position_m = (
                start_position_m
                + self.direction * step_index * self.step_distance_m
            )
            slope_cache[step_index] = sum(
                get_slope_scalar_numba(
                    position_m + self.direction * float(offset),
                    self.track.slopes,
                    self.track.slope_intervals,
                )
                for offset in offsets
            ) / offsets.size
            upper_speed_cache[step_index] = sum(
                self._get_upper_speed_or_zero(
                    position_m + self.direction * float(offset)
                )
                for offset in offsets
            ) / offsets.size
        slope_cache.flags.writeable = False
        upper_speed_cache.flags.writeable = False
        return slope_cache, upper_speed_cache

    def _validate_lookahead_step_index(self, step_index: int) -> int:
        if not isinstance(step_index, (int, np.integer)):
            raise TypeError("step_index must be an integer")
        index = int(step_index)
        if not 0 <= index < self._lookahead_avg_slope_by_step.size:
            raise IndexError(
                f"step index {index} is outside cached lookahead range "
                + f"[0, {self._lookahead_avg_slope_by_step.size - 1}]"
            )
        return index

    def get_lookahead_avg_slope(self, step_index: int) -> float:
        index = self._validate_lookahead_step_index(step_index)
        return float(self._lookahead_avg_slope_by_step[index])

    def get_lookahead_avg_upper_speed(self, step_index: int) -> float:
        index = self._validate_lookahead_step_index(step_index)
        return float(self._lookahead_avg_upper_speed_by_step[index])
