"""Pure conversion from an operational state to the agent observation."""

import numpy as np
from numpy.typing import NDArray

from model.ocs import TrainService
from model.track import TrackInfo, get_slope_scalar_numba
from model.vehicle import VehicleInfo, calc_levi_deceleration_scalar_numba
from rl.operational_state import OperationalState


class ObservationBuilder:
    target_attraction_domain_radius_m = 3000.0

    def __init__(
        self,
        *,
        vehicle: VehicleInfo,
        track: TrackInfo,
        train_service: TrainService,
        max_step_distance_m: float,
        direction: int,
        whole_distance_m: float,
        get_upper_speed_or_zero,
    ) -> None:
        self.vehicle, self.track, self.train_service = vehicle, track, train_service
        self.max_step_distance_m, self.direction = max_step_distance_m, direction
        self.whole_distance_m = max(whole_distance_m, 1e-12)
        self._get_upper_speed_or_zero = get_upper_speed_or_zero

    def build(self, state: OperationalState) -> NDArray[np.float32]:
        distance = self.train_service.target_position - state.position_m
        suggested = self.calc_coasting_acc(state)
        if abs(distance) <= self.target_attraction_domain_radius_m:
            suggested = -(state.speed_mps**2) / (2.0 * max(abs(distance), 1e-6))
        values = np.asarray(
            [
                distance / self.whole_distance_m,
                state.speed_mps / self.vehicle.max_speed,
                self.normalize_acc_to_action(state.acceleration_mps2),
                self.normalize_acc_to_action(suggested),
                (self.train_service.schedule_time - state.operation_time_s)
                / self.train_service.schedule_time,
                state.redundant_operation_time_s / self.train_service.schedule_time,
                state.max_speed_mps / self.vehicle.max_speed,
                state.min_speed_mps / self.vehicle.max_speed,
                state.slope_permille / self.vehicle.max_slope_capacity,
                self.calc_lookahead_avg_slope(state) / self.vehicle.max_slope_capacity,
                self.calc_lookahead_avg_upper_speed(state) / self.vehicle.max_speed,
                self.calc_approach_progress(distance),
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    def normalize_acc_to_action(self, acc: float) -> float:
        value = (
            2.0
            * (float(acc) - self.vehicle.max_dec)
            / (self.vehicle.max_acc - self.vehicle.max_dec)
            - 1.0
        )
        return float(np.clip(value, -1.0, 1.0))

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
        return float(
            np.clip(
                1.0 - abs(float(distance_m)) / self.target_attraction_domain_radius_m,
                0.0,
                1.0,
            )
        )

    def calc_lookahead_avg_slope(
        self,
        state: OperationalState,
        lookahead_distance_m: float = 1000.0,
        num_samples: int = 10,
    ) -> float:
        offsets = np.linspace(
            self.max_step_distance_m,
            lookahead_distance_m,
            max(1, int(num_samples)),
            dtype=np.float64,
        )
        return float(
            sum(
                get_slope_scalar_numba(
                    state.position_m + self.direction * float(offset),
                    self.track.slopes,
                    self.track.slope_intervals,
                )
                for offset in offsets
            )
            / offsets.size
        )

    def calc_lookahead_avg_upper_speed(
        self,
        state: OperationalState,
        lookahead_distance_m: float = 1000.0,
        num_samples: int = 10,
    ) -> float:
        offsets = np.linspace(
            self.max_step_distance_m,
            lookahead_distance_m,
            max(1, int(num_samples)),
            dtype=np.float64,
        )
        return float(
            sum(
                self._get_upper_speed_or_zero(
                    state.position_m + self.direction * float(offset)
                )
                for offset in offsets
            )
            / offsets.size
        )
