"""Reward calculation shared by Gym training and external rollouts."""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from model.ocs import TrainService
from rl.operational_state import OperationalState, OperationalTransition

DEFAULT_ENERGY_REWARD_SCALE: float = 15.0
DEFAULT_COMFORT_REWARD_SCALE: float = 20.0
DEFAULT_SURVIVAL_REWARD_SCALE: float = 50.0
PUNCTUALITY_POTENTIAL_SCALE: float = 5.0
PUNCTUALITY_POTENTIAL_SIGMA_S: float = 20.0
PUNCTUALITY_DECAY_TIME_S: float = 45.0
STOPPING_SCORE_BETA: float = 0.8


def punctuality_potential_from_error(
    error_s: float | NDArray[np.floating],
) -> float | NDArray[np.float64]:
    """Evaluate the bounded punctuality potential from a slack error."""
    error = np.asarray(error_s, dtype=np.float64)
    ratio = error / np.hypot(error, PUNCTUALITY_POTENTIAL_SIGMA_S)
    potential = -PUNCTUALITY_POTENTIAL_SCALE * ratio * ratio
    if potential.ndim == 0:
        return float(potential)
    return np.asarray(potential, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    energy_reward_scale: float = DEFAULT_ENERGY_REWARD_SCALE
    comfort_reward_scale: float = DEFAULT_COMFORT_REWARD_SCALE
    enable_potential_safety: bool = True
    survival_reward_scale: float = DEFAULT_SURVIVAL_REWARD_SCALE
    enable_potential_punctuality: bool = False

    def __post_init__(self) -> None:
        defaults = {
            "energy_reward_scale": DEFAULT_ENERGY_REWARD_SCALE,
            "comfort_reward_scale": DEFAULT_COMFORT_REWARD_SCALE,
            "survival_reward_scale": DEFAULT_SURVIVAL_REWARD_SCALE,
        }
        for field_name, default_value in defaults.items():
            raw_value = getattr(self, field_name)
            try:
                value = float(raw_value)
            except TypeError, ValueError:
                value = default_value
            if not math.isfinite(value) or value < 0.0:
                value = default_value
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    safety: float = 0.0
    energy: float = 0.0
    comfort: float = 0.0
    terminal_stopping: float = 0.0
    terminal_punctuality: float = 0.0
    survival: float = 0.0
    truncation: float = 0.0
    total: float = 0.0
    punctuality_shaping: float = 0.0


class RewardCalculator:
    """Calculate reward from an explicit transition.

    Optional potential-based shaping guides safety and linear consumption of
    timetable slack. Together these priors form the PPRS method module.
    Terminal stopping and punctuality scores remain the task objectives.
    """

    PUNCTUALITY_DECAY_TIME_S: ClassVar[float] = PUNCTUALITY_DECAY_TIME_S
    STOPPING_SCORE_BETA: ClassVar[float] = STOPPING_SCORE_BETA

    def __init__(
        self,
        train_service: TrainService,
        *,
        max_episode_steps: int,
        whole_distance_m: float,
        max_energy_consumption_kj: float,
        gamma: float,
        reward_config: RewardConfig | None = None,
        initial_min_operation_time_s: float | None = None,
    ) -> None:
        self.train_service: TrainService = train_service
        self.max_episode_steps: int = max_episode_steps
        self.whole_distance_m: float = whole_distance_m
        self.max_energy_consumption_kj: float = max(max_energy_consumption_kj, 1e-12)
        self.gamma: float = float(gamma)
        self.reward_config: RewardConfig = reward_config or RewardConfig()
        self.initial_min_operation_time_s = initial_min_operation_time_s
        if self.reward_config.enable_potential_punctuality and (
            initial_min_operation_time_s is None
            or not math.isfinite(initial_min_operation_time_s)
            or initial_min_operation_time_s < 0
            or not math.isfinite(whole_distance_m)
            or whole_distance_m <= 0
        ):
            raise ValueError(
                "punctuality shaping requires a finite initial minimum time "
                "and positive distance"
            )

    def calculate(self, transition: OperationalTransition) -> RewardBreakdown:
        state = transition.next_state
        punctuality_shaping = self.reward_punctuality_potential(transition)
        safety = (
            self._reward_safety_potential(transition)
            if self.reward_config.enable_potential_safety
            else 0.0
        )
        if transition.truncated:
            progress = abs(state.position_m - self.train_service.target_position) / max(
                self.whole_distance_m, 1e-12
            )
            truncation = -(1.0 + progress**2) * 5.0
            return RewardBreakdown(
                safety=safety,
                truncation=truncation,
                punctuality_shaping=punctuality_shaping,
                total=safety + truncation + punctuality_shaping,
            )

        energy = (
            -self.reward_config.energy_reward_scale
            * transition.energy_delta_kj
            / self.max_energy_consumption_kj
        )
        delta_acc = abs(
            transition.acceleration_mps2 - transition.previous_state.acceleration_mps2
        )
        norm_jerk = delta_acc / max(self.train_service.max_acc_change, 1e-12)
        comfort = (
            -self.reward_config.comfort_reward_scale
            / self.max_episode_steps
            * norm_jerk**2
        )
        terminal_stopping = 0.0
        terminal_punctuality = 0.0
        if transition.terminated:
            stopping_score = self.stopping_score(state.stop_error_m)
            punctuality_score = self.punctuality_score(state.operation_time_s)
            terminal_stopping = stopping_score * 15.0
            terminal_punctuality = (
                punctuality_score * 5.0 + stopping_score**2 * punctuality_score * 20.0
            )

        survival = self.reward_config.survival_reward_scale / self.max_episode_steps
        total = (
            safety
            + energy
            + comfort
            + terminal_stopping
            + terminal_punctuality
            + survival
            + punctuality_shaping
        )
        return RewardBreakdown(
            safety=safety,
            energy=energy,
            comfort=comfort,
            terminal_stopping=terminal_stopping,
            terminal_punctuality=terminal_punctuality,
            survival=survival,
            total=total,
            punctuality_shaping=punctuality_shaping,
        )

    def reference_punctuality_slack(self, position_m: float) -> float:
        """Global linear slack reference; never re-anchor at curriculum resets."""
        if self.initial_min_operation_time_s is None:
            raise ValueError("initial minimum operation time is required")
        service = self.train_service
        direction = 1 if service.target_position > service.start_position else -1
        fraction = min(
            1.0,
            max(
                0.0,
                (service.target_position - position_m)
                * direction
                / self.whole_distance_m,
            ),
        )
        return (service.schedule_time - self.initial_min_operation_time_s) * fraction

    def potential_punctuality(self, state: OperationalState) -> float:
        if not self.reward_config.enable_potential_punctuality:
            return 0.0
        error = state.redundant_operation_time_s - self.reference_punctuality_slack(
            state.position_m
        )
        return float(punctuality_potential_from_error(error))

    def reward_punctuality_potential(self, transition: OperationalTransition) -> float:
        if not self.reward_config.enable_potential_punctuality:
            return 0.0
        return self.gamma * self.potential_punctuality(
            transition.next_state
        ) - self.potential_punctuality(transition.previous_state)

    def stopping_score(self, stop_error_m: float) -> float:
        delta = max(0.0, abs(stop_error_m) - self.train_service.max_stop_error)
        return 1.0 / (1.0 + (delta / self.STOPPING_SCORE_BETA) ** 2)

    def punctuality_score(self, operation_time_s: float) -> float:
        time_error = abs(self.train_service.schedule_time - operation_time_s)
        return math.exp(-time_error / self.PUNCTUALITY_DECAY_TIME_S)

    def _reward_safety_potential(self, transition: OperationalTransition) -> float:
        previous = transition.previous_state
        current = transition.next_state
        phi_previous = self._potential_safety(
            speed_mps=previous.speed_mps,
            min_speed_mps=previous.min_speed_mps,
            max_speed_mps=previous.max_speed_mps,
        )
        phi_current = self._potential_safety(
            speed_mps=current.speed_mps,
            min_speed_mps=current.min_speed_mps,
            max_speed_mps=current.max_speed_mps,
        )
        return self.gamma * phi_current - phi_previous

    @staticmethod
    def _potential_safety(
        *, speed_mps: float, min_speed_mps: float, max_speed_mps: float
    ) -> float:
        """Safety potential with a band-scaled, non-overlapping buffer."""
        K_safety = 1.0
        speed_band = max_speed_mps - min_speed_mps
        safety_buffer = min(max(0.15 * speed_band, 1.0), 5.0)

        alpha = 3.0

        margin_upper = max_speed_mps - speed_mps
        x_upper = 1.0 - margin_upper / safety_buffer
        z_upper = math.log1p(math.exp(alpha * x_upper)) / alpha
        phi_upper = -(z_upper**2)
        if min_speed_mps > 0.0:
            margin_lower = speed_mps - min_speed_mps
            x_lower = 1.0 - margin_lower / safety_buffer
            z_lower = math.log1p(math.exp(alpha * x_lower)) / alpha
            phi_lower = -(z_lower**2)
        else:
            phi_lower = 0.0
        return K_safety * (phi_upper + phi_lower)
