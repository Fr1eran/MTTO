"""Reward calculation shared by Gym training and external rollouts."""

import math
from dataclasses import dataclass

from model.ocs import TrainService
from rl.operational_state import OperationalTransition


@dataclass(frozen=True, slots=True)
class RewardConfig:
    enable_energy: bool = True
    enable_comfort: bool = True
    enable_potential_safety: bool = True


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


class RewardCalculator:
    """Calculate reward from an explicit transition.

    Safety PBRS is the only dense potential-based guidance term. Stopping
    accuracy and punctuality deliberately remain terminal-only objectives.
    """

    def __init__(
        self,
        train_service: TrainService,
        *,
        max_episode_steps: int,
        whole_distance_m: float,
        max_energy_consumption_kj: float,
        gamma: float,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.train_service: TrainService = train_service
        self.max_episode_steps: int = max_episode_steps
        self.whole_distance_m: float = whole_distance_m
        self.max_energy_consumption_kj: float = max(max_energy_consumption_kj, 1e-12)
        self.gamma: float = float(gamma)
        self.reward_config: RewardConfig = reward_config or RewardConfig()

    def calculate(self, transition: OperationalTransition) -> RewardBreakdown:
        state = transition.next_state
        if transition.truncated:
            progress = abs(state.position_m - self.train_service.target_position) / max(
                self.whole_distance_m, 1e-12
            )
            truncation = -(1.0 + progress**2) * 5.0
            return RewardBreakdown(truncation=truncation, total=truncation)

        safety = (
            self._reward_safety_potential(transition)
            if self.reward_config.enable_potential_safety
            else 0.0
        )
        energy = (
            -15.0 * transition.energy_delta_kj / self.max_energy_consumption_kj
            if self.reward_config.enable_energy
            else 0.0
        )
        comfort = 0.0
        if self.reward_config.enable_comfort:
            delta_acc = abs(
                transition.acceleration_mps2
                - transition.previous_state.acceleration_mps2
            )
            norm_jerk = delta_acc / max(self.train_service.max_acc_change, 1e-12)
            comfort = -20.0 / self.max_episode_steps * norm_jerk**2
        terminal_stopping = 0.0
        terminal_punctuality = 0.0
        if transition.terminated:
            stopping_score = self.stopping_score(state.stop_error_m)
            punctuality_score = self.punctuality_score(state.operation_time_s)
            terminal_stopping = stopping_score * 15.0
            terminal_punctuality = (
                punctuality_score * 5.0 + stopping_score**2 * punctuality_score * 20.0
            )

        survival = 100.0 / self.max_episode_steps
        total = (
            safety
            + energy
            + comfort
            + terminal_stopping
            + terminal_punctuality
            + survival
        )
        return RewardBreakdown(
            safety=safety,
            energy=energy,
            comfort=comfort,
            terminal_stopping=terminal_stopping,
            terminal_punctuality=terminal_punctuality,
            survival=survival,
            total=total,
        )

    def stopping_score(self, stop_error_m: float) -> float:
        beta = 0.8
        delta = max(0.0, abs(stop_error_m) - self.train_service.max_stop_error)
        return 1.0 / (1.0 + (delta / beta) ** 2)

    def punctuality_score(self, operation_time_s: float) -> float:
        time_error = abs(self.train_service.schedule_time - operation_time_s)
        return math.exp(-time_error / 45.0)

    def task_completion(
        self,
        *,
        terminated: bool,
        truncated: bool,
        stop_error_m: float,
        operation_time_s: float,
        success_base: float = 0.6,
        stopping_weight: float = 0.25,
        punctuality_weight: float = 0.15,
    ) -> float:
        """Return the bounded terminal task-completion target."""
        if truncated or not terminated:
            return 0.0
        completion = (
            float(success_base)
            + float(stopping_weight) * self.stopping_score(stop_error_m)
            + float(punctuality_weight) * self.punctuality_score(operation_time_s)
        )
        return min(1.0, max(0.0, completion))

    # Private aliases retained for callers from earlier project revisions.
    def _stopping_score(self, stop_error_m: float) -> float:
        return self.stopping_score(stop_error_m)

    def _punctuality_score(self, operation_time_s: float) -> float:
        return self.punctuality_score(operation_time_s)

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
        """PBRS safety potential with a band-scaled, non-overlapping buffer."""
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
