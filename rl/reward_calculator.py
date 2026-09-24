"""Reward calculation shared by Gym training and external rollouts."""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from model.ocs import TrainService
from rl.operational_state import OperationalState, OperationalTransition

ENERGY_REWARD_SCALE: float = 15.0
COMFORT_REWARD_SCALE: float = 20.0
SURVIVAL_REWARD_SCALE: float = 50.0
SAFETY_POTENTIAL_SCALE: float = 0.5
SAFETY_POTENTIAL_STEEPNESS: float = 8.0
PUNCTUALITY_POTENTIAL_SCALE: float = 5.0
PUNCTUALITY_POTENTIAL_SIGMA_S: float = 20.0
PUNCTUALITY_DECAY_TIME_S: float = 45.0
STOPPING_SCORE_BETA: float = 0.8

# Floor on the step duration when a jerk is computed from a spatial step;
# near-zero-duration stopping transitions would otherwise blow it up.
JERK_CONTROL_PERIOD_S: float = 1.0

# Li et al. (2023), IEEE Access 11, Table 1 and Appendix: binary
# goal-directed reward with the published values. T_lim uses this study's
# 10 s strict punctuality tolerance (TrainService.max_arr_time_error_s).
LI_STEP_OPERATION: float = 2.5
LI_STEP_SPEED_LIMIT: float = -1.0
LI_STEP_PUNCTUALITY: float = -0.5
LI_STEP_ENERGY: float = -0.5
LI_STEP_COMFORT: float = -0.5
LI_GOAL_OPERATION: float = 350.0
LI_GOAL_PUNCTUAL_BONUS: float = 50.0
LI_COEF_TIME: float = -0.4
LI_COEF_ENERGY: float = -0.6
LI_COMFORT_LIMIT_MPS3: float = 0.3 * 9.81  # C_lim = 0.3 g/s
# E_lim: traction energy of the recorded operation on the same section,
# re-evaluated with the current long-stator energy model (c30cc36).
LI_ENERGY_LIMIT_KJ: float = 1431.879 * 3600.0
# Every goal-state term is scaled so that r_g keeps the margin over
# r_inf / (1 - gamma) that Li et al. require (their gamma = 0.99, here 0.998).
LI_GOAL_REWARD_SCALE: float = (1.0 - 0.99) / (1.0 - 0.998)


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
    """Selects which reward terms are active; all magnitudes are fixed."""

    enable_potential_safety: bool = True
    enable_potential_punctuality: bool = False
    reward_scheme: str = "base"

    def __post_init__(self) -> None:
        if self.reward_scheme not in ("base", "li2023_scaled"):
            raise ValueError(f"unknown reward scheme: {self.reward_scheme}")


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
    timetable slack. Together these priors form the Physics-Informed Reward
    Shaping (PIRS) method module.
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
        train_mass_kg: float | None = None,
    ) -> None:
        self.train_service: TrainService = train_service
        self.max_episode_steps: int = max_episode_steps
        self.whole_distance_m: float = whole_distance_m
        self.max_energy_consumption_kj: float = max(max_energy_consumption_kj, 1e-12)
        self.gamma: float = float(gamma)
        self.reward_config: RewardConfig = reward_config or RewardConfig()
        self.initial_min_operation_time_s = initial_min_operation_time_s
        self.train_mass_kg = train_mass_kg
        if self.reward_config.reward_scheme == "li2023_scaled" and not train_mass_kg:
            raise ValueError("the li2023_scaled reward scheme requires the train mass")
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

    def _calculate_li(self, transition: OperationalTransition) -> RewardBreakdown:
        previous, state = transition.previous_state, transition.next_state
        service = self.train_service
        # Per-step terms r_inf: constant bonus and binary penalties.
        survival = LI_STEP_OPERATION
        safety = LI_STEP_SPEED_LIMIT if state.speed_mps >= state.max_speed_mps else 0.0
        # Mid-trip trip-time error is the delay that remains unavoidable even
        # when running at the minimum-time profile from the current state.
        projected_delay_s = max(0.0, -state.redundant_operation_time_s)
        punctuality = (
            LI_STEP_PUNCTUALITY
            if projected_delay_s >= service.max_arr_time_error_s
            else 0.0
        )
        energy = (
            LI_STEP_ENERGY if state.energy_consumption_kj >= LI_ENERGY_LIMIT_KJ else 0.0
        )
        jerk = abs(transition.acceleration_mps2 - previous.acceleration_mps2) / max(
            transition.duration_s, JERK_CONTROL_PERIOD_S
        )
        comfort = LI_STEP_COMFORT if jerk >= LI_COMFORT_LIMIT_MPS3 else 0.0
        terminal_stopping = 0.0
        if transition.terminated:
            # Goal-state reward r_g (r_C = 0, so ride comfort adds nothing).
            goal_scale = LI_GOAL_REWARD_SCALE
            survival += goal_scale * LI_GOAL_OPERATION
            time_error_s = abs(service.schedule_time - state.operation_time_s)
            punctuality += goal_scale * (
                LI_COEF_TIME * time_error_s
                if time_error_s >= service.max_arr_time_error_s
                else LI_GOAL_PUNCTUAL_BONUS
            )
            assert self.train_mass_kg is not None
            specific_energy = (
                state.energy_consumption_kj
                * 1000.0
                / (self.train_mass_kg * self.whole_distance_m)
            )
            energy += goal_scale * LI_COEF_ENERGY * specific_energy
            if state.stop_error_m >= service.max_stop_error:
                terminal_stopping = -goal_scale * state.stop_error_m
        total = safety + energy + comfort + terminal_stopping + punctuality + survival
        return RewardBreakdown(
            safety=safety,
            energy=energy,
            comfort=comfort,
            terminal_stopping=terminal_stopping,
            terminal_punctuality=punctuality,
            survival=survival,
            total=total,
        )

    def calculate(self, transition: OperationalTransition) -> RewardBreakdown:
        if self.reward_config.reward_scheme == "li2023_scaled":
            return self._calculate_li(transition)
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
            -ENERGY_REWARD_SCALE
            * transition.energy_delta_kj
            / self.max_energy_consumption_kj
        )
        delta_acc = abs(
            transition.acceleration_mps2 - transition.previous_state.acceleration_mps2
        )
        norm_jerk = delta_acc / max(self.train_service.max_acc_change, 1e-12)
        comfort = -COMFORT_REWARD_SCALE / self.max_episode_steps * norm_jerk**2
        terminal_stopping = 0.0
        terminal_punctuality = 0.0
        if transition.terminated:
            stopping_score = self.stopping_score(state.stop_error_m)
            punctuality_score = self.punctuality_score(state.operation_time_s)
            terminal_stopping = stopping_score * 15.0
            terminal_punctuality = (
                punctuality_score * 5.0 + stopping_score**2 * punctuality_score * 20.0
            )

        survival = SURVIVAL_REWARD_SCALE / self.max_episode_steps
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
        """Global linear slack reference; never re-anchor at resets."""
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
        """Bounded Logistic risk at the two normalized envelope margins."""
        span = max(max_speed_mps - min_speed_mps, 1.0)

        upper_exponent = SAFETY_POTENTIAL_STEEPNESS * (max_speed_mps - speed_mps) / span
        if upper_exponent >= 0.0:
            upper_tail = math.exp(-upper_exponent)
            upper_risk = 2.0 * upper_tail / (1.0 + upper_tail)
        else:
            upper_risk = 2.0 / (1.0 + math.exp(upper_exponent))

        if min_speed_mps > 0.0:
            lower_exponent = (
                SAFETY_POTENTIAL_STEEPNESS * (speed_mps - min_speed_mps) / span
            )
            if lower_exponent >= 0.0:
                lower_tail = math.exp(-lower_exponent)
                lower_risk = 2.0 * lower_tail / (1.0 + lower_tail)
            else:
                lower_risk = 2.0 / (1.0 + math.exp(lower_exponent))
        else:
            lower_risk = 0.0
        return -SAFETY_POTENTIAL_SCALE * (upper_risk + lower_risk)
