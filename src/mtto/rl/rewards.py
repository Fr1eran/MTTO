"""Reward calculation shared by Gym training and external rollouts."""

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from mtto.domain.scenario import STOPPED_SPEED_MPS, Task
from mtto.rl.state import State, StepResult, TerminationReason

# Progress dominates: every safe step towards the target is rewarded and the
# trip total is PROGRESS_REWARD_SCALE. Energy is a per-metre penalty and never
# outweighs a step's progress. Comfort is charged per m/s^2 of acceleration
# change, so a trip's comfort cost is COMFORT_REWARD_WEIGHT times its total
# variation (the evaluation metric TAV), independent of the control period and
# of where on the line the change happens; the multiplier (1 + rho^2) makes a
# change cost twice as much at the jerk limit and grow cubically beyond it. The
# weight sits near the energy/comfort exchange rate of the trained policies
# (about 0.19 reward per TAV unit): below it comfort comes almost free, above
# it every further TAV unit costs energy.
PROGRESS_REWARD_SCALE: float = 50.0
ENERGY_REWARD_WEIGHT: float = 0.4
COMFORT_REWARD_WEIGHT: float = 0.2
# Terminal reward of a stop in the zone, split 0.3 stopping / 0.7 punctuality.
TERMINAL_REWARD_SCALE: float = 100.0
# Every failed termination costs the same, more than the whole progress.
TRUNCATION_PENALTY: float = -80.0
# Safety potential: quadratic hinge over the last SAFETY_RESERVE_HORIZON_STEPS
# steps of full-braking (traction) reserve before the speed envelope.
SAFETY_RESERVE_HORIZON_STEPS: float = 3.0
SAFETY_RESERVE_UPPER_SCALE: float = 3.0
SAFETY_RESERVE_LOWER_SCALE: float = 1.0
PUNCTUALITY_POTENTIAL_SCALE: float = 5.0
PUNCTUALITY_POTENTIAL_SIGMA_S: float = 20.0
PUNCTUALITY_DECAY_TIME_S: float = 15.0
# Stopping score 1 / (1 + (|e| / tol)^p): the tolerance is the half-score
# point, steepest around it, flat near the target and a power-law tail beyond.
STOPPING_SCORE_POWER: float = 4.0


def punctuality_potential_from_error(
    error_s: float | NDArray[np.floating],
) -> float | NDArray[np.float64]:
    """Evaluate the punctuality potential from a slack error.

    Pseudo-Huber: ``-K * e^2 / sigma^2`` near zero and linear, with slope
    ``2K / sigma`` per second, far from it. It does not saturate, so every
    further second behind (or ahead of) the timetable keeps costing.
    """
    error = np.asarray(error_s, dtype=np.float64)
    sigma = PUNCTUALITY_POTENTIAL_SIGMA_S
    potential = (
        -2.0 * PUNCTUALITY_POTENTIAL_SCALE * (np.hypot(error, sigma) - sigma) / sigma
    )
    if potential.ndim == 0:
        return float(potential)
    return np.asarray(potential, dtype=np.float64)


def reference_punctuality_slack(
    position_m: float,
    schedule_time_s: float,
    task: Task,
    *,
    initial_min_operation_time_s: float,
) -> float:
    """Global linear slack reference; never re-anchor at resets."""
    fraction = min(
        1.0,
        max(
            0.0,
            (task.target_position_m - position_m)
            / (task.target_position_m - task.start_position_m),
        ),
    )
    return (schedule_time_s - initial_min_operation_time_s) * fraction


def braking_reserve_steps(
    speed_mps: float,
    upper_now_mps: float,
    upper_ahead_mps: float,
    max_dec_abs_mps2: float,
    step_time_s: float,
) -> float:
    """Control periods left before the upper limit, now or one period ahead.

    The reserve is the number of control periods, run at the current speed,
    after which the full-braking distance margin to the upper limit is used
    up. Kinematics are v'^2 = v^2 + 2 b ds, so it is counted in v^2 with
    ``2 * b * ds_eff`` per period, ``ds_eff = v * dt + b * dt^2 / 2``; the
    quadratic term only keeps the denominator away from zero at low speed.
    <= 0 means the state already exceeds the limit or cannot avoid exceeding
    it next period. A stopped train has an unbounded reserve.
    """
    if abs(speed_mps) <= STOPPED_SPEED_MPS:
        return math.inf
    braking_per_step = (
        2.0
        * max_dec_abs_mps2
        * (speed_mps * step_time_s + 0.5 * max_dec_abs_mps2 * step_time_s**2)
    )
    speed_sq = speed_mps**2
    return min(
        (upper_now_mps**2 - speed_sq) / braking_per_step,
        (upper_ahead_mps**2 - speed_sq) / braking_per_step + 1.0,
    )


def traction_reserve_steps(
    speed_mps: float,
    lower_now_mps: float,
    lower_ahead_mps: float,
    max_acc_mps2: float,
    step_time_s: float,
) -> float:
    """Control periods left above the lower limit (only where it is > 0).

    Mirrors :func:`braking_reserve_steps` with full traction:
    ``2 * a * (v * dt + a * dt^2 / 2)`` of v^2 per control period.
    """
    traction_per_step = (
        2.0
        * max_acc_mps2
        * (speed_mps * step_time_s + 0.5 * max_acc_mps2 * step_time_s**2)
    )
    speed_sq = speed_mps**2
    reserve = math.inf
    if lower_now_mps > 0.0:
        reserve = (speed_sq - lower_now_mps**2) / traction_per_step
    if lower_ahead_mps > 0.0:
        reserve = min(
            reserve, (speed_sq - lower_ahead_mps**2) / traction_per_step + 1.0
        )
    return reserve


def safety_potential(braking_reserve: float, traction_reserve: float) -> float:
    """Quadratic hinge on the reserves: 0 when ample, saturates when exhausted."""
    horizon = SAFETY_RESERVE_HORIZON_STEPS
    upper = (1.0 - min(max(braking_reserve, 0.0), horizon) / horizon) ** 2
    lower = (1.0 - min(max(traction_reserve, 0.0), horizon) / horizon) ** 2
    return -SAFETY_RESERVE_UPPER_SCALE * upper - SAFETY_RESERVE_LOWER_SCALE * lower


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Selects which reward terms are active; all magnitudes are fixed."""

    enable_potential_safety: bool = True
    enable_potential_punctuality: bool = False


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    safety: float = 0.0
    energy: float = 0.0
    comfort: float = 0.0
    terminal_stopping: float = 0.0
    terminal_punctuality: float = 0.0
    progress: float = 0.0
    truncation: float = 0.0
    total: float = 0.0
    punctuality_shaping: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardNormalization:
    peak_propulsion_kj_per_m: float
    initial_min_operation_time_s: float


class RewardCalculator:
    """Calculate reward from an explicit transition.

    The base reward is signed progress towards the target (negative once the
    train runs past it) minus per-metre propulsion energy, normalised by the
    peak propulsion energy per metre, and a total-variation comfort penalty
    weighted up by the jerk ratio. Levitation
    energy is not rewarded: it grows with time, diverges per metre at low
    speed and is nearly policy-independent for an on-time arrival. Failed
    terminations return a constant truncation penalty.
    Optional potential-based shaping guides safety and linear consumption of
    timetable slack. Together these priors form the Physics-Informed Reward
    Shaping (PIRS) method module.
    Terminal stopping and punctuality scores remain the task objectives.
    """

    PUNCTUALITY_DECAY_TIME_S: ClassVar[float] = PUNCTUALITY_DECAY_TIME_S
    STOPPING_SCORE_POWER: ClassVar[float] = STOPPING_SCORE_POWER

    def __init__(
        self,
        normalization: RewardNormalization,
        *,
        gamma: float,
        step_time_s: float,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.step_time_s = float(step_time_s)
        self.peak_propulsion_kj_per_m = normalization.peak_propulsion_kj_per_m
        self.gamma = float(gamma)
        self.reward_config = reward_config or RewardConfig()
        self.initial_min_operation_time_s = normalization.initial_min_operation_time_s
        if self.reward_config.enable_potential_punctuality and (
            not math.isfinite(self.initial_min_operation_time_s)
            or self.initial_min_operation_time_s < 0
        ):
            raise ValueError(
                "punctuality shaping requires a finite initial minimum time "
                "and positive distance"
            )

    def calculate(
        self, previous: State, result: StepResult, task: Task
    ) -> RewardBreakdown:
        state = result.step_end_state
        punctuality_shaping = self.reward_punctuality_potential(previous, result, task)
        safety = (
            self._reward_safety_potential(previous, result)
            if self.reward_config.enable_potential_safety
            else 0.0
        )
        if result.termination_reason not in (
            None,
            TerminationReason.STOPPED_IN_ZONE,
        ):
            return RewardBreakdown(
                safety=safety,
                truncation=TRUNCATION_PENALTY,
                punctuality_shaping=punctuality_shaping,
                total=safety + TRUNCATION_PENALTY + punctuality_shaping,
            )

        route_m = task.target_position_m - task.start_position_m
        # Signed: running past the target increases the distance again.
        progress = (
            PROGRESS_REWARD_SCALE
            * (
                abs(task.target_position_m - previous.s_m)
                - abs(task.target_position_m - state.s_m)
            )
            / route_m
        )
        energy = (
            -ENERGY_REWARD_WEIGHT
            * PROGRESS_REWARD_SCALE
            * result.propulsion_delta_kj
            / (self.peak_propulsion_kj_per_m * route_m)
        )
        # Jerk over the nominal control period: a step cut short by a stop
        # must not inflate it.
        acceleration_change = abs(
            result.commanded_acceleration_mps2 - previous.commanded_acceleration_mps2
        )
        jerk_ratio = acceleration_change / self.step_time_s / task.max_jerk_mps3
        comfort = -COMFORT_REWARD_WEIGHT * acceleration_change * (1.0 + jerk_ratio**2)
        terminal_stopping = 0.0
        terminal_punctuality = 0.0
        if result.termination_reason is TerminationReason.STOPPED_IN_ZONE:
            stopping_score = self.stopping_score(state.stop_error_m, task)
            punctuality_score = self.punctuality_score(state.t_s, state.schedule_time_s)
            terminal_stopping = TERMINAL_REWARD_SCALE * 0.3 * stopping_score
            terminal_punctuality = (
                TERMINAL_REWARD_SCALE * 0.7 * stopping_score * punctuality_score
            )

        total = (
            safety
            + energy
            + comfort
            + terminal_stopping
            + terminal_punctuality
            + progress
            + punctuality_shaping
        )
        return RewardBreakdown(
            safety=safety,
            energy=energy,
            comfort=comfort,
            terminal_stopping=terminal_stopping,
            terminal_punctuality=terminal_punctuality,
            progress=progress,
            total=total,
            punctuality_shaping=punctuality_shaping,
        )

    def reference_punctuality_slack(
        self, position_m: float, schedule_time_s: float, task: Task
    ) -> float:
        return reference_punctuality_slack(
            position_m,
            schedule_time_s,
            task,
            initial_min_operation_time_s=self.initial_min_operation_time_s,
        )

    def potential_punctuality(self, state: State, task: Task) -> float:
        if not self.reward_config.enable_potential_punctuality:
            return 0.0
        error = state.slack_time_s - self.reference_punctuality_slack(
            state.s_m, state.schedule_time_s, task
        )
        return float(punctuality_potential_from_error(error))

    def reward_punctuality_potential(
        self, previous: State, result: StepResult, task: Task
    ) -> float:
        if not self.reward_config.enable_potential_punctuality:
            return 0.0
        return self.gamma * self.potential_punctuality(
            result.step_end_state, task
        ) - self.potential_punctuality(previous, task)

    def stopping_score(self, stop_error_m: float, task: Task) -> float:
        """Smooth score that is 1 on target and exactly 1/2 at the tolerance."""
        ratio = abs(stop_error_m) / task.max_stop_error_m
        return 1.0 / (1.0 + ratio**self.STOPPING_SCORE_POWER)

    def punctuality_score(
        self, operation_time_s: float, schedule_time_s: float
    ) -> float:
        time_error = abs(schedule_time_s - operation_time_s)
        return math.exp(-time_error / self.PUNCTUALITY_DECAY_TIME_S)

    def _reward_safety_potential(self, previous: State, result: StepResult) -> float:
        current = result.step_end_state
        return self.gamma * safety_potential(
            current.braking_reserve_steps, current.traction_reserve_steps
        ) - safety_potential(
            previous.braking_reserve_steps, previous.traction_reserve_steps
        )


DEFAULT_REWARD_PRESET_NAME = "basic_safety_punctuality"


@dataclass(frozen=True)
class RewardPreset:
    """具名奖励预设，将实验身份与运行时奖励配置组合在一起。"""

    name: str
    label: str
    description: str
    config: RewardConfig

    def enabled_shaping_components(self) -> tuple[str, ...]:
        components: list[str] = []
        if self.config.enable_potential_safety:
            components.append("safety")
        if self.config.enable_potential_punctuality:
            components.append("punctuality")
        return tuple(components)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reward_preset_name": self.name,
            "reward_preset_label": self.label,
            "reward_preset_description": self.description,
            "potential_shaping_components": list(self.enabled_shaping_components()),
            "reward_config": reward_config_parameters(self.config),
        }


def reward_config_parameters(reward_config: RewardConfig) -> dict[str, Any]:
    """Return the reward switches plus the fixed reward magnitudes."""
    return {
        "progress_reward_scale": PROGRESS_REWARD_SCALE,
        "energy_reward_weight": ENERGY_REWARD_WEIGHT,
        "comfort_reward_weight": COMFORT_REWARD_WEIGHT,
        "comfort_penalty": "abs_delta_a_times_1_plus_rho_squared",
        "terminal_reward_scale": TERMINAL_REWARD_SCALE,
        "truncation_penalty": TRUNCATION_PENALTY,
        "enable_potential_safety": bool(reward_config.enable_potential_safety),
        "safety_reserve_horizon_steps": SAFETY_RESERVE_HORIZON_STEPS,
        "safety_reserve_upper_scale": SAFETY_RESERVE_UPPER_SCALE,
        "safety_reserve_lower_scale": SAFETY_RESERVE_LOWER_SCALE,
        "enable_potential_punctuality": reward_config.enable_potential_punctuality,
        "stopping_score_power": STOPPING_SCORE_POWER,
        "punctuality_potential_scale": PUNCTUALITY_POTENTIAL_SCALE,
        "punctuality_potential_sigma_s": PUNCTUALITY_POTENTIAL_SIGMA_S,
        "potential_transition_formula": "gamma_phi_next_minus_phi_previous",
        "terminal_next_potential": "observed_next_state",
    }


REWARD_PRESETS: dict[str, RewardPreset] = {
    "basic": RewardPreset(
        name="basic",
        label="basic",
        description="Base reward only: energy and comfort are always enabled.",
        config=RewardConfig(enable_potential_safety=False),
    ),
    "basic_safety": RewardPreset(
        name="basic_safety",
        label="basic+safety",
        description="Base reward plus potential-based safety shaping.",
        config=RewardConfig(enable_potential_safety=True),
    ),
    "basic_punctuality": RewardPreset(
        name="basic_punctuality",
        label="basic+punctuality",
        description="Base reward plus linear-slack punctuality potential.",
        config=RewardConfig(
            enable_potential_safety=False,
            enable_potential_punctuality=True,
        ),
    ),
    "basic_safety_punctuality": RewardPreset(
        name="basic_safety_punctuality",
        label="basic+safety+punctuality",
        description=(
            "Base reward plus Physics-Informed Reward Shaping (PIRS), combining "
            "safety and linear-slack punctuality potentials."
        ),
        config=RewardConfig(enable_potential_punctuality=True),
    ),
}

REWARD_PRESET_ALIASES: dict[str, str] = {
    "default": DEFAULT_REWARD_PRESET_NAME,
    "basic": "basic",
    "basic+safety": "basic_safety",
    "basic+punctuality": "basic_punctuality",
    "basic+safety+punctuality": "basic_safety_punctuality",
}


def reward_preset_names() -> tuple[str, ...]:
    """返回所有已注册奖励情形的名称元组。"""
    return tuple(REWARD_PRESETS.keys())


def _normalize_reward_preset_token(preset_name: str | None) -> str:
    if preset_name is None:
        return DEFAULT_REWARD_PRESET_NAME
    normalized = str(preset_name).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return DEFAULT_REWARD_PRESET_NAME
    return normalized


def resolve_reward_preset(preset_name: str | None = None) -> RewardPreset:
    """将奖励情形名称（含别名）解析为 RewardPreset 实例。

    Args:
        preset_name: 预设名，支持 ``basic``、``basic_safety``、
            ``basic_safety_punctuality`` 和 ``default``。

    Returns:
        对应的 RewardPreset 实例。

    Raises:
        ValueError: 未知的情形名。
    """
    normalized = _normalize_reward_preset_token(preset_name)
    canonical = REWARD_PRESET_ALIASES.get(normalized, normalized)
    preset = REWARD_PRESETS.get(canonical)
    if preset is None:
        available = ", ".join(reward_preset_names())
        raise ValueError(
            f"Unknown reward preset '{preset_name}'. Available presets: {available}"
        )
    return preset


def build_reward_config(preset_name: str | None = None) -> RewardConfig:
    """根据奖励情形名返回 RewardConfig 实例。

    Args:
        preset_name: 预设名，同 resolve_reward_preset。
    Returns:
        用于初始化 MTTOEnv 的 RewardConfig。
    """
    return resolve_reward_preset(preset_name).config
