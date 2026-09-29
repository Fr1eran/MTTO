"""Reward calculation shared by Gym training and external rollouts."""

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from mtto.domain.scenario import STOPPED_SPEED_MPS, Task
from mtto.rl.state import State, StepResult, TerminationReason

ENERGY_REWARD_SCALE: float = 15.0
COMFORT_REWARD_SCALE: float = 20.0
SURVIVAL_REWARD_SCALE: float = 50.0
# Safety potential: quadratic hinge over the last SAFETY_RESERVE_HORIZON_STEPS
# steps of full-braking (traction) reserve before the speed envelope.
SAFETY_RESERVE_HORIZON_STEPS: float = 3.0
SAFETY_RESERVE_UPPER_SCALE: float = 3.0
SAFETY_RESERVE_LOWER_SCALE: float = 1.0
PUNCTUALITY_POTENTIAL_SCALE: float = 5.0
PUNCTUALITY_POTENTIAL_SIGMA_S: float = 20.0
PUNCTUALITY_DECAY_TIME_S: float = 45.0
# Stopping score decays past the tolerance with this scale (m): 0.5 at a
# 0.3 m excess, so overrunning the strict tolerance is no longer cheap.
STOPPING_SCORE_BETA: float = 0.3

# Floor on the step duration when a jerk is computed from a spatial step;
# near-zero-duration stopping transitions would otherwise blow it up.
JERK_CONTROL_PERIOD_S: float = 1.0

# Li et al. (2023), IEEE Access 11, Table 1 and Appendix: binary
# goal-directed reward with the published values. T_lim uses this study's
# 10 s strict punctuality tolerance (Task.max_arr_time_error_s).
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
    braking_per_step: float,
) -> float:
    """Full-braking steps left before the upper limit, now or one step ahead.

    Kinematics are v'^2 = v^2 + 2 a dx, so the reserve is counted in v^2 with
    ``braking_per_step = 2 * b * dx``; <= 0 means the state already exceeds
    the limit or cannot avoid exceeding it next step. A stopped train has an
    unbounded reserve.
    """
    if abs(speed_mps) <= STOPPED_SPEED_MPS:
        return math.inf
    speed_sq = speed_mps**2
    return min(
        (upper_now_mps**2 - speed_sq) / braking_per_step,
        (upper_ahead_mps**2 - speed_sq) / braking_per_step + 1.0,
    )


def traction_reserve_steps(
    speed_mps: float,
    lower_now_mps: float,
    lower_ahead_mps: float,
    traction_per_step: float,
) -> float:
    """Full-traction steps left above the lower limit (only where it is > 0)."""
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


@dataclass(frozen=True, slots=True)
class RewardNormalization:
    max_energy_consumption_kj: float
    initial_min_operation_time_s: float
    required_episode_steps: int


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
        normalization: RewardNormalization,
        *,
        gamma: float,
        reward_config: RewardConfig | None = None,
        train_mass_kg: float | None = None,
    ) -> None:
        self.required_episode_steps = normalization.required_episode_steps
        self.max_energy_consumption_kj = max(
            normalization.max_energy_consumption_kj, 1e-12
        )
        self.gamma = float(gamma)
        self.reward_config = reward_config or RewardConfig()
        self.initial_min_operation_time_s = normalization.initial_min_operation_time_s
        self.train_mass_kg = train_mass_kg
        if self.reward_config.reward_scheme == "li2023_scaled" and not train_mass_kg:
            raise ValueError("the li2023_scaled reward scheme requires the train mass")
        if self.reward_config.enable_potential_punctuality and (
            not math.isfinite(self.initial_min_operation_time_s)
            or self.initial_min_operation_time_s < 0
        ):
            raise ValueError(
                "punctuality shaping requires a finite initial minimum time "
                "and positive distance"
            )

    def _calculate_li(
        self, previous: State, result: StepResult, task: Task
    ) -> RewardBreakdown:
        state = result.step_end_state
        # Per-step terms r_inf: constant bonus and binary penalties.
        survival = LI_STEP_OPERATION
        safety = LI_STEP_SPEED_LIMIT if state.v_mps >= state.max_speed_mps else 0.0
        # Mid-trip trip-time error is the delay that remains unavoidable even
        # when running at the minimum-time profile from the current state.
        projected_delay_s = max(0.0, -state.slack_time_s)
        punctuality = (
            LI_STEP_PUNCTUALITY
            if projected_delay_s >= task.max_arr_time_error_s
            else 0.0
        )
        energy = LI_STEP_ENERGY if state.total_energy_kj >= LI_ENERGY_LIMIT_KJ else 0.0
        jerk = abs(
            result.commanded_acceleration_mps2 - previous.commanded_acceleration_mps2
        ) / max(result.motion.duration_s, JERK_CONTROL_PERIOD_S)
        comfort = LI_STEP_COMFORT if jerk >= LI_COMFORT_LIMIT_MPS3 else 0.0
        terminal_stopping = 0.0
        if result.termination_reason is TerminationReason.STOPPED_IN_ZONE:
            # Goal-state reward r_g (r_C = 0, so ride comfort adds nothing).
            goal_scale = LI_GOAL_REWARD_SCALE
            survival += goal_scale * LI_GOAL_OPERATION
            time_error_s = abs(state.schedule_time_s - state.t_s)
            punctuality += goal_scale * (
                LI_COEF_TIME * time_error_s
                if time_error_s >= task.max_arr_time_error_s
                else LI_GOAL_PUNCTUAL_BONUS
            )
            assert self.train_mass_kg is not None
            specific_energy = (
                state.total_energy_kj
                * 1000.0
                / (
                    self.train_mass_kg
                    * (task.target_position_m - task.start_position_m)
                )
            )
            energy += goal_scale * LI_COEF_ENERGY * specific_energy
            if state.stop_error_m >= task.max_stop_error_m:
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

    def calculate(
        self, previous: State, result: StepResult, task: Task
    ) -> RewardBreakdown:
        if self.reward_config.reward_scheme == "li2023_scaled":
            return self._calculate_li(previous, result, task)
        state = result.step_end_state
        punctuality_shaping = self.reward_punctuality_potential(previous, result, task)
        safety = (
            self._reward_safety_potential(previous, result)
            if self.reward_config.enable_potential_safety
            else 0.0
        )
        if (
            result.termination_reason is not None
            and result.termination_reason is not TerminationReason.STOPPED_IN_ZONE
        ):
            progress = abs(state.s_m - task.target_position_m) / max(
                (task.target_position_m - task.start_position_m), 1e-12
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
            * (result.propulsion_delta_kj + result.levitation_delta_kj)
            / self.max_energy_consumption_kj
        )
        delta_acc = abs(
            result.commanded_acceleration_mps2 - previous.commanded_acceleration_mps2
        )
        norm_jerk = delta_acc / max(task.max_acc_change, 1e-12)
        comfort = -COMFORT_REWARD_SCALE / self.required_episode_steps * norm_jerk**2
        terminal_stopping = 0.0
        terminal_punctuality = 0.0
        if result.termination_reason is TerminationReason.STOPPED_IN_ZONE:
            stopping_score = self.stopping_score(state.stop_error_m, task)
            punctuality_score = self.punctuality_score(state.t_s, state.schedule_time_s)
            terminal_stopping = stopping_score * 15.0
            terminal_punctuality = (
                punctuality_score * 5.0 + stopping_score**2 * punctuality_score * 20.0
            )

        survival = SURVIVAL_REWARD_SCALE / self.required_episode_steps
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
        delta = max(0.0, abs(stop_error_m) - task.max_stop_error_m)
        return 1.0 / (1.0 + (delta / self.STOPPING_SCORE_BETA) ** 2)

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


def goal_reward_scale(reward_config: RewardConfig) -> float:
    """Fixed goal-state multiplier recorded in metadata for the reward scheme."""
    if reward_config.reward_scheme == "li2023_scaled":
        return LI_GOAL_REWARD_SCALE
    return 1.0


def reward_config_parameters(reward_config: RewardConfig) -> dict[str, Any]:
    """Return the reward switches plus the fixed reward magnitudes."""
    return {
        "energy_reward_scale": ENERGY_REWARD_SCALE,
        "comfort_reward_scale": COMFORT_REWARD_SCALE,
        "enable_potential_safety": bool(reward_config.enable_potential_safety),
        "survival_reward_scale": SURVIVAL_REWARD_SCALE,
        "safety_reserve_horizon_steps": SAFETY_RESERVE_HORIZON_STEPS,
        "safety_reserve_upper_scale": SAFETY_RESERVE_UPPER_SCALE,
        "safety_reserve_lower_scale": SAFETY_RESERVE_LOWER_SCALE,
        "enable_potential_punctuality": reward_config.enable_potential_punctuality,
        "stopping_score_beta": STOPPING_SCORE_BETA,
        "punctuality_potential_scale": PUNCTUALITY_POTENTIAL_SCALE,
        "punctuality_potential_sigma_s": PUNCTUALITY_POTENTIAL_SIGMA_S,
        "potential_transition_formula": "gamma_phi_next_minus_phi_previous",
        "terminal_next_potential": "observed_next_state",
        "reward_scheme": reward_config.reward_scheme,
        "goal_reward_scale": goal_reward_scale(reward_config),
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
    "li2023_scaled": RewardPreset(
        name="li2023_scaled",
        label="Li et al. (2023), goal reward rescaled",
        description=(
            "Binary goal-directed reward of Li et al. (2023) with the published "
            "per-step penalties and tuned coefficients; T_lim set to the 10 s "
            "strict punctuality tolerance; every goal-state term multiplied by "
            "(1 - 0.99) / (1 - 0.998) = 5 so that r_g keeps the margin over "
            "r_inf / (1 - gamma) that Li et al. require (their gamma = 0.99)."
        ),
        config=RewardConfig(
            enable_potential_safety=False, reward_scheme="li2023_scaled"
        ),
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
