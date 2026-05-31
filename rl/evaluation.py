from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.env_factory import make_env
from rl.mtto_env import MTTOEnv, RewardConfig
from utils.io_utils import save_curve_and_metrics

BEST_TRAJECTORY_SELECTION_RULE = "success_strict_stop_strict_time_energy_else_reward"
BEST_TRAJECTORY_SELECTION_RULE_DESCRIPTION = (
    "Any successful evaluation outranks any non-successful evaluation. "
    "Among non-successful evaluations, higher total_reward wins. "
    "Among successful evaluations, strict stop accuracy wins first; if neither "
    "trajectory meets it, lower stop_error_m wins. Strict arrival time wins next; "
    "if neither trajectory meets it, lower abs(time_error_s) wins. Lower "
    "total_energy_j wins only after those strict requirements."
)


@dataclass(frozen=True)
class PolicyEvaluationResult:
    success: bool
    total_reward: float
    total_time_s: float
    target_time_s: float
    total_energy_j: float
    total_energy_kj: float
    start_position_m: float
    target_position_m: float
    final_position_m: float
    final_speed_mps: float
    stop_error_m: float
    time_error_s: float
    strict_stop_error_limit_m: float
    strict_time_error_limit_s: float
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float
    terminated: bool
    truncated: bool
    episode_steps: int
    trajectory_pos_m: NDArray[np.float32]
    trajectory_speed_mps: NDArray[np.float32]

    @property
    def strict_stop_requirement_met(self) -> bool:
        return float(self.stop_error_m) <= float(self.strict_stop_error_limit_m)

    @property
    def strict_time_requirement_met(self) -> bool:
        return abs(float(self.time_error_s)) <= float(self.strict_time_error_limit_s)

    def to_metrics(
        self,
        *,
        num_timesteps: int | None = None,
        eval_trigger_mode: str | None = None,
        eval_trigger_interval: int | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "total_reward": self.total_reward,
            "target_time_s": self.target_time_s,
            "total_time_s": self.total_time_s,
            "time_error_s": self.time_error_s,
            "start_position_m": self.start_position_m,
            "target_position_m": self.target_position_m,
            "final_position_m": self.final_position_m,
            "stop_error_m": self.stop_error_m,
            "total_energy_kj": self.total_energy_kj,
            "total_energy_j": self.total_energy_j,
            "final_speed_mps": self.final_speed_mps,
            "comfort_tav": self.comfort_tav,
            "comfort_er_pct": self.comfort_er_pct,
            "comfort_rms": self.comfort_rms,
            "episode_steps": self.episode_steps,
            "success": self.success,
            "strict_stop_error_limit_m": self.strict_stop_error_limit_m,
            "strict_time_error_limit_s": self.strict_time_error_limit_s,
            "strict_stop_requirement_met": self.strict_stop_requirement_met,
            "strict_time_requirement_met": self.strict_time_requirement_met,
            "selection_comparison_key": list(
                build_policy_evaluation_comparison_key(self)
            ),
        }
        metrics["selection_rule"] = BEST_TRAJECTORY_SELECTION_RULE
        if num_timesteps is not None:
            metrics["num_timesteps"] = int(num_timesteps)
        if eval_trigger_mode is not None:
            metrics["eval_trigger_mode"] = eval_trigger_mode
        if eval_trigger_interval is not None:
            metrics["eval_trigger_interval"] = int(eval_trigger_interval)

        return metrics


def build_single_eval_env(
    *,
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    max_step_distance: float,
    enable_diagnostics: bool = False,
    enable_trajectory_tracking: bool = True,
    render_mode: str | None = None,
    reward_config: RewardConfig | None = None,
) -> gym.Env[Any, Any]:
    return make_env(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=gamma,
        max_step_distance=max_step_distance,
        enable_diagnostics=enable_diagnostics,
        enable_trajectory_tracking=enable_trajectory_tracking,
        render_mode=render_mode,
        reward_config=reward_config,
    )


def unwrap_mtto_env(env: gym.Env[Any, Any]) -> MTTOEnv:
    mtto_env = env.unwrapped
    if not isinstance(mtto_env, MTTOEnv):
        raise TypeError(f"Expected MTTOEnv, got {type(mtto_env)!r}")
    return mtto_env


def is_success_within_train_service_limits(
    *,
    stop_error_m: float,
    time_error_s: float,
    train_service: TrainService,
) -> bool:
    schedule_time_s = float(train_service.schedule_time)
    if schedule_time_s <= 0.0:
        return False

    time_error_ratio = abs(float(time_error_s)) / schedule_time_s
    return (
        float(stop_error_m) <= float(train_service.max_stop_error)
        and time_error_ratio <= float(train_service.max_arr_time_error_ratio)
    )


def get_strict_stop_error_limit_m(train_service: TrainService) -> float:
    return float(train_service.max_stop_error)


def get_strict_time_error_limit_s(train_service: TrainService) -> float:
    return float(train_service.schedule_time) * float(
        train_service.max_arr_time_error_ratio
    )


def evaluate_policy_once(
    model: Any,
    env: gym.Env[Any, Any],
    *,
    deterministic: bool = True,
) -> PolicyEvaluationResult:
    obs, _ = env.reset()
    total_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        episode_steps += 1

    mtto_env = unwrap_mtto_env(env)
    basic_info = mtto_env.basic_info

    trajectory_pos = np.asarray(
        [] if mtto_env.trajectory_pos is None else mtto_env.trajectory_pos,
        dtype=np.float32,
    )
    trajectory_speed = np.asarray(
        [] if mtto_env.trajectory_speed_mps is None else mtto_env.trajectory_speed_mps,
        dtype=np.float32,
    )

    final_position = float(basic_info.get("position", mtto_env.current_position))
    target_time_s = float(mtto_env.train_service.schedule_time)
    total_time_s = float(
        basic_info.get("operation_time", mtto_env.current_operation_time)
    )
    total_energy_kj = float(
        basic_info.get("energy_consumption", mtto_env.current_energy_consumption)
    )
    stop_error_m = abs(float(mtto_env.train_service.target_position) - final_position)
    time_error_s = total_time_s - target_time_s
    success = is_success_within_train_service_limits(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        train_service=mtto_env.train_service,
    )

    return PolicyEvaluationResult(
        success=success,
        total_reward=float(total_reward),
        total_time_s=total_time_s,
        target_time_s=target_time_s,
        total_energy_j=total_energy_kj * 1000.0,
        total_energy_kj=total_energy_kj,
        start_position_m=float(mtto_env.train_service.start_position),
        target_position_m=float(mtto_env.train_service.target_position),
        final_position_m=final_position,
        final_speed_mps=float(basic_info.get("speed", mtto_env.current_speed)),
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        strict_stop_error_limit_m=get_strict_stop_error_limit_m(
            mtto_env.train_service
        ),
        strict_time_error_limit_s=get_strict_time_error_limit_s(
            mtto_env.train_service
        ),
        comfort_tav=float(basic_info.get("comfort_tav", 0.0)),
        comfort_er_pct=float(basic_info.get("comfort_er_pct", 0.0)),
        comfort_rms=float(basic_info.get("comfort_rms", 0.0)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_steps=episode_steps,
        trajectory_pos_m=trajectory_pos,
        trajectory_speed_mps=trajectory_speed,
    )


def save_policy_evaluation_curve(
    result: PolicyEvaluationResult,
    output_path: str,
    *,
    extra_metrics: dict[str, Any] | None = None,
) -> tuple[str, str]:
    metrics = result.to_metrics()
    if extra_metrics:
        metrics.update(extra_metrics)

    return save_curve_and_metrics(
        pos_arr=result.trajectory_pos_m,
        speed_arr=result.trajectory_speed_mps,
        output_path=output_path,
        metrics=metrics,
    )


def build_policy_evaluation_comparison_key(
    result: PolicyEvaluationResult,
) -> tuple[float, ...]:
    if not result.success:
        return (0.0, float(result.total_reward))

    strict_stop_met = result.strict_stop_requirement_met
    strict_time_met = result.strict_time_requirement_met
    stop_component = 0.0 if strict_stop_met else -float(result.stop_error_m)
    time_component = 0.0 if strict_time_met else -abs(float(result.time_error_s))

    return (
        1.0,
        1.0 if strict_stop_met else 0.0,
        stop_component,
        1.0 if strict_time_met else 0.0,
        time_component,
        -float(result.total_energy_j),
    )


def _best_update_reason_for_successes(
    candidate: PolicyEvaluationResult,
    previous: PolicyEvaluationResult,
) -> str | None:
    if (
        candidate.strict_stop_requirement_met
        and not previous.strict_stop_requirement_met
    ):
        return "strict_stop_requirement_reached"
    if (
        not candidate.strict_stop_requirement_met
        and not previous.strict_stop_requirement_met
        and float(candidate.stop_error_m) < float(previous.stop_error_m)
    ):
        return "lower_stop_error_before_strict_stop"

    if (
        candidate.strict_time_requirement_met
        and not previous.strict_time_requirement_met
    ):
        return "strict_time_requirement_reached"
    if (
        not candidate.strict_time_requirement_met
        and not previous.strict_time_requirement_met
        and abs(float(candidate.time_error_s)) < abs(float(previous.time_error_s))
    ):
        return "lower_time_error_before_strict_time"

    if float(candidate.total_energy_j) < float(previous.total_energy_j):
        return "lower_energy_after_strict_requirements"

    return None


def describe_best_update_reason(
    candidate: PolicyEvaluationResult,
    previous: PolicyEvaluationResult | None,
) -> str | None:
    if previous is None:
        return "first_evaluation"

    if build_policy_evaluation_comparison_key(
        candidate
    ) <= build_policy_evaluation_comparison_key(previous):
        return None

    if candidate.success and not previous.success:
        return "success_replaces_reward_fallback"
    if not candidate.success and not previous.success:
        return "higher_total_reward_without_success"
    if candidate.success and previous.success:
        return _best_update_reason_for_successes(candidate, previous)

    return None
