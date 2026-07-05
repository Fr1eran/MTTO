import math
import os
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.env_factory import make_env
from rl.mtto_env import DEFAULT_PUNCTUALITY_DP_CURVE_DIR, MTTOEnv, RewardConfig
from utils.io_utils import save_curve_and_metrics

ARRIVAL_STOP_SPEED_ABS_TOL_MPS = 0.01

BEST_TRAJECTORY_SELECTION_RULE = "arrival_precise_punctual_energy_else_reward"
BEST_TRAJECTORY_SELECTION_RULE_DESCRIPTION = (
    "Any successful arrival outranks any non-arrival evaluation. "
    "Among non-arrivals, higher total_reward wins. Among successful arrivals, "
    "precise arrival wins first; if neither trajectory is precise, lower "
    "stop_error_m wins. Punctual arrival wins next; if neither trajectory is "
    "punctual, lower abs(time_error_s) wins. Lower total_energy_j wins only "
    "after those task-completion levels."
)


@dataclass(frozen=True)
class PolicyEvaluationResult:
    success: bool
    precise_arrival: bool
    punctual_arrival: bool
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
    min_safety_margin_mps: float = 0.0
    mean_safety_margin_mps: float = 0.0

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
            "precise_arrival": self.precise_arrival,
            "punctual_arrival": self.punctual_arrival,
            "min_safety_margin_mps": self.min_safety_margin_mps,
            "mean_safety_margin_mps": self.mean_safety_margin_mps,
            "strict_stop_error_limit_m": self.strict_stop_error_limit_m,
            "strict_time_error_limit_s": self.strict_time_error_limit_s,
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
    punctuality_dp_curve_dir: str | os.PathLike[str] | None = (
        DEFAULT_PUNCTUALITY_DP_CURVE_DIR
    ),
    punctuality_reference_match_tolerance: float = 1e-3,
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
        punctuality_dp_curve_dir=punctuality_dp_curve_dir,
        punctuality_reference_match_tolerance=punctuality_reference_match_tolerance,
    )


def unwrap_mtto_env(env: gym.Env[Any, Any]) -> MTTOEnv:
    mtto_env = env.unwrapped
    if not isinstance(mtto_env, MTTOEnv):
        raise TypeError(f"Expected MTTOEnv, got {type(mtto_env)!r}")
    return mtto_env


def is_successful_arrival(
    *,
    stop_error_m: float,
    final_speed_mps: float,
    train_service: TrainService,
) -> bool:
    is_stopped = math.isclose(
        float(final_speed_mps),
        0.0,
        abs_tol=ARRIVAL_STOP_SPEED_ABS_TOL_MPS,
    )
    return bool(
        is_stopped
        and float(stop_error_m) <= float(train_service.max_stop_error) * 10.0
    )


def is_precise_arrival(
    *,
    success: bool,
    stop_error_m: float,
    train_service: TrainService,
) -> bool:
    return bool(success and float(stop_error_m) <= float(train_service.max_stop_error))


def is_punctual_arrival(
    *,
    precise_arrival: bool,
    time_error_s: float,
    train_service: TrainService,
) -> bool:
    schedule_time_s = float(train_service.schedule_time)
    if not precise_arrival or schedule_time_s <= 0.0:
        return False

    time_error_ratio = abs(float(time_error_s)) / schedule_time_s
    return bool(time_error_ratio <= float(train_service.max_arr_time_error_ratio))


def classify_arrival_status(
    *,
    stop_error_m: float,
    time_error_s: float,
    final_speed_mps: float,
    train_service: TrainService,
) -> tuple[bool, bool, bool]:
    success = is_successful_arrival(
        stop_error_m=stop_error_m,
        final_speed_mps=final_speed_mps,
        train_service=train_service,
    )
    precise_arrival = is_precise_arrival(
        success=success,
        stop_error_m=stop_error_m,
        train_service=train_service,
    )
    punctual_arrival = is_punctual_arrival(
        precise_arrival=precise_arrival,
        time_error_s=time_error_s,
        train_service=train_service,
    )
    return success, precise_arrival, punctual_arrival


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
    mtto_env = unwrap_mtto_env(env)
    total_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False
    safety_margins: list[float] = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        episode_steps += 1

        margin_to_vmax = float(mtto_env.current_max_speed) - float(
            mtto_env.current_speed
        )
        margin_to_vmin = float(mtto_env.current_speed) - float(
            mtto_env.current_min_speed
        )
        safety_margin = min(margin_to_vmax, margin_to_vmin)
        safety_margins.append(safety_margin)

    basic_info = mtto_env.basic_info
    if safety_margins:
        safety_margin_arr = np.asarray(safety_margins, dtype=np.float64)
        min_safety_margin_mps = float(np.min(safety_margin_arr))
        mean_safety_margin_mps = float(np.mean(safety_margin_arr))
    else:
        min_safety_margin_mps = 0.0
        mean_safety_margin_mps = 0.0

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
    final_speed_mps = float(basic_info.get("speed", mtto_env.current_speed))
    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        final_speed_mps=final_speed_mps,
        train_service=mtto_env.train_service,
    )

    return PolicyEvaluationResult(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=float(total_reward),
        total_time_s=total_time_s,
        target_time_s=target_time_s,
        total_energy_j=total_energy_kj * 1000.0,
        total_energy_kj=total_energy_kj,
        start_position_m=float(mtto_env.train_service.start_position),
        target_position_m=float(mtto_env.train_service.target_position),
        final_position_m=final_position,
        final_speed_mps=final_speed_mps,
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
        min_safety_margin_mps=min_safety_margin_mps,
        mean_safety_margin_mps=mean_safety_margin_mps,
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

    precise_arrival = bool(result.precise_arrival)
    punctual_arrival = bool(result.punctual_arrival)
    stop_component = 0.0 if precise_arrival else -float(result.stop_error_m)
    time_component = 0.0 if punctual_arrival else -abs(float(result.time_error_s))

    return (
        1.0,
        1.0 if precise_arrival else 0.0,
        stop_component,
        1.0 if punctual_arrival else 0.0,
        time_component,
        -float(result.total_energy_j),
    )


def _best_update_reason_for_successes(
    candidate: PolicyEvaluationResult,
    previous: PolicyEvaluationResult,
) -> str | None:
    if (
        candidate.precise_arrival
        and not previous.precise_arrival
    ):
        return "precise_arrival_reached"
    if (
        not candidate.precise_arrival
        and not previous.precise_arrival
        and float(candidate.stop_error_m) < float(previous.stop_error_m)
    ):
        return "lower_stop_error_before_precise_arrival"

    if (
        candidate.punctual_arrival
        and not previous.punctual_arrival
    ):
        return "punctual_arrival_reached"
    if (
        not candidate.punctual_arrival
        and not previous.punctual_arrival
        and abs(float(candidate.time_error_s)) < abs(float(previous.time_error_s))
    ):
        return "lower_time_error_before_punctual_arrival"

    if float(candidate.total_energy_j) < float(previous.total_energy_j):
        return "lower_energy_after_arrival_requirements"

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
