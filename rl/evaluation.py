from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.env_factory import make_env
from rl.mtto_env import MTTOEnv
from utils.io_utils import save_curve_and_metrics


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
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float
    terminated: bool
    truncated: bool
    episode_steps: int
    trajectory_pos_m: NDArray[np.float32]
    trajectory_speed_mps: NDArray[np.float32]

    def comparison_key(self) -> tuple[int, float, float, float, float]:
        return (
            int(self.success),
            float(self.total_reward),
            -float(self.stop_error_m),
            -abs(float(self.time_error_s)),
            -float(self.total_energy_j),
        )

    def to_metrics(
        self,
        *,
        num_timesteps: int | None = None,
        trigger_mode: str | None = None,
        trigger_value: int | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "success": self.success,
            "total_reward": self.total_reward,
            "target_time_s": self.target_time_s,
            "total_time_s": self.total_time_s,
            "total_energy_j": self.total_energy_j,
            "total_energy_kj": self.total_energy_kj,
            "start_position_m": self.start_position_m,
            "target_position_m": self.target_position_m,
            "final_position_m": self.final_position_m,
            "final_speed_mps": self.final_speed_mps,
            "stop_error_m": self.stop_error_m,
            "time_error_s": self.time_error_s,
            "comfort_tav": self.comfort_tav,
            "comfort_er_pct": self.comfort_er_pct,
            "comfort_rms": self.comfort_rms,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_steps": self.episode_steps,
        }
        if num_timesteps is not None:
            metrics["num_timesteps"] = int(num_timesteps)
        if trigger_mode is not None:
            metrics["trigger_mode"] = trigger_mode
        if trigger_value is not None:
            metrics["trigger_value"] = int(trigger_value)

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
    )


def unwrap_mtto_env(env: gym.Env[Any, Any]) -> MTTOEnv:
    mtto_env = env.unwrapped
    if not isinstance(mtto_env, MTTOEnv):
        raise TypeError(f"Expected MTTOEnv, got {type(mtto_env)!r}")
    return mtto_env


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

    final_position = float(basic_info.get("position", mtto_env.current_pos))
    total_time_s = float(
        basic_info.get("operation_time", mtto_env.current_operation_time)
    )
    total_energy_kj = float(
        basic_info.get("energy_consumption", mtto_env.current_energy_consumption)
    )

    return PolicyEvaluationResult(
        success=bool(terminated and not truncated),
        total_reward=float(total_reward),
        total_time_s=total_time_s,
        target_time_s=float(mtto_env.task.schedule_time),
        total_energy_j=total_energy_kj * 1000.0,
        total_energy_kj=total_energy_kj,
        start_position_m=float(mtto_env.task.start_position),
        target_position_m=float(mtto_env.task.target_position),
        final_position_m=final_position,
        final_speed_mps=float(mtto_env.current_speed),
        stop_error_m=abs(float(mtto_env.task.target_position) - final_position),
        time_error_s=total_time_s - float(mtto_env.task.schedule_time),
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
