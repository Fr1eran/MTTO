from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from contracts.common import JSONValue, as_json_value
from contracts.evaluation import EvaluationArtifact, EvaluationMetrics, TrajectoryData
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.env_factory import make_env
from rl.mtto_env import MTTOEnv
from rl.observation_builder import ObservationBuilder
from rl.operational_stepper import OperationalStepper
from rl.reward_calculator import RewardCalculator, RewardConfig
from utils.io_utils import save_evaluation_artifact

BEST_TRAJECTORY_SELECTION_RULE = "arrival_precise_punctual_energy_else_reward"
BEST_TRAJECTORY_SELECTION_RULE_DESCRIPTION = (
    "Any successful arrival (terminated=True and truncated=False) outranks "
    "any non-arrival "
    "evaluation. "
    "Among non-arrivals, higher total_reward wins. Among successful arrivals, "
    "precise arrival wins first; if neither trajectory is precise, lower "
    "stop_error_m wins. Punctual arrival wins next; if neither trajectory is "
    "punctual, lower abs(time_error_s) wins. Punctual arrival requires "
    "abs(time_error_s) < TrainService.max_arr_time_error_s. Lower total_energy_j "
    "wins only after those task-completion levels."
)


@dataclass(frozen=True, init=False)
class PolicyEvaluationResult:
    success: bool
    precise_arrival: bool
    punctual_arrival: bool
    total_reward: float
    total_time_s: float
    target_time_s: float
    total_energy_j: float
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
    safety_violation_positions_m: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    def __init__(
        self,
        *,
        success: bool,
        precise_arrival: bool,
        punctual_arrival: bool,
        total_reward: float,
        total_time_s: float,
        target_time_s: float,
        total_energy_j: float,
        start_position_m: float,
        target_position_m: float,
        final_position_m: float,
        final_speed_mps: float,
        stop_error_m: float,
        time_error_s: float,
        strict_stop_error_limit_m: float,
        strict_time_error_limit_s: float,
        comfort_tav: float,
        comfort_er_pct: float,
        comfort_rms: float,
        terminated: bool,
        truncated: bool,
        episode_steps: int,
        trajectory_pos_m: NDArray[np.float32],
        trajectory_speed_mps: NDArray[np.float32],
        min_safety_margin_mps: float = 0.0,
        mean_safety_margin_mps: float = 0.0,
        safety_violation_positions_m: NDArray[np.float32] | None = None,
        # Accepted only as an in-memory migration alias.  It is not a
        # dataclass field and is never serialized; Joules remain canonical.
        total_energy_kj: float | None = None,
    ) -> None:
        if total_energy_kj is not None and not np.isclose(
            float(total_energy_kj), float(total_energy_j) / 1000.0
        ):
            raise ValueError("total_energy_kj does not match total_energy_j")
        values = {
            "success": bool(success),
            "precise_arrival": bool(precise_arrival),
            "punctual_arrival": bool(punctual_arrival),
            "total_reward": float(total_reward),
            "total_time_s": float(total_time_s),
            "target_time_s": float(target_time_s),
            "total_energy_j": float(total_energy_j),
            "start_position_m": float(start_position_m),
            "target_position_m": float(target_position_m),
            "final_position_m": float(final_position_m),
            "final_speed_mps": float(final_speed_mps),
            "stop_error_m": float(stop_error_m),
            "time_error_s": float(time_error_s),
            "strict_stop_error_limit_m": float(strict_stop_error_limit_m),
            "strict_time_error_limit_s": float(strict_time_error_limit_s),
            "comfort_tav": float(comfort_tav),
            "comfort_er_pct": float(comfort_er_pct),
            "comfort_rms": float(comfort_rms),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_steps": int(episode_steps),
            "trajectory_pos_m": np.asarray(trajectory_pos_m, dtype=np.float32),
            "trajectory_speed_mps": np.asarray(trajectory_speed_mps, dtype=np.float32),
            "min_safety_margin_mps": float(min_safety_margin_mps),
            "mean_safety_margin_mps": float(mean_safety_margin_mps),
            "safety_violation_positions_m": np.asarray(
                np.empty(0, dtype=np.float32)
                if safety_violation_positions_m is None
                else safety_violation_positions_m,
                dtype=np.float32,
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def to_metrics(
        self,
        *,
        num_timesteps: int | None = None,
        evaluation_rollout_index: int | None = None,
    ) -> dict[str, object]:
        """Serialize metrics using the canonical versioned contract."""
        return self.to_evaluation_metrics(
            num_timesteps=num_timesteps,
            evaluation_rollout_index=evaluation_rollout_index,
        ).to_mapping()

    @property
    def total_energy_kj(self) -> float:
        """Presentation conversion; Joules remain the canonical field."""
        return self.total_energy_j / 1000.0

    def to_evaluation_metrics(
        self,
        *,
        extensions: Mapping[str, object] | None = None,
        num_timesteps: int | None = None,
        evaluation_rollout_index: int | None = None,
    ) -> EvaluationMetrics:
        extension_values: dict[str, JSONValue] = {}
        if extensions:
            for key, value in extensions.items():
                extension_values[str(key)] = as_json_value(
                    value,
                    field=f"evaluation_metrics.extensions.{key}",
                )
        return EvaluationMetrics(
            success=bool(self.success),
            precise_arrival=bool(self.precise_arrival),
            punctual_arrival=bool(self.punctual_arrival),
            total_reward=float(self.total_reward),
            total_time_s=float(self.total_time_s),
            target_time_s=float(self.target_time_s),
            time_error_s=float(self.time_error_s),
            start_position_m=float(self.start_position_m),
            target_position_m=float(self.target_position_m),
            final_position_m=float(self.final_position_m),
            final_speed_mps=float(self.final_speed_mps),
            stop_error_m=float(self.stop_error_m),
            total_energy_j=float(self.total_energy_j),
            comfort_tav=float(self.comfort_tav),
            comfort_er_pct=float(self.comfort_er_pct),
            comfort_rms=float(self.comfort_rms),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            episode_steps=int(self.episode_steps),
            min_safety_margin_mps=float(self.min_safety_margin_mps),
            mean_safety_margin_mps=float(self.mean_safety_margin_mps),
            strict_stop_error_limit_m=float(self.strict_stop_error_limit_m),
            strict_time_error_limit_s=float(self.strict_time_error_limit_s),
            selection_comparison_key=build_policy_evaluation_comparison_key(self),
            selection_rule=BEST_TRAJECTORY_SELECTION_RULE,
            num_timesteps=(None if num_timesteps is None else int(num_timesteps)),
            evaluation_rollout_index=(
                None
                if evaluation_rollout_index is None
                else int(evaluation_rollout_index)
            ),
            extensions=extension_values,
        )

    def to_artifact(self) -> EvaluationArtifact:
        return EvaluationArtifact(
            metrics=self.to_evaluation_metrics(),
            trajectory=TrajectoryData(
                position_m=self.trajectory_pos_m,
                speed_mps=self.trajectory_speed_mps,
                safety_violation_positions_m=self.safety_violation_positions_m,
            ),
        )


def build_single_eval_env(
    *,
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    step_distance: float,
    enable_trajectory_tracking: bool = True,
    render_mode: str | None = None,
    reward_config: RewardConfig | None = None,
) -> gym.Env[np.ndarray, np.ndarray]:
    return make_env(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=gamma,
        step_distance=step_distance,
        enable_trajectory_tracking=enable_trajectory_tracking,
        render_mode=render_mode,
        reward_config=reward_config,
    )


def unwrap_mtto_env(env: gym.Env[np.ndarray, np.ndarray]) -> MTTOEnv:
    mtto_env = env.unwrapped
    if not isinstance(mtto_env, MTTOEnv):
        raise TypeError(f"Expected MTTOEnv, got {type(mtto_env)!r}")
    return mtto_env


def is_successful_arrival(
    *,
    terminated: bool,
    truncated: bool,
) -> bool:
    """Return the environment's authoritative task-completion result."""
    return bool(terminated and not truncated)


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
    return bool(
        precise_arrival
        and abs(float(time_error_s)) < float(train_service.max_arr_time_error_s)
    )


def classify_arrival_status(
    *,
    stop_error_m: float,
    time_error_s: float,
    final_speed_mps: float,
    train_service: TrainService,
    terminated: bool,
    truncated: bool,
) -> tuple[bool, bool, bool]:
    del final_speed_mps
    success = is_successful_arrival(
        terminated=terminated,
        truncated=truncated,
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
    return float(train_service.max_arr_time_error_s)


def evaluate_policy_once(
    model: Any,
    env: gym.Env[np.ndarray, np.ndarray],
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
    safety_violation_positions_m: list[float] = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        episode_steps += 1

        margin_to_vmax = float(mtto_env.state.max_speed_mps) - float(
            mtto_env.state.speed_mps
        )
        margin_to_vmin = float(mtto_env.state.speed_mps) - float(
            mtto_env.state.min_speed_mps
        )
        safety_margin = min(margin_to_vmax, margin_to_vmin)
        safety_margins.append(safety_margin)
        if safety_margin < 0.0:
            safety_violation_positions_m.append(float(mtto_env.state.position_m))

    episode_info = mtto_env.episode_info
    if episode_info is None:
        raise RuntimeError("evaluation environment did not produce episode info")
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

    final_position = float(episode_info.position_m)
    target_time_s = float(mtto_env.train_service.schedule_time)
    total_time_s = float(episode_info.operation_time_s)
    total_energy_j = float(episode_info.energy_consumption_j)
    stop_error_m = abs(float(mtto_env.train_service.target_position) - final_position)
    time_error_s = total_time_s - target_time_s
    final_speed_mps = float(episode_info.speed_mps)
    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        final_speed_mps=final_speed_mps,
        train_service=mtto_env.train_service,
        terminated=terminated,
        truncated=truncated,
    )

    return PolicyEvaluationResult(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=float(total_reward),
        total_time_s=total_time_s,
        target_time_s=target_time_s,
        total_energy_j=total_energy_j,
        start_position_m=float(mtto_env.train_service.start_position),
        target_position_m=float(mtto_env.train_service.target_position),
        final_position_m=final_position,
        final_speed_mps=final_speed_mps,
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        strict_stop_error_limit_m=get_strict_stop_error_limit_m(mtto_env.train_service),
        strict_time_error_limit_s=get_strict_time_error_limit_s(mtto_env.train_service),
        comfort_tav=float(episode_info.comfort_tav),
        comfort_er_pct=float(episode_info.comfort_er_pct),
        comfort_rms=float(episode_info.comfort_rms),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_steps=episode_steps,
        trajectory_pos_m=trajectory_pos,
        trajectory_speed_mps=trajectory_speed,
        min_safety_margin_mps=min_safety_margin_mps,
        mean_safety_margin_mps=mean_safety_margin_mps,
        safety_violation_positions_m=np.asarray(
            safety_violation_positions_m, dtype=np.float32
        ),
    )


def evaluate_operational_policy_once(
    policy: Any,
    *,
    stepper: OperationalStepper,
    reward_calculator: RewardCalculator,
    observation_builder: ObservationBuilder,
    deterministic: bool = True,
) -> PolicyEvaluationResult:
    """Evaluate a policy without a Gym environment.

    ``policy`` receives the same float32 observation as the training
    environment and may return either an action array or a scalar action.
    Stable-Baselines models are accepted directly through their ``predict``
    method.
    """
    state = stepper.reset()
    total_reward = 0.0
    comfort_tav = comfort_sum_sq = 0.0
    comfort_exceedance_count = 0
    positions = [state.position_m]
    speeds = [abs(state.speed_mps)]
    safety_margins: list[float] = []
    safety_violation_positions_m: list[float] = []
    observation_buffer = np.empty(ObservationBuilder.OBSERVATION_DIM, dtype=np.float32)
    terminated = truncated = False

    while not (terminated or truncated):
        observation = observation_builder.build(state, out=observation_buffer)
        if hasattr(policy, "predict"):
            action, _ = policy.predict(observation, deterministic=deterministic)
        else:
            action = policy(observation)
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        acceleration = (
            stepper.vehicle.max_acc + stepper.vehicle.max_dec
        ) / 2 + action_value * (stepper.vehicle.max_acc - stepper.vehicle.max_dec) / 2
        transition = stepper.advance(state, acceleration)
        breakdown = reward_calculator.calculate(transition)
        total_reward += breakdown.total
        delta_acc = abs(transition.acceleration_mps2 - state.acceleration_mps2)
        comfort_tav += delta_acc
        comfort_sum_sq += delta_acc**2
        if delta_acc > stepper.train_service.max_acc_change:
            comfort_exceedance_count += 1
        state = transition.next_state
        terminated, truncated = transition.terminated, transition.truncated
        positions.append(state.position_m)
        speeds.append(abs(state.speed_mps))
        safety_margin = min(
            state.max_speed_mps - state.speed_mps,
            state.speed_mps - state.min_speed_mps,
        )
        safety_margins.append(safety_margin)
        if safety_margin < 0.0:
            safety_violation_positions_m.append(float(state.position_m))

    steps = state.step_count
    stop_error = state.stop_error_m
    time_error = state.operation_time_s - stepper.train_service.schedule_time
    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=stop_error,
        time_error_s=time_error,
        final_speed_mps=state.speed_mps,
        train_service=stepper.train_service,
        terminated=terminated,
        truncated=truncated,
    )
    return PolicyEvaluationResult(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=total_reward,
        total_time_s=state.operation_time_s,
        target_time_s=stepper.train_service.schedule_time,
        total_energy_j=state.energy_consumption_kj * 1000.0,
        start_position_m=stepper.train_service.start_position,
        target_position_m=stepper.train_service.target_position,
        final_position_m=state.position_m,
        final_speed_mps=state.speed_mps,
        stop_error_m=stop_error,
        time_error_s=time_error,
        strict_stop_error_limit_m=get_strict_stop_error_limit_m(stepper.train_service),
        strict_time_error_limit_s=get_strict_time_error_limit_s(stepper.train_service),
        comfort_tav=comfort_tav,
        comfort_er_pct=comfort_exceedance_count / max(steps, 1) * 100.0,
        comfort_rms=float(np.sqrt(comfort_sum_sq / max(steps, 1))),
        terminated=terminated,
        truncated=truncated,
        episode_steps=steps,
        trajectory_pos_m=np.asarray(positions, dtype=np.float32),
        trajectory_speed_mps=np.asarray(speeds, dtype=np.float32),
        min_safety_margin_mps=float(np.min(safety_margins)) if safety_margins else 0.0,
        mean_safety_margin_mps=float(np.mean(safety_margins))
        if safety_margins
        else 0.0,
        safety_violation_positions_m=np.asarray(
            safety_violation_positions_m, dtype=np.float32
        ),
    )


def save_policy_evaluation_curve(
    result: PolicyEvaluationResult,
    output_path: str,
    *,
    extra_metrics: Mapping[str, object] | None = None,
    metrics_path: str | None = None,
) -> tuple[str, str]:
    canonical_fields = {
        "artifact_type",
        "schema_version",
        "created_at",
        "success",
        "precise_arrival",
        "punctual_arrival",
        "total_reward",
        "total_time_s",
        "target_time_s",
        "time_error_s",
        "start_position_m",
        "target_position_m",
        "final_position_m",
        "final_speed_mps",
        "stop_error_m",
        "total_energy_j",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
        "terminated",
        "truncated",
        "episode_steps",
        "min_safety_margin_mps",
        "mean_safety_margin_mps",
        "strict_stop_error_limit_m",
        "strict_time_error_limit_s",
        "selection_comparison_key",
        "selection_rule",
        "extensions",
    }
    num_timesteps = None
    evaluation_rollout_index = None
    extensions: dict[str, object] = {}
    if extra_metrics:
        raw_num_timesteps = extra_metrics.get("num_timesteps")
        if raw_num_timesteps is not None:
            num_timesteps = int(raw_num_timesteps)
        raw_evaluation_rollout_index = extra_metrics.get("evaluation_rollout_index")
        if raw_evaluation_rollout_index is not None:
            evaluation_rollout_index = int(raw_evaluation_rollout_index)
        raw_extensions = extra_metrics.get("extensions")
        if isinstance(raw_extensions, Mapping):
            extensions.update(raw_extensions)
        extensions.update(
            {
                key: value
                for key, value in extra_metrics.items()
                if key not in canonical_fields
                and key not in {"num_timesteps", "evaluation_rollout_index"}
            }
        )
    metrics = result.to_evaluation_metrics(
        extensions=extensions,
        num_timesteps=num_timesteps,
        evaluation_rollout_index=evaluation_rollout_index,
    )
    return save_evaluation_artifact(
        EvaluationArtifact(
            metrics=metrics,
            trajectory=TrajectoryData(
                position_m=result.trajectory_pos_m,
                speed_mps=result.trajectory_speed_mps,
                safety_violation_positions_m=result.safety_violation_positions_m,
            ),
        ),
        output_path,
        metrics_path=metrics_path,
    )


def evaluate_and_save_final_policy(
    model: Any,
    env: gym.Env[np.ndarray, np.ndarray],
    *,
    output_path: str,
    metadata: Mapping[str, object] | None = None,
    deterministic: bool = True,
    metrics_path: str | None = None,
) -> tuple[PolicyEvaluationResult, str, str]:
    """Evaluate a final policy once and persist its canonical final artifacts.

    The caller owns the environment lifecycle.  ``output_path`` should normally
    be ``<final_output_dir>/final_trajectory.npz``; metrics are emitted next to
    it as ``metrics_final.json``.
    """
    result = evaluate_policy_once(model, env, deterministic=deterministic)
    extra_metrics = dict(metadata or {})
    extra_metrics.update(
        {
            "trajectory_source": "final",
            "deterministic": bool(deterministic),
        }
    )
    npz_path, metrics_path = save_policy_evaluation_curve(
        result,
        output_path,
        extra_metrics=extra_metrics,
        metrics_path=metrics_path,
    )
    return result, npz_path, metrics_path


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
    if candidate.precise_arrival and not previous.precise_arrival:
        return "precise_arrival_reached"
    if (
        not candidate.precise_arrival
        and not previous.precise_arrival
        and float(candidate.stop_error_m) < float(previous.stop_error_m)
    ):
        return "lower_stop_error_before_precise_arrival"

    if candidate.punctual_arrival and not previous.punctual_arrival:
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
