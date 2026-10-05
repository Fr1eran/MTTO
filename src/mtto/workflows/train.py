"""Training workflow entry point for reinforcement learning."""

from __future__ import annotations

import dataclasses
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import mtto
from mtto.domain.energy import segment_energy
from mtto.domain.kinematics import run_time
from mtto.domain.scenario import Scenario, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.domain.srtsp import (
    SrtspLookup,
    build_srtsp_lookup,
    min_operation_time_curve,
    min_remaining_time_s,
)
from mtto.evaluation.quality import QualityReport, assess
from mtto.io.artifacts import (
    BEST_DIR,
    POLICY_ZIP,
    RLResult,
    RunKind,
    RunPayload,
    RunRecord,
    TrainingOutcome,
    file_sha256,
    task_to_json,
    write_run,
)
from mtto.rl.diagnostics import TrainingDiagnostics
from mtto.rl.env import MTTOEnv, make_env
from mtto.rl.evaluate import run_policy
from mtto.rl.observation import POLICY_IO_VERSION
from mtto.rl.ppo import (
    TORCH_NUM_THREADS,
    CompletedEpisodeProgress,
    RewardDiagnosticsCallback,
    SafetyTruncationHistogramCallback,
    ScheduledPolicyEvaluationCallback,
    StopTrainingOnCompletedEpisodes,
    build_ppo,
    completed_episode_cosine_annealing_schedule,
    environment_step_cosine_annealing_schedule,
)
from mtto.rl.rewards import (
    RewardNormalization,
    build_reward_config,
    reward_config_parameters,
)

__all__ = [
    "TrainConfig",
    "Progress",
    "TrainResult",
    "build_env_references",
    "derive_training_budget",
    "training_record_config",
    "train",
]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    reward_preset: str
    step_time_s: float
    gamma: float
    budget_mode: Literal["completed_episodes", "environment_steps"]
    training_episodes: int | None
    training_rollouts: int | None
    num_envs: int
    n_steps_per_env: int
    evaluation_interval_rollouts: int | None
    evaluation_interval_episodes: int | None
    evaluation_deterministic: bool
    keep_best: bool
    safety_truncation_bin_size_m: float
    device: str
    seed: int | None


def training_record_config(config: TrainConfig) -> dict[str, object]:
    """Return the config persisted in a training run record."""
    record_config = dataclasses.asdict(config)
    record_config["reward"] = reward_config_parameters(
        build_reward_config(config.reward_preset)
    )
    return record_config


@dataclass(frozen=True, slots=True)
class Progress:
    training_timesteps: int
    total_timesteps: int
    completed_episodes: int


@dataclass(frozen=True, slots=True, eq=False)
class TrainResult:
    run_id: str
    profile: SpeedProfile
    quality: QualityReport
    result: RLResult
    best: RunPayload | None


def build_env_references(
    scenario: Scenario, task: Task
) -> tuple[SrtspLookup, RewardNormalization]:
    """Compute task-level SRTSP lookup and reward normalization once."""
    if task.schedule_time_s is None:
        raise ValueError("task.schedule_time_s must not be None")
    vehicle = scenario.vehicle
    line = scenario.line
    factor = scenario.safeguard.params.factor
    # The upper curve accelerates from rest at the track origin, so a train
    # leaving the task start from rest always stays below it.
    profile_pos, profile_speed = min_operation_time_curve(
        vehicle=vehicle,
        track=line,
        factor=factor,
        begin_pos=0.0,
        begin_speed=0.0,
        end_pos=task.target_position_m + task.max_stop_error_m * 20,
        end_speed=0.0,
    )
    lookup = build_srtsp_lookup(profile_pos, profile_speed)
    # Peak traction energy per metre: full acceleration over a short step from
    # every moving speed (a 0.1 s step from rest covers only 5 mm and its
    # per-metre loss is a discretisation artefact), at the start of the task
    # and after each slope breakpoint inside it (the step energy uses the
    # slope at its start point only).
    breakpoints = line.slope_intervals[
        (line.slope_intervals > task.start_position_m)
        & (line.slope_intervals < task.target_position_m)
    ]
    peak_propulsion_kj_per_m = 0.0
    for begin_pos in (task.start_position_m, *breakpoints):
        for speed in np.arange(0.5, vehicle.max_speed + 0.25, 0.5):
            _, distance, duration = run_time(float(speed), vehicle.max_acc, 0.1)
            propulsion, _ = segment_energy(
                scenario.energy,
                vehicle,
                line,
                begin_pos=float(begin_pos),
                begin_speed=float(speed),
                acc=vehicle.max_acc,
                distance=distance,
                direction=1,
                operation_time=duration,
            )
            peak_propulsion_kj_per_m = max(
                peak_propulsion_kj_per_m, propulsion / distance
            )
    min_remaining = min_remaining_time_s(
        vehicle, line, factor, task.start_position_m, 0.0, task.target_position_m
    )
    initial_min_operation_time_s = task.schedule_time_s - (
        task.schedule_time_s - 0.0 - min_remaining
    )
    return lookup, RewardNormalization(
        peak_propulsion_kj_per_m=peak_propulsion_kj_per_m,
        initial_min_operation_time_s=initial_min_operation_time_s,
    )


def derive_training_budget(
    *,
    training_episodes: int,
    num_envs: int,
    step_time_s: float,
    episode_time_limit_s: float,
    rollout_steps_per_update: int,
) -> tuple[int, int, int]:
    """Resolve the effective episode target, max episode steps, and timestep ceiling."""
    if training_episodes <= 0:
        raise ValueError("training_episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not math.isfinite(step_time_s) or step_time_s <= 0.0:
        raise ValueError("step_time_s must be finite and positive")
    if not math.isfinite(episode_time_limit_s) or episode_time_limit_s <= 0.0:
        raise ValueError("episode_time_limit_s must be finite and positive")
    if rollout_steps_per_update <= 0:
        raise ValueError("rollout_steps_per_update must be positive")

    effective_training_episodes = math.ceil(training_episodes / num_envs) * num_envs
    max_episode_steps = math.ceil(episode_time_limit_s / step_time_s)
    raw_total_timesteps = effective_training_episodes * max_episode_steps
    derived_total_timesteps = (
        math.ceil(raw_total_timesteps / rollout_steps_per_update)
        * rollout_steps_per_update
    )
    return (
        int(effective_training_episodes),
        int(max_episode_steps),
        int(derived_total_timesteps),
    )


def train(
    scenario: Scenario,
    task: Task,
    config: TrainConfig,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    tensorboard_log_dir: str | Path | None = None,
    progress: Callable[[Progress], None] | None = None,
) -> TrainResult:
    """Execute RL training workflow and write canonical run artifacts."""
    out_dir = Path(output_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    resolved_run_id = str(uuid.uuid4()) if run_id is None else str(run_id)

    if task.schedule_time_s is None:
        raise ValueError("task.schedule_time_s must not be None for training")
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.n_steps_per_env <= 0:
        raise ValueError("n_steps_per_env must be positive")
    if not math.isfinite(config.step_time_s) or config.step_time_s <= 0.0:
        raise ValueError("step_time_s must be finite and positive")
    if not math.isfinite(config.gamma) or config.gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if (
        config.evaluation_interval_rollouts is not None
        and config.evaluation_interval_episodes is not None
    ):
        raise ValueError(
            "At most one of evaluation_interval_rollouts and "
            "evaluation_interval_episodes can be configured"
        )
    if config.budget_mode == "completed_episodes":
        if config.training_episodes is None or config.training_episodes <= 0:
            raise ValueError(
                "training_episodes must be positive in completed_episodes budget_mode"
            )
    elif config.budget_mode == "environment_steps":
        if config.training_rollouts is None or config.training_rollouts <= 0:
            raise ValueError(
                "training_rollouts must be positive in environment_steps budget_mode"
            )
    else:
        raise ValueError(f"Unknown budget_mode: {config.budget_mode}")

    torch.set_num_threads(TORCH_NUM_THREADS)
    out_dir.mkdir(parents=True, exist_ok=False)

    reward_config = build_reward_config(config.reward_preset)
    srtsp_lookup, normalization = build_env_references(scenario, task)

    if config.seed is not None:
        set_random_seed(
            seed=config.seed,
            using_cuda=(config.device == "cuda"),
        )

    rollout_steps_per_update = config.num_envs * config.n_steps_per_env
    episode_progress: CompletedEpisodeProgress | None = None
    episode_stop_callback: StopTrainingOnCompletedEpisodes | None = None
    effective_training_episodes: int | None = None

    if config.budget_mode == "completed_episodes":
        assert config.training_episodes is not None
        # Episodes have no time limit; the strict deadline of the latest
        # schedule only sizes the timestep ceiling of the episode budget.
        latest_schedule_time_s = task.schedule_time_s
        if task.schedule_change is not None:
            latest_schedule_time_s = max(
                latest_schedule_time_s, task.schedule_change.new_schedule_time_s
            )
        effective_training_episodes, _, total_timesteps = derive_training_budget(
            training_episodes=config.training_episodes,
            num_envs=config.num_envs,
            step_time_s=config.step_time_s,
            episode_time_limit_s=latest_schedule_time_s + task.max_arr_time_error_s,
            rollout_steps_per_update=rollout_steps_per_update,
        )
        episode_progress = CompletedEpisodeProgress(effective_training_episodes)
        episode_stop_callback = StopTrainingOnCompletedEpisodes(episode_progress)
        learning_rate = completed_episode_cosine_annealing_schedule(episode_progress)
    else:
        assert config.training_rollouts is not None
        total_timesteps = config.training_rollouts * rollout_steps_per_update
        learning_rate = environment_step_cosine_annealing_schedule()

    def _make_training_env(worker_rank: int) -> Callable[[], MTTOEnv]:
        def _init() -> MTTOEnv:
            return make_env(
                scenario=scenario,
                task=task,
                gamma=config.gamma,
                step_time_s=config.step_time_s,
                srtsp_lookup=srtsp_lookup,
                normalization=normalization,
                compact_training_info=True,
                reward_config=reward_config,
                enable_safety_truncation_tracking=True,
                reward_diagnostics_worker_rank=worker_rank,
                reward_diagnostics_rollout_capacity=config.n_steps_per_env,
            )

        return _init

    venv_train = DummyVecEnv([_make_training_env(i) for i in range(config.num_envs)])

    model = build_ppo(
        venv_train,
        device=config.device,
        n_steps=config.n_steps_per_env,
        gamma=config.gamma,
        learning_rate=learning_rate,
        tensorboard_log=(
            str(tensorboard_log_dir) if tensorboard_log_dir is not None else None
        ),
    )

    callbacks: list[BaseCallback] = []
    if episode_stop_callback is not None:
        callbacks.append(episode_stop_callback)

    reward_diagnostics_callback = RewardDiagnosticsCallback()
    callbacks.append(reward_diagnostics_callback)

    safety_callback = SafetyTruncationHistogramCallback(
        position_bin_size_m=config.safety_truncation_bin_size_m,
    )
    callbacks.append(safety_callback)

    eval_env = make_env(
        scenario=scenario,
        task=task,
        gamma=config.gamma,
        step_time_s=config.step_time_s,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        compact_training_info=False,
        enable_trajectory_tracking=True,
        reward_config=reward_config,
    )

    scheduled_eval_callback = ScheduledPolicyEvaluationCallback(
        eval_env=eval_env,
        evaluation_interval_rollouts=config.evaluation_interval_rollouts,
        evaluation_interval_episodes=config.evaluation_interval_episodes,
        deterministic=config.evaluation_deterministic,
        get_completed_training_episodes=(
            (lambda: episode_progress.completed_episodes)
            if episode_progress is not None
            else (lambda: reward_diagnostics_callback.completed_episode_count)
        ),
        max_rollouts_exclusive=(
            config.training_rollouts
            if config.budget_mode == "environment_steps"
            else None
        ),
        max_completed_episodes_exclusive=(
            int(effective_training_episodes)
            if config.budget_mode == "completed_episodes"
            and effective_training_episodes is not None
            else None
        ),
    )
    callbacks.append(scheduled_eval_callback)

    if progress is not None:
        progress_fn = progress

        class _ProgressCallback(BaseCallback):
            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                completed = (
                    episode_progress.completed_episodes
                    if episode_progress is not None
                    else reward_diagnostics_callback.completed_episode_count
                )
                progress_fn(
                    Progress(
                        training_timesteps=int(self.model.num_timesteps),
                        total_timesteps=total_timesteps,
                        completed_episodes=int(completed),
                    )
                )

        callbacks.append(_ProgressCallback())

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
        )
    finally:
        venv_train.close()
        eval_env.close()

    actual_completed_episodes = int(reward_diagnostics_callback.completed_episode_count)
    actual_training_timesteps = int(model.num_timesteps)
    actual_training_rollouts = (
        actual_training_timesteps // rollout_steps_per_update
        if config.budget_mode == "environment_steps"
        else None
    )
    if config.budget_mode == "completed_episodes":
        assert episode_progress is not None
        assert effective_training_episodes is not None
        target_reached = bool(
            episode_progress.completed_episodes >= effective_training_episodes
        )
        stop_reason = (
            "completed_episode_target" if target_reached else "derived_timestep_ceiling"
        )
    else:
        target_reached = bool(actual_training_timesteps >= total_timesteps)
        stop_reason = (
            "environment_step_target"
            if target_reached
            else "training_stopped_before_step_target"
        )

    training_outcome = TrainingOutcome(
        actual_training_timesteps=actual_training_timesteps,
        actual_training_rollouts=actual_training_rollouts,
        actual_completed_episodes=actual_completed_episodes,
        target_reached=target_reached,
        stop_reason=stop_reason,
    )

    final_policy_path = out_dir / POLICY_ZIP
    model.save(str(final_policy_path))
    final_policy_sha = file_sha256(final_policy_path)

    final_eval_env = make_env(
        scenario=scenario,
        task=task,
        gamma=config.gamma,
        step_time_s=config.step_time_s,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        compact_training_info=False,
        enable_trajectory_tracking=True,
        reward_config=reward_config,
    )
    try:
        final_run = run_policy(model, final_eval_env, deterministic=True)
    finally:
        final_eval_env.close()

    final_quality = assess(final_run.profile, scenario, task)

    final_result = RLResult(
        termination_reason=final_run.termination_reason,
        truncated=False,
        total_reward=float(final_run.total_reward),
        steps=int(final_run.steps),
        final_position_m=float(final_run.profile.position_m[-1]),
        final_speed_mps=float(final_run.profile.speed_mps[-1]),
        final_time_s=float(final_run.profile.time_s[-1]),
        deterministic=bool(final_run.deterministic),
        policy_run_id=resolved_run_id,
        policy_sha256=final_policy_sha,
        training=training_outcome,
    )

    best_payload: RunPayload | None = None
    if config.keep_best and scheduled_eval_callback.best is not None:
        best = scheduled_eval_callback.best
        model.set_parameters(best.parameters)
        best_dir = out_dir / BEST_DIR
        best_dir.mkdir(parents=True, exist_ok=True)
        best_policy_path = best_dir / POLICY_ZIP
        model.save(str(best_policy_path))
        best_sha = file_sha256(best_policy_path)

        best_result = RLResult(
            termination_reason=best.run.termination_reason,
            truncated=False,
            total_reward=float(best.run.total_reward),
            steps=int(best.run.steps),
            final_position_m=float(best.run.profile.position_m[-1]),
            final_speed_mps=float(best.run.profile.speed_mps[-1]),
            final_time_s=float(best.run.profile.time_s[-1]),
            deterministic=bool(best.run.deterministic),
            policy_run_id=resolved_run_id,
            policy_sha256=best_sha,
            training=None,
        )
        best_payload = RunPayload(
            profile=best.run.profile,
            quality=best.quality,
            result=best_result,
        )

    record_config = training_record_config(config)

    record = RunRecord(
        run_id=resolved_run_id,
        kind=RunKind.RL_TRAIN,
        config=record_config,
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(task),
        policy_io_version=POLICY_IO_VERSION,
        mtto_version=mtto.__version__,
        created_at=datetime.now(UTC).isoformat(),
    )

    training_diagnostics = TrainingDiagnostics(
        reward=reward_diagnostics_callback.diagnostics,
        safety=safety_callback.histogram,
    )
    evaluation_history = scheduled_eval_callback.history

    payload = RunPayload(
        profile=final_run.profile,
        quality=final_quality,
        result=final_result,
        best=best_payload,
        diagnostics=training_diagnostics,
        evaluations=evaluation_history,
    )

    write_run(out_dir, record, payload)

    return TrainResult(
        run_id=resolved_run_id,
        profile=final_run.profile,
        quality=final_quality,
        result=final_result,
        best=best_payload,
    )
