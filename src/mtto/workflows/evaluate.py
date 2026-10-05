"""Evaluation workflow entry point for reinforcement learning policies."""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO

import mtto
from mtto.domain.scenario import Scenario, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import QualityReport, assess
from mtto.io.artifacts import (
    BEST_DIR,
    POLICY_ZIP,
    RLResult,
    RunKind,
    RunPayload,
    RunRecord,
    file_sha256,
    read_completed_run,
    task_to_json,
    write_run,
)
from mtto.rl.env import make_env
from mtto.rl.evaluate import run_policy
from mtto.rl.observation import POLICY_IO_VERSION
from mtto.rl.ppo import TORCH_NUM_THREADS
from mtto.rl.rewards import build_reward_config
from mtto.workflows.train import build_env_references

__all__ = [
    "EvaluateConfig",
    "EvaluateResult",
    "evaluation_record_config",
    "evaluate",
]


@dataclass(frozen=True, slots=True)
class EvaluateConfig:
    use_best: bool
    deterministic: bool
    device: str


@dataclass(frozen=True, slots=True, eq=False)
class EvaluateResult:
    run_id: str
    profile: SpeedProfile
    quality: QualityReport
    result: RLResult


def evaluation_record_config(
    config: EvaluateConfig, source_run_id: str
) -> dict[str, object]:
    """Return the config persisted in an evaluation run record."""
    record_config = dataclasses.asdict(config)
    record_config["source_run_id"] = source_run_id
    record_config["source_policy_path"] = (
        f"{BEST_DIR}/{POLICY_ZIP}" if config.use_best else POLICY_ZIP
    )
    return record_config


def evaluate(
    scenario: Scenario,
    task: Task,
    policy_run_dir: str | Path,
    config: EvaluateConfig,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> EvaluateResult:
    """Execute evaluation workflow for a saved training policy."""
    out_dir = Path(output_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    source_run = read_completed_run(policy_run_dir)
    if source_run.record.kind != RunKind.RL_TRAIN:
        raise ValueError(
            f"Source run kind must be {RunKind.RL_TRAIN}, got {source_run.record.kind}"
        )
    if source_run.record.policy_io_version != POLICY_IO_VERSION:
        raise ValueError(
            f"Policy IO version mismatch: expected {POLICY_IO_VERSION}, "
            f"got {source_run.record.policy_io_version}"
        )
    if config.use_best and source_run.payload.best is None:
        raise ValueError(f"Source run does not contain a best policy: {policy_run_dir}")

    policy_rel_path = f"{BEST_DIR}/{POLICY_ZIP}" if config.use_best else POLICY_ZIP
    policy_file = Path(policy_run_dir) / policy_rel_path
    policy_sha = file_sha256(policy_file)

    out_dir.mkdir(parents=True, exist_ok=False)

    src_config = source_run.record.config
    step_time_s = float(src_config["step_time_s"])
    gamma = float(src_config["gamma"])
    reward_preset = str(src_config["reward_preset"])
    reward_config = build_reward_config(reward_preset)

    srtsp_lookup, normalization = build_env_references(scenario, task)

    env = make_env(
        scenario=scenario,
        task=task,
        gamma=gamma,
        step_time_s=step_time_s,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        compact_training_info=False,
        enable_trajectory_tracking=True,
        reward_config=reward_config,
    )

    torch.set_num_threads(TORCH_NUM_THREADS)
    model = PPO.load(str(policy_file), device=config.device)
    try:
        run = run_policy(model, env, deterministic=config.deterministic)
    finally:
        env.close()

    quality = assess(run.profile, scenario, task)

    result = RLResult(
        termination_reason=run.termination_reason,
        truncated=False,
        total_reward=float(run.total_reward),
        steps=int(run.steps),
        final_position_m=float(run.profile.position_m[-1]),
        final_speed_mps=float(run.profile.speed_mps[-1]),
        final_time_s=float(run.profile.time_s[-1]),
        deterministic=bool(run.deterministic),
        policy_run_id=source_run.record.run_id,
        policy_sha256=policy_sha,
        training=None,
    )

    eval_record_config = evaluation_record_config(config, source_run.record.run_id)

    resolved_run_id = str(uuid.uuid4()) if run_id is None else str(run_id)

    record = RunRecord(
        run_id=resolved_run_id,
        kind=RunKind.EVALUATION,
        config=eval_record_config,
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(task),
        policy_io_version=POLICY_IO_VERSION,
        mtto_version=mtto.__version__,
        created_at=datetime.now(UTC).isoformat(),
    )

    payload = RunPayload(
        profile=run.profile,
        quality=quality,
        result=result,
    )

    write_run(out_dir, record, payload)

    return EvaluateResult(
        run_id=resolved_run_id,
        profile=run.profile,
        quality=quality,
        result=result,
    )
