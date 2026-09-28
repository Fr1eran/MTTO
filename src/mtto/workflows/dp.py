"""Dynamic programming workflow entry point."""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import mtto
from mtto.domain.scenario import Scenario, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.dp.solver import VariableSpacingDPOptimizer
from mtto.evaluation.quality import QualityReport, assess
from mtto.io.artifacts import (
    RunKind,
    RunPayload,
    RunRecord,
    task_to_json,
    write_run,
)

__all__ = [
    "DPConfig",
    "DPResult",
    "solve_dp",
]


@dataclass(frozen=True, slots=True)
class DPConfig:
    delta_speed: float
    stage_division: Literal["variable", "uniform"]
    uniform_step_size: float
    sub_stage_count: int
    max_outer_iterations: int
    precompute_mode: Literal["serial", "parallel"]
    precompute_workers: int | None
    precompute_chunk_size: int | None


@dataclass(frozen=True, slots=True, eq=False)
class DPResult:
    run_id: str
    profile: SpeedProfile
    quality: QualityReport


def solve_dp(
    scenario: Scenario,
    task: Task,
    config: DPConfig,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    run_id: str | None = None,
) -> DPResult:
    """Execute dynamic programming optimization workflow and write run artifacts."""
    out_dir = Path(output_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    if task.schedule_time_s is None:
        raise ValueError("task.schedule_time_s must not be None for DP")

    optimizer = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        cache_dir=cache_dir,
        delta_speed=config.delta_speed,
        max_outer_iterations=config.max_outer_iterations,
        show_precompute_progress=False,
        precompute_mode=config.precompute_mode,
        precompute_workers=config.precompute_workers,
        precompute_chunk_size=config.precompute_chunk_size,
        stage_division=config.stage_division,
        uniform_step_size=config.uniform_step_size,
        sub_stage_count=config.sub_stage_count,
    )
    profile = optimizer.optimize(
        start_pos=float(task.start_position_m),
        start_speed=0.0,
        target_pos=float(task.target_position_m),
        target_speed=0.0,
        schedule_time=float(task.schedule_time_s),
    )
    if profile is None:
        raise RuntimeError("DP optimization did not find a feasible trajectory")

    out_dir.mkdir(parents=True, exist_ok=False)

    quality = assess(profile, scenario, task)
    resolved_run_id = str(uuid.uuid4()) if run_id is None else str(run_id)

    record = RunRecord(
        run_id=resolved_run_id,
        kind=RunKind.DP_SOLVE,
        config=dataclasses.asdict(config),
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(task),
        policy_io_version=None,
        mtto_version=mtto.__version__,
        created_at=datetime.now(UTC).isoformat(),
    )
    payload = RunPayload(
        profile=profile,
        quality=quality,
    )
    write_run(out_dir, record, payload)

    return DPResult(
        run_id=resolved_run_id,
        profile=profile,
        quality=quality,
    )
