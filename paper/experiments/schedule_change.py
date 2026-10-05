"""Mid-route schedule change: RL policy inference against DP re-solving.

The representative PPO+PIRS policy (the final policy selected as in the method
ablation) is evaluated with the schedule changed at an absolute track
position. DP re-solves from the state of its nominal solution at the first
grid node at or beyond that position. Both recomputation times are measured on
the current machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import statistics
import time
import tomllib
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

import mtto
from mtto.domain.scenario import Scenario, ScheduleChange, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.dp.solver import VariableSpacingDPOptimizer
from mtto.evaluation.quality import assess, selection_key
from mtto.io.artifacts import (
    POLICY_ZIP,
    RunKind,
    RunPayload,
    RunRecord,
    canonical_json,
    file_sha256,
    read_completed_run,
    task_to_json,
    write_run,
)
from mtto.io.scenario import load_scenario, load_tasks
from mtto.rl.env import MTTOEnv, make_env
from mtto.rl.ppo import TORCH_NUM_THREADS
from mtto.rl.rewards import build_reward_config
from mtto.workflows.dp import DPConfig
from mtto.workflows.evaluate import EvaluateConfig
from mtto.workflows.train import build_env_references
from paper.experiments import runner
from paper.experiments.method_ablation import representative_index
from paper.experiments.runner import (
    EvaluationRunResult,
    PlannedEvaluation,
    completed_evaluations,
    completed_matrix,
    execute_evaluations,
)
from paper.experiments.spec import ROOT, expand_matrix, load_experiment_spec

# Section 5.3 always evaluates a policy trained with the full PIRS reward.
SOURCE_VARIANT = "ppo_pirs"
RL_LABEL = "PPO-PIRS"
DP_LABEL = "DP"
TIMING_JSON = "timing.json"
SPEC_KEYS = {
    "source_spec",
    "dp_run",
    "delta_times_s",
    "change_distance_m",
    "timing_repeats",
    "output_root",
}


@dataclass(frozen=True)
class ScheduleSpec:
    source_spec: Path
    dp_run: Path
    delta_times_s: tuple[float, ...]
    change_distance_m: float
    timing_repeats: int
    output_root: Path


@dataclass(frozen=True)
class ScheduleCase:
    delta_time_s: float
    label: str
    token: str


@dataclass(frozen=True)
class RepresentativePolicy:
    run_dir: Path

    @property
    def policy_file(self) -> Path:
        return self.run_dir / POLICY_ZIP


def load_schedule_spec(path: str | Path) -> ScheduleSpec:
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"experiment"} or set(data["experiment"]) != SPEC_KEYS:
        raise ValueError("Invalid schedule-change definition")
    raw = data["experiment"]
    deltas = tuple(float(value) for value in raw["delta_times_s"])
    if not deltas or len(set(deltas)) != len(deltas):
        raise ValueError("Invalid delta times")
    if raw["timing_repeats"] < 1:
        raise ValueError("timing_repeats must be >= 1")
    return ScheduleSpec(
        source_spec=ROOT / raw["source_spec"],
        dp_run=ROOT / raw["dp_run"],
        delta_times_s=deltas,
        change_distance_m=float(raw["change_distance_m"]),
        timing_repeats=int(raw["timing_repeats"]),
        output_root=ROOT / raw["output_root"],
    )


def build_schedule_change_case(delta_time_s: float) -> ScheduleCase:
    delta = float(delta_time_s)
    if delta == 0.0:
        return ScheduleCase(0.0, "Unchanged", "original")
    token = str(abs(delta)).replace(".", "p")
    prefix = "plus" if delta > 0 else "minus"
    return ScheduleCase(
        delta, f"{prefix.title()} {abs(delta):g}s", f"{prefix}_{token}s"
    )


def build_case_task(task: Task, delta_time_s: float, change_distance_m: float) -> Task:
    """Apply a schedule-change delta to ``task`` at an absolute track position.

    ``change_distance_m`` is the absolute track position (m) at which the
    schedule changes, not a distance relative to the task's start position.
    """
    if delta_time_s == 0.0:
        return task
    return replace(
        task,
        schedule_change=ScheduleChange(
            trigger_position_m=change_distance_m,
            new_schedule_time_s=task.schedule_time_s + delta_time_s,
        ),
    )


def _source_context(spec: ScheduleSpec) -> tuple[Scenario, Task, RepresentativePolicy]:
    source_spec = load_experiment_spec(spec.source_spec)
    scenario = load_scenario(source_spec.scenario, source_spec.line_dir)
    task = load_tasks(source_spec.tasks)[source_spec.task]
    selected = [
        (planned.seed, path)
        for planned, path in zip(
            expand_matrix(source_spec), completed_matrix(source_spec), strict=True
        )
        if planned.variant.id == SOURCE_VARIANT
    ]
    if not selected:
        raise ValueError(f"Variant not in source spec: {SOURCE_VARIANT}")
    metrics = [
        {
            "selection_comparison_key": list(
                selection_key(read_completed_run(path).payload.quality)
            )
        }
        for _, path in selected
    ]
    index = representative_index(metrics, tuple(seed for seed, _ in selected))
    policy = RepresentativePolicy(run_dir=selected[index][1])
    return scenario, task, policy


def planned_evaluations(
    spec: ScheduleSpec,
) -> tuple[Scenario, tuple[PlannedEvaluation, ...]]:
    scenario, task, policy = _source_context(spec)
    source = read_completed_run(policy.run_dir)
    plans = tuple(
        PlannedEvaluation(
            run_label=(
                f"schedule_change__{SOURCE_VARIANT}__"
                f"{build_schedule_change_case(delta).token}"
            ),
            policy_run_dir=policy.run_dir,
            task=build_case_task(task, delta, spec.change_distance_m),
            config=EvaluateConfig(
                use_best=False,
                deterministic=True,
                device=source.record.config["device"],
            ),
        )
        for delta in spec.delta_times_s
    )
    return scenario, plans


def time_rl_replanning(
    model: PPO, env: MTTOEnv, change_distance_m: float
) -> tuple[float, int, float]:
    """Roll the policy out and time the steps from the change point to the stop.

    Returns the total wall time (s) of observation building, inference and
    simulator transitions from the first state at or beyond the change point,
    the number of such steps, and their mean inference time (s).
    """
    task = env.task
    direction = np.sign(task.target_position_m - task.start_position_m)
    state = env.initial_state()
    observation = np.empty(env.observation_builder.OBSERVATION_DIM, dtype=np.float32)
    total_s = 0.0
    inference_s = []
    while True:
        timed = (state.s_m - change_distance_m) * direction >= 0.0
        start = time.perf_counter()
        env.observation_builder.build(state, out=observation)
        action, _ = model.predict(observation, deterministic=True)
        inferred = time.perf_counter()
        acceleration = env.observation_builder.denormalize_action(
            float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        )
        result = env.transition(state, acceleration)
        finished = time.perf_counter()
        if timed:
            total_s += finished - start
            inference_s.append(inferred - start)
        state = result.next_state
        if result.termination_reason is not None:
            break
    if not inference_s:
        raise ValueError("The policy stopped before reaching the change point")
    return total_s, len(inference_s), float(np.mean(inference_s))


def _replan_key(
    spec: ScheduleSpec, scenario: Scenario, policy: RepresentativePolicy
) -> str:
    dp_run = read_completed_run(spec.dp_run)
    inputs = {
        "scenario_hash": scenario.scenario_hash,
        "policy_sha256": file_sha256(policy.policy_file),
        "dp_run_id": dp_run.record.run_id,
        "delta_times_s": list(spec.delta_times_s),
        "change_distance_m": spec.change_distance_m,
        "timing_repeats": spec.timing_repeats,
        "mtto_version": mtto.__version__,
    }
    return hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()


def replan_directory(spec: ScheduleSpec, key: str) -> Path:
    return spec.output_root / f"replan__{key[:16]}"


def run_replanning(spec: ScheduleSpec) -> tuple[Path, bool]:
    """Re-solve DP from the change point and time both recomputations."""
    scenario, task, policy = _source_context(spec)
    key = _replan_key(spec, scenario, policy)
    directory = replan_directory(spec, key)
    if (directory / TIMING_JSON).exists():
        return directory, True
    if directory.exists():
        shutil.rmtree(directory)
    dp_run = read_completed_run(spec.dp_run)
    if dp_run.record.kind != RunKind.DP_SOLVE:
        raise ValueError(f"Not a DP run: {spec.dp_run}")
    if dp_run.record.scenario_hash != scenario.scenario_hash:
        raise ValueError(f"Scenario hash mismatch for DP run: {spec.dp_run}")
    dp_task = dp_run.record.task
    if (
        dp_task["start_position_m"] != task.start_position_m
        or dp_task["target_position_m"] != task.target_position_m
        or dp_task["schedule_time_s"] != task.schedule_time_s
    ):
        raise ValueError(f"Task mismatch between DP run and source task: {spec.dp_run}")
    commit, dirty = runner.git_state()
    directory.mkdir(parents=True)

    nominal = dp_run.payload.profile
    direction = np.sign(task.target_position_m - task.start_position_m)
    reached = np.flatnonzero(
        (nominal.position_m - spec.change_distance_m) * direction >= 0.0
    )
    if not reached.size or reached[0] == nominal.position_m.size - 1:
        raise ValueError("The change point is not inside the DP profile")
    node = int(reached[0])
    start_position = float(nominal.position_m[node])
    start_speed = float(nominal.speed_mps[node])
    start_time = float(nominal.time_s[node])
    dp_config = DPConfig(**dp_run.record.config)

    source = read_completed_run(policy.run_dir)
    source_config = source.record.config
    torch.set_num_threads(TORCH_NUM_THREADS)
    model = PPO.load(str(policy.policy_file), device=source_config["device"])
    reward_config = build_reward_config(str(source_config["reward_preset"]))

    cases = []
    for delta in spec.delta_times_s:
        case = build_schedule_change_case(delta)
        case_task = build_case_task(task, delta, spec.change_distance_m)
        srtsp_lookup, normalization = build_env_references(scenario, case_task)
        env = make_env(
            scenario=scenario,
            task=case_task,
            gamma=float(source_config["gamma"]),
            step_time_s=float(source_config["step_time_s"]),
            srtsp_lookup=srtsp_lookup,
            normalization=normalization,
            reward_config=reward_config,
        )
        rl_runs = [
            time_rl_replanning(model, env, spec.change_distance_m)
            for _ in range(spec.timing_repeats)
        ]
        env.close()

        remaining_time = task.schedule_time_s + delta - start_time
        cold_s = []
        cached_s = []
        tail = None
        for _ in range(spec.timing_repeats):
            optimizer = VariableSpacingDPOptimizer(
                scenario=scenario,
                task=task,
                cache_dir=None,
                delta_speed=dp_config.delta_speed,
                max_outer_iterations=dp_config.max_outer_iterations,
                show_precompute_progress=False,
                precompute_mode=dp_config.precompute_mode,
                precompute_workers=dp_config.precompute_workers,
                precompute_chunk_size=dp_config.precompute_chunk_size,
                stage_division=dp_config.stage_division,
                uniform_step_size=dp_config.uniform_step_size,
                sub_stage_count=dp_config.sub_stage_count,
            )
            # The first call builds the transition graph; the second reuses the
            # graph kept in memory and measures the lambda search alone.
            for timings in (cold_s, cached_s):
                started = time.perf_counter()
                tail = optimizer.optimize(
                    start_position,
                    start_speed,
                    float(task.target_position_m),
                    0.0,
                    remaining_time,
                )
                timings.append(time.perf_counter() - started)
        if tail is None:
            raise RuntimeError(f"DP re-solving found no trajectory: {case.label}")
        profile = SpeedProfile.from_arrays(
            np.concatenate((nominal.position_m[:node], tail.position_m)),
            np.concatenate((nominal.speed_mps[:node], tail.speed_mps)),
            np.concatenate((nominal.time_s[:node], tail.time_s + start_time)),
            np.concatenate(
                (
                    nominal.propulsion_energy_kj[:node],
                    tail.propulsion_energy_kj + nominal.propulsion_energy_kj[node],
                )
            ),
            np.concatenate(
                (
                    nominal.levitation_energy_kj[:node],
                    tail.levitation_energy_kj + nominal.levitation_energy_kj[node],
                )
            ),
        )
        write_run(
            directory / f"dp__{case.token}",
            RunRecord(
                run_id=str(uuid.uuid4()),
                kind=RunKind.DP_SOLVE,
                config={
                    **dataclasses.asdict(dp_config),
                    "replanned_from_run_id": dp_run.record.run_id,
                    "replan_start_position_m": start_position,
                    "replan_start_speed_mps": start_speed,
                    "replan_start_time_s": start_time,
                    "remaining_schedule_time_s": remaining_time,
                },
                scenario_hash=scenario.scenario_hash,
                task=task_to_json(case_task),
                policy_io_version=None,
                mtto_version=mtto.__version__,
                created_at=datetime.now(UTC).isoformat(),
            ),
            RunPayload(profile=profile, quality=assess(profile, scenario, case_task)),
        )
        cases.append(
            {
                "token": case.token,
                "delta_time_s": case.delta_time_s,
                "rl_total_s": [item[0] for item in rl_runs],
                "rl_steps": rl_runs[0][1],
                "rl_mean_inference_s": [item[2] for item in rl_runs],
                "dp_cold_s": cold_s,
                "dp_cached_s": cached_s,
            }
        )
    timing = {
        "reuse_key": key,
        "git_commit": commit,
        "dirty": dirty,
        "policy_run_dir": str(policy.run_dir),
        "dp_run": str(spec.dp_run),
        "dp_run_id": dp_run.record.run_id,
        "dp_precompute_mode": dp_config.precompute_mode,
        "dp_precompute_workers": dp_config.precompute_workers,
        "torch_threads": torch.get_num_threads(),
        "change_node": {
            "position_m": start_position,
            "speed_mps": start_speed,
            "time_s": start_time,
        },
        "timing_repeats": spec.timing_repeats,
        "cases": cases,
    }
    (directory / TIMING_JSON).write_text(
        json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return directory, False


def run(spec_path: str | Path) -> tuple[tuple[EvaluationRunResult, ...], Path, bool]:
    spec = load_schedule_spec(spec_path)
    scenario, plans = planned_evaluations(spec)
    evaluations = execute_evaluations(spec.output_root, scenario, plans)
    directory, reused = run_replanning(spec)
    return evaluations, directory, reused


def _case_metrics(completed_dir: Path) -> dict[str, Any]:
    quality = read_completed_run(completed_dir).payload.quality
    return {
        "feasible": quality.feasible,
        "safe": quality.safe,
        "time_error_s": quality.metrics.arrival_time_error_s,
        "stop_error_m": abs(quality.metrics.stop_error_m),
        "total_energy_kwh": quality.metrics.total_energy_kj / 3600.0,
        "comfort_tav": quality.metrics.comfort_tav_mps2,
        "run_dir": str(completed_dir),
    }


def summarize(
    spec: ScheduleSpec, run_dirs: tuple[Path, ...], replan_dir: Path
) -> dict[str, object]:
    if len(run_dirs) != len(spec.delta_times_s):
        raise ValueError("Schedule-change evaluations are incomplete")
    timing = json.loads((replan_dir / TIMING_JSON).read_text(encoding="utf-8"))
    rows = [
        "| Schedule change | Method | Within stop/time tolerance | Δt (s) "
        "| Stop error (m) | Total energy (kWh) "
        "| Cumulative acceleration variation (m/s²) | Recomputation time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    entries = []
    for delta, run_dir, case_timing in zip(
        spec.delta_times_s, run_dirs, timing["cases"], strict=True
    ):
        case = build_schedule_change_case(delta)
        rl_total = statistics.median(case_timing["rl_total_s"])
        rl_inference = statistics.median(case_timing["rl_mean_inference_s"])
        dp_cold = statistics.median(case_timing["dp_cold_s"])
        dp_cached = statistics.median(case_timing["dp_cached_s"])
        for method, directory, recompute in (
            (
                RL_LABEL,
                run_dir,
                {
                    "recompute_time_s": rl_total,
                    "steps_after_change": case_timing["rl_steps"],
                    "mean_inference_ms": rl_inference * 1000.0,
                },
            ),
            (
                DP_LABEL,
                replan_dir / f"dp__{case.token}",
                {"recompute_time_s": dp_cold, "cached_graph_time_s": dp_cached},
            ),
        ):
            entry = {
                "case": dataclasses.asdict(case),
                "method": method,
                **_case_metrics(directory),
                **recompute,
            }
            entries.append(entry)
            time_cell = (
                f"{rl_total:.3f} ({rl_inference * 1000.0:.2f} ms/step)"
                if method == RL_LABEL
                else f"{dp_cold:.2f} (cached graph {dp_cached:.2f})"
            )
            label = "Unchanged" if delta == 0 else f"{delta:+g} s"
            rows.append(
                f"| {label} | {method} | {'Yes' if entry['feasible'] else 'No'} | "
                f"{entry['time_error_s']:+.2f} | {entry['stop_error_m']:.3f} | "
                f"{entry['total_energy_kwh']:.1f} | {entry['comfort_tav']:.3f} | "
                f"{time_cell} |"
            )
    workers = timing["dp_precompute_workers"]
    rows.append(
        "\n*Note: The schedule changes at "
        f"{spec.change_distance_m / 1000.0:g} km; Δt is measured against the new "
        "schedule (positive = late) and energy covers the whole run. PPO-PIRS time "
        "is the wall time of observation building, policy inference and simulator "
        "transitions from the change point to the stop (mean inference time per "
        "step in parentheses). DP time is re-solving from the state of its nominal "
        "solution at the first grid node beyond the change point, including "
        "transition-graph construction and the lambda search (lambda search alone "
        "with the graph kept in memory in parentheses). Medians of "
        f"{timing['timing_repeats']} repetitions; DP graph precomputation mode "
        f"{timing['dp_precompute_mode']}"
        + (f" with {workers} workers" if workers is not None else "")
        + f", PyTorch threads {timing['torch_threads']}.*"
    )
    return {
        "change_distance_m": spec.change_distance_m,
        "replan_dir": str(replan_dir),
        "replan_dirty": timing["dirty"],
        "policy_run_dir": timing["policy_run_dir"],
        "change_node": timing["change_node"],
        "entries": entries,
        "table": "\n".join(rows) + "\n",
    }


def completed_runs(spec: ScheduleSpec) -> tuple[tuple[Path, ...], Path]:
    scenario, _, policy = _source_context(spec)
    _, plans = planned_evaluations(spec)
    replan_dir = replan_directory(spec, _replan_key(spec, scenario, policy))
    if not (replan_dir / TIMING_JSON).exists():
        raise FileNotFoundError(f"No completed DP re-solving: {replan_dir}")
    return completed_evaluations(spec.output_root, scenario, plans), replan_dir


def write_summary(summary: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "schedule_time_change_summary.json").write_text(
        json.dumps(
            {key: value for key, value in summary.items() if key != "table"},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "schedule_time_change_table.md").write_text(
        summary["table"], encoding="utf-8"
    )
