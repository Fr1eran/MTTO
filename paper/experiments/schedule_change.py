"""Evaluate trained candidates under schedule-change cases and rank them."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from mtto.domain.scenario import Scenario, ScheduleChange, Task
from mtto.io.artifacts import read_completed_run
from mtto.io.scenario import load_scenario, load_tasks
from mtto.workflows.evaluate import EvaluateConfig
from paper.experiments.runner import (
    EvaluationRunResult,
    PlannedEvaluation,
    completed_evaluations,
    completed_matrix,
    execute_evaluations,
)
from paper.experiments.spec import ROOT, expand_matrix, load_experiment_spec

RANK_KEY_FIELDS = (
    "feasible_case_count",
    "safe_case_count",
    "success_case_count",
    "precise_case_count",
    "punctual_case_count",
    "negative_safety_violation_count",
    "minimum_safety_margin_mps",
    "negative_max_stop_error_m",
    "negative_max_abs_time_error_s",
    "negative_mean_stop_error_m",
    "negative_mean_abs_time_error_s",
    "negative_mean_energy_j",
)


@dataclass(frozen=True)
class ScheduleSpec:
    source_spec: Path
    source_variant: str
    candidate_sources: tuple[str, ...]
    delta_times_s: tuple[float, ...]
    change_distance_m: float
    output_root: Path


@dataclass(frozen=True)
class ScheduleCase:
    delta_time_s: float
    label: str
    token: str


@dataclass(frozen=True)
class CaseResult:
    case: ScheduleCase
    feasible: bool
    safe: bool
    success: bool
    precise_arrival: bool
    punctual_arrival: bool
    safety_violation_count: int
    min_safety_margin_mps: float
    stop_error_m: float
    abs_time_error_s: float
    time_error_s: float
    total_energy_j: float
    total_energy_kj: float
    total_reward: float
    comfort_tav: float
    run_dir: str
    schedule_change_triggered: bool


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    run_id: str
    seed: int
    source: str
    results: tuple[CaseResult, ...]
    rank_key: tuple[float, ...]


def load_schedule_spec(path: str | Path) -> ScheduleSpec:
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"experiment"} or set(data["experiment"]) != {
        "source_spec",
        "source_variant",
        "candidate_sources",
        "delta_times_s",
        "change_distance_m",
        "output_root",
    }:
        raise ValueError("Invalid schedule-change definition")
    raw = data["experiment"]
    sources = tuple(raw["candidate_sources"])
    deltas = tuple(float(value) for value in raw["delta_times_s"])
    if (
        not sources
        or set(sources) - {"best", "final"}
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("Invalid candidate sources")
    if not deltas or len(set(deltas)) != len(deltas):
        raise ValueError("Invalid delta times")
    return ScheduleSpec(
        source_spec=ROOT / raw["source_spec"],
        source_variant=raw["source_variant"],
        candidate_sources=sources,
        delta_times_s=deltas,
        change_distance_m=float(raw["change_distance_m"]),
        output_root=ROOT / raw["output_root"],
    )


def build_schedule_change_case(delta_time_s: float) -> ScheduleCase:
    delta = float(delta_time_s)
    if delta == 0.0:
        return ScheduleCase(0.0, "Original", "original")
    token = str(abs(delta)).replace(".", "p")
    prefix = "plus" if delta > 0 else "minus"
    return ScheduleCase(
        delta, f"{prefix.title()} {abs(delta):g}s", f"{prefix}_{token}s"
    )


def build_case_task(task: Task, delta_time_s: float, change_distance_m: float) -> Task:
    """Apply a schedule-change delta to ``task`` at an absolute track position.

    ``change_distance_m`` is the absolute track position (m) at which the
    schedule changes, matching the legacy ``--change-distance-m`` semantics
    ("Track position at which the schedule time changes"), not a distance
    relative to the task's start position.
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


def planned_evaluations(
    spec: ScheduleSpec,
) -> tuple[Scenario, tuple[PlannedEvaluation, ...]]:
    source_spec = load_experiment_spec(spec.source_spec)
    scenario = load_scenario(source_spec.scenario, source_spec.line_dir)
    task = load_tasks(source_spec.tasks)[source_spec.task]
    source_dirs = completed_matrix(source_spec)
    selected = [
        (planned, path)
        for planned, path in zip(expand_matrix(source_spec), source_dirs, strict=True)
        if planned.variant.id == spec.source_variant
    ]
    expected = len(source_spec.seeds) * len(spec.candidate_sources)
    if len(selected) * len(spec.candidate_sources) != expected:
        raise ValueError(f"Expected {expected} candidates")
    plans = []
    for planned, source_dir in selected:
        source = read_completed_run(source_dir)
        for policy_source in spec.candidate_sources:
            if policy_source == "best" and source.payload.best is None:
                raise ValueError(f"Missing best policy: {source_dir}")
            candidate_id = (
                f"{spec.source_variant}__seed{planned.seed:04d}__{policy_source}"
            )
            for delta in spec.delta_times_s:
                case = build_schedule_change_case(delta)
                case_task = build_case_task(task, delta, spec.change_distance_m)
                plans.append(
                    PlannedEvaluation(
                        run_label=f"schedule_change__{candidate_id}__{case.token}",
                        policy_run_dir=source_dir,
                        task=case_task,
                        config=EvaluateConfig(
                            use_best=policy_source == "best",
                            deterministic=True,
                            device=planned.config.device,
                        ),
                    )
                )
    return scenario, tuple(plans)


def run(spec_path: str | Path) -> tuple[EvaluationRunResult, ...]:
    spec = load_schedule_spec(spec_path)
    scenario, plans = planned_evaluations(spec)
    return execute_evaluations(spec.output_root, scenario, plans)


def summarize_candidate_results(
    results: tuple[CaseResult, ...],
) -> dict[str, float | int]:
    if not results:
        raise ValueError("candidate evaluation must contain at least one case")
    return {
        "case_count": len(results),
        "feasible_case_count": sum(item.feasible for item in results),
        "safe_case_count": sum(item.safe for item in results),
        "success_case_count": sum(item.success for item in results),
        "precise_case_count": sum(item.precise_arrival for item in results),
        "punctual_case_count": sum(item.punctual_arrival for item in results),
        "safety_violation_count": sum(item.safety_violation_count for item in results),
        "minimum_safety_margin_mps": min(
            item.min_safety_margin_mps for item in results
        ),
        "max_stop_error_m": max(item.stop_error_m for item in results),
        "max_abs_time_error_s": max(item.abs_time_error_s for item in results),
        "mean_stop_error_m": float(np.mean([item.stop_error_m for item in results])),
        "mean_abs_time_error_s": float(
            np.mean([item.abs_time_error_s for item in results])
        ),
        "mean_energy_j": float(np.mean([item.total_energy_j for item in results])),
        "mean_energy_kj": float(np.mean([item.total_energy_kj for item in results])),
        "mean_total_reward": float(np.mean([item.total_reward for item in results])),
    }


def build_candidate_rank_key(results: tuple[CaseResult, ...]) -> tuple[float, ...]:
    aggregate = summarize_candidate_results(results)
    return (
        float(aggregate["feasible_case_count"]),
        float(aggregate["safe_case_count"]),
        float(aggregate["success_case_count"]),
        float(aggregate["precise_case_count"]),
        float(aggregate["punctual_case_count"]),
        -float(aggregate["safety_violation_count"]),
        float(aggregate["minimum_safety_margin_mps"]),
        -float(aggregate["max_stop_error_m"]),
        -float(aggregate["max_abs_time_error_s"]),
        -float(aggregate["mean_stop_error_m"]),
        -float(aggregate["mean_abs_time_error_s"]),
        -float(aggregate["mean_energy_j"]),
    )


def rank_candidate_evaluations(
    evaluations: list[CandidateEvaluation],
) -> list[CandidateEvaluation]:
    ordered = sorted(evaluations, key=lambda item: (item.source != "best", item.run_id))
    ordered.sort(key=lambda item: item.rank_key, reverse=True)
    return ordered


def summarize(spec: ScheduleSpec, run_dirs: tuple[Path, ...]) -> dict[str, object]:
    scenario, plans = planned_evaluations(spec)
    if len(run_dirs) != len(plans):
        raise ValueError("Schedule-change matrix is incomplete")
    candidates = []
    case_count = len(spec.delta_times_s)
    for start in range(0, len(plans), case_count):
        source = read_completed_run(plans[start].policy_run_dir)
        cases = []
        for plan, directory in zip(
            plans[start : start + case_count],
            run_dirs[start : start + case_count],
            strict=True,
        ):
            completed = read_completed_run(directory)
            quality = completed.payload.quality
            result = completed.payload.result
            task = plan.task
            delta = (
                0.0
                if task.schedule_change is None
                else task.schedule_change.new_schedule_time_s - task.schedule_time_s
            )
            cases.append(
                CaseResult(
                    case=build_schedule_change_case(delta),
                    feasible=quality.feasible,
                    safe=quality.safe,
                    success=quality.completed,
                    precise_arrival=quality.precise_stop,
                    punctual_arrival=bool(quality.punctual),
                    safety_violation_count=len(quality.audit.violations),
                    min_safety_margin_mps=quality.audit.min_margin_mps,
                    stop_error_m=quality.metrics.stop_error_m,
                    abs_time_error_s=abs(quality.metrics.arrival_time_error_s),
                    time_error_s=quality.metrics.arrival_time_error_s,
                    total_energy_j=quality.metrics.total_energy_kj * 1000.0,
                    total_energy_kj=quality.metrics.total_energy_kj,
                    total_reward=result.total_reward,
                    comfort_tav=quality.metrics.comfort_tav_mps2,
                    run_dir=str(directory),
                    schedule_change_triggered=task.schedule_change is not None
                    and bool(
                        np.any(
                            completed.payload.profile.position_m
                            >= task.schedule_change.trigger_position_m
                        )
                    ),
                )
            )
        plan = plans[start]
        candidate_id = plan.run_label.removeprefix("schedule_change__").rsplit("__", 1)[
            0
        ]
        candidate = CandidateEvaluation(
            candidate_id=candidate_id,
            run_id=source.record.run_id,
            seed=int(candidate_id.split("__seed")[1].split("__")[0]),
            source="best" if plan.config.use_best else "final",
            results=tuple(cases),
            rank_key=build_candidate_rank_key(tuple(cases)),
        )
        candidates.append(candidate)
    ranked = rank_candidate_evaluations(candidates)
    selected = ranked[0]
    rows = [
        "| Schedule change | Final time error (s) | Stop error (m) "
        "| Trajectory energy (kWh) | Cumulative acceleration variation (m/s²) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in selected.results:
        delta = case.case.delta_time_s
        label = "Original" if delta == 0 else f"{delta:+g} s"
        rows.append(
            f"| {label} | {case.time_error_s:+.4f} | {case.stop_error_m:.4f} | "
            f"{case.total_energy_j / 3_600_000.0:.4f} | {case.comfort_tav:.4f} |"
        )
    rows.append(
        "\n*Note: Cumulative acceleration variation is "
        r"$\sum_t |a_t-a_{t-1}|$ (m/s²).*"
    )
    return {
        "candidate_count": len(ranked),
        "rank_key_fields": list(RANK_KEY_FIELDS),
        "selected": {
            "rank": 1,
            "candidate_id": selected.candidate_id,
            "run_id": selected.run_id,
            "seed": selected.seed,
            "source": selected.source,
            "rank_key": list(selected.rank_key),
            "aggregate_metrics": summarize_candidate_results(selected.results),
        },
        "candidates": [
            {
                "rank": index,
                "candidate_id": candidate.candidate_id,
                "run_id": candidate.run_id,
                "seed": candidate.seed,
                "source": candidate.source,
                "rank_key": list(candidate.rank_key),
                "aggregate_metrics": summarize_candidate_results(candidate.results),
                "cases": [asdict(case) for case in candidate.results],
            }
            for index, candidate in enumerate(ranked, start=1)
        ],
        "cases": [asdict(case) for case in selected.results],
        "table": "\n".join(rows) + "\n",
    }


def completed_runs(spec: ScheduleSpec) -> tuple[Path, ...]:
    scenario, plans = planned_evaluations(spec)
    return completed_evaluations(spec.output_root, scenario, plans)


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
