"""Temporal control-step ablation from completed training artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtto.io.artifacts import RunPayload, read_completed_run
from paper.analysis import aggregate_matrix
from paper.experiments.method_ablation import LATE_STAGE_START_FRACTION
from paper.experiments.runner import RunResult, execute_matrix
from paper.experiments.spec import ExperimentSpec, load_experiment_spec

POLICY_SOURCES = ("best", "final")
METRIC_KEYS = ("stop_error_m", "time_error_s", "energy_kwh", "comfort_tav")
# Reliability gate: minimum mean share of strictly feasible periodic evaluations
# in the last quarter of training.
MIN_LATE_FEASIBLE_RATE = 0.9


def run(spec_path: str | Path, workers: int = 1) -> tuple[RunResult, ...]:
    return execute_matrix(load_experiment_spec(spec_path), workers)


def _policy_metrics(payload: RunPayload) -> dict[str, object]:
    quality = payload.quality
    metrics = quality.metrics
    return {
        "feasible": quality.feasible,
        "safe": quality.safe,
        "success": quality.completed,
        "stop_error_m": abs(metrics.stop_error_m),
        "time_error_s": metrics.arrival_time_error_s,
        "energy_kwh": metrics.total_energy_kj / 3600.0,
        "comfort_tav": metrics.comfort_tav_mps2,
    }


def _source_statistics(seed_data: list[dict[str, object]]) -> dict[str, object]:
    """Feasibility over all runs; trajectory metrics over runs that arrived."""
    feasible = [item for item in seed_data if item["feasible"]]
    arrived = [item for item in seed_data if item["success"]]
    metric_stats = {}
    for key in METRIC_KEYS:
        values = np.asarray([item[key] for item in arrived], dtype=float)
        mean, std, _ = aggregate_matrix(values.reshape(-1, 1))
        metric_stats[key] = (
            {"mean": float(mean[0]), "std": float(std[0])} if arrived else None
        )
    return {
        "feasible_count": len(feasible),
        "feasible_rate": len(feasible) / len(seed_data),
        "arrived_count": len(arrived),
        "safe_count": sum(item["safe"] for item in seed_data),
        "metrics": metric_stats,
    }


def select_step_time(variants: list[dict[str, object]]) -> dict[str, object]:
    """Pick the control period by reliability, energy, then convergence stability.

    1. Gate: best and final policies strictly feasible in every run and a mean
       late-stage feasible-evaluation share of at least MIN_LATE_FEASIBLE_RATE.
    2. Energy: candidates whose mean final-policy energy lies within the pooled
       across-run standard deviation of the lowest one are equivalent. Final
       policies, not best checkpoints, are compared: the best checkpoint is the
       lowest-energy feasible evaluation and tends to spend the timetable
       tolerance on arriving late, so its energy mixes in the time error.
    3. Stability: among equivalent candidates, the smallest across-run standard
       deviation of final-policy energy, then the smallest mean energy drift
       from best to final, then the shorter control period.
    """
    excluded = {}
    candidates = []
    for item in variants:
        total = len(item["best"]["per_seed"])
        if item["best"]["feasible_count"] < total:
            excluded[item["variant_id"]] = "best policy not feasible in every run"
        elif item["final"]["feasible_count"] < total:
            excluded[item["variant_id"]] = "final policy not feasible in every run"
        elif item["late_feasible_rate"] < MIN_LATE_FEASIBLE_RATE:
            excluded[item["variant_id"]] = "late-stage feasible share below gate"
        else:
            candidates.append(item)
    if not candidates:
        return {
            "recommended_step_time_s": None,
            "selection_status": "本轮不能确定合格步长",
            "excluded": excluded,
        }
    energy = {
        item["variant_id"]: item["final"]["metrics"]["energy_kwh"]
        for item in candidates
    }
    variances = [
        value["std"] ** 2 for value in energy.values() if np.isfinite(value["std"])
    ]
    margin = float(np.sqrt(np.mean(variances))) if variances else 0.0
    reference = min(value["mean"] for value in energy.values())
    equivalent = [
        item
        for item in candidates
        if energy[item["variant_id"]]["mean"] - reference <= margin
    ]
    for item in candidates:
        if item not in equivalent:
            excluded[item["variant_id"]] = "final-policy energy outside margin"
    chosen = min(
        equivalent,
        key=lambda item: (
            np.nan_to_num(item["final"]["metrics"]["energy_kwh"]["std"]),
            item["energy_drift_kwh"]["mean"],
            item["step_time_s"],
        ),
    )
    return {
        "recommended_step_time_s": chosen["step_time_s"],
        "selection_status": f"推荐步长: {chosen['step_time_s']:g} s",
        "reference_energy_kwh": reference,
        "energy_margin_kwh": margin,
        "equivalent": [item["variant_id"] for item in equivalent],
        "excluded": excluded,
    }


def _format_cells(stats: dict[str, object], total: int) -> list[str]:
    cells = [f"{stats['feasible_rate'] * 100:.1f}% ({stats['feasible_count']}/{total})"]
    for key in METRIC_KEYS:
        value = stats["metrics"][key]
        if value is None:
            cells.append("—")
        elif key == "time_error_s":
            cells.append(f"{value['mean']:+.2f}±{value['std']:.2f}")
        else:
            cells.append(f"{value['mean']:.3f}±{value['std']:.3f}")
    return cells


def summarize(spec: ExperimentSpec, run_dirs: tuple[Path, ...]) -> dict[str, object]:
    if len(run_dirs) != len(spec.variants) * len(spec.seeds):
        raise ValueError("Step-time matrix is incomplete")
    variants = {}
    rows = [
        "| Control period | Policy | Strict feasibility rate | Stop error (m) "
        "| Δt (s) | Total energy (kWh) "
        "| Cumulative acceleration variation (m/s²) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rollout_steps = spec.train["num_envs"] * spec.train["n_steps_per_env"]
    late_start_step = (
        LATE_STAGE_START_FRACTION * spec.train["training_rollouts"] * rollout_steps
    )
    for index, variant in enumerate(spec.variants):
        paths = run_dirs[index * len(spec.seeds) : (index + 1) * len(spec.seeds)]
        completed = [read_completed_run(path) for path in paths]
        per_seed = {source: [] for source in POLICY_SOURCES}
        late_rates = []
        for seed, item in zip(spec.seeds, completed, strict=True):
            payloads = {
                "best": item.payload.best or item.payload,
                "final": item.payload,
            }
            for source, payload in payloads.items():
                per_seed[source].append({"seed": seed, **_policy_metrics(payload)})
            history = item.payload.evaluations
            late = history.feasible[history.training_steps > late_start_step]
            late_rates.append(float(np.mean(late)) if late.size else 0.0)
        stats = {
            source: _source_statistics(per_seed[source]) for source in POLICY_SOURCES
        }
        drift = np.asarray(
            [
                final["energy_kwh"] - best["energy_kwh"]
                for best, final in zip(per_seed["best"], per_seed["final"], strict=True)
            ]
        )
        drift_mean, drift_std, _ = aggregate_matrix(drift.reshape(-1, 1))
        variants[variant.id] = {
            "variant_id": variant.id,
            "label": variant.label,
            "step_time_s": float(completed[0].record.config["step_time_s"]),
            "late_feasible_rate": float(np.mean(late_rates)),
            "late_feasible_rate_per_seed": late_rates,
            "energy_drift_kwh": {
                "mean": float(drift_mean[0]),
                "std": float(drift_std[0]),
            },
            **{
                source: {**stats[source], "per_seed": per_seed[source]}
                for source in POLICY_SOURCES
            },
        }
        for source in POLICY_SOURCES:
            period = variant.label if source == POLICY_SOURCES[0] else ""
            rows.append(
                f"| {period} | {source.title()} | "
                + " | ".join(_format_cells(stats[source], len(spec.seeds)))
                + " |"
            )
    selection = select_step_time(list(variants.values()))
    rows.append(
        "\n*Note: Best is the checkpoint kept by periodic evaluation; final is the "
        "policy at the end of training; both are evaluated deterministically. "
        "Values are mean ± sample standard deviation across independent training "
        "runs. The strict feasibility rate counts all runs; the other metrics count "
        "only policies that reached the target (— if none did). All control periods "
        "share the per-step discount factor, the safety-reserve horizon in control "
        "periods and the budget in environment steps. Δt is the actual "
        "minus the planned running time (positive = late). Cumulative acceleration "
        r"variation is $\sum_t |a_t-a_{t-1}|$ (m/s²).*"
    )
    return {
        "matrix_id": "step_time",
        **selection,
        "variants": variants,
        "table": "\n".join(rows) + "\n",
    }


def write_summary(summary: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "step_time_summary.json").write_text(
        json.dumps(
            {key: value for key, value in summary.items() if key != "table"},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "step_time_table.md").write_text(summary["table"], encoding="utf-8")
