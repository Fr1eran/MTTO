"""Method ablation execution and artifact-based tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtto.evaluation.quality import selection_key
from mtto.io.artifacts import read_completed_run
from mtto.rl.state import TerminationReason
from paper.analysis import aggregate_matrix
from paper.experiments.runner import RunResult, execute_matrix
from paper.experiments.spec import ExperimentSpec, load_experiment_spec

# Late-stage window for the feasible-evaluation ratio: the last quarter of the
# training budget, matching the last period of the training-process table.
LATE_STAGE_START_FRACTION = 0.75
POLICY_SOURCES = ("best", "final")
TRAJECTORY_METRICS = ("stop_error_m", "time_error_s", "total_energy_kwh", "comfort_tav")
PERFORMANCE_NOTE = (
    "\n\n*Note: Best is the checkpoint kept by periodic evaluation; final is the "
    "policy at the end of training; both are evaluated deterministically. Values "
    "are mean ± sample standard deviation across independent training runs. The "
    "strict feasibility rate counts all runs; stop error, Δt, energy and cumulative "
    "acceleration variation count only policies that reached the target (— if none "
    "did). Late-stage feasible evaluations and first feasible evaluation describe "
    "the training process and are given in the best row only: the former is the "
    "share of periodic deterministic evaluations in the last quarter of training "
    "that are strictly feasible, the latter the share of the training budget used "
    "when the first strictly feasible evaluation occurred, over runs that had one. "
    "Δt is the actual minus the planned running time (positive = late).*"
)


def _mean_std(series: list[float]) -> dict[str, float] | None:
    if not series:
        return None
    mean, std, _ = aggregate_matrix(np.asarray(series, dtype=float).reshape(-1, 1))
    return {"mean": float(mean[0]), "std": float(std[0])}


def _cell(value: dict[str, float] | None, scale: float, digits: int, sign: str) -> str:
    if value is None:
        return "—"
    return (
        f"{scale * value['mean']:{sign}.{digits}f} ± {scale * value['std']:.{digits}f}"
    )


def representative_index(
    metrics: list[dict[str, object]], seeds: tuple[int, ...]
) -> int:
    """Strictly feasible with the lowest energy first, then the selection fallback.

    Callers pass final-policy metrics: the best checkpoint is the lowest-energy
    feasible evaluation and so leans towards arriving late within the tolerance.
    """
    return max(
        range(len(seeds)),
        key=lambda item: (metrics[item]["selection_comparison_key"], -seeds[item]),
    )


def run(spec_path: str | Path, workers: int = 1) -> tuple[RunResult, ...]:
    """Execute the full matrix described by a TOML file."""
    return execute_matrix(load_experiment_spec(spec_path), workers)


def summarize(spec: ExperimentSpec, run_dirs: tuple[Path, ...]) -> dict[str, object]:
    """Summarize best- and final-policy quality and training diagnostics."""
    runs = [read_completed_run(path) for path in run_dirs]
    expected = len(spec.variants) * len(spec.seeds)
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} run directories, got {len(runs)}")
    raw: dict[str, list[dict[str, object]]] = {}
    raw_final: dict[str, list[dict[str, object]]] = {}
    performance: dict[str, dict[str, object]] = {}
    training: dict[str, dict[str, object]] = {}
    feasible: dict[str, dict[str, object]] = {}
    representatives: dict[str, dict[str, object]] = {}
    training_curves: dict[str, dict[str, object]] = {}
    performance_rows = [
        "# Method Ablation Performance Table",
        "",
        "| Method | Policy | Strict feasibility rate | "
        "Late-stage feasible evaluations | "
        "First feasible evaluation (% of budget) | Stop error (m) | Δt (s) | "
        "Total energy (kWh) | Cumulative acceleration variation (m/s²) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    training_rows = [
        "# Method Ablation Training Process Table",
        "",
        "| Method | Rollouts | Low Viol. Rate (/10⁴) | High Viol. Rate (/10⁴) | "
        "Total Viol. Rate (/10⁴) | Arrival Ratio | Low Violations | "
        "High Violations | Total Violations | Transitions | Arrivals / Completed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rollout_steps = spec.train["num_envs"] * spec.train["n_steps_per_env"]
    total_steps = spec.train["training_rollouts"] * rollout_steps
    late_start_step = LATE_STAGE_START_FRACTION * total_steps
    periods = tuple(
        (start, min(start + 99, spec.train["training_rollouts"]))
        for start in range(1, spec.train["training_rollouts"] + 1, 100)
    )
    for index, method in enumerate(spec.variants):
        group = runs[index * len(spec.seeds) : (index + 1) * len(spec.seeds)]
        by_source: dict[str, list[dict[str, object]]] = {
            source: [] for source in POLICY_SOURCES
        }
        for seed, completed in zip(spec.seeds, group, strict=True):
            payloads = {
                "best": completed.payload.best or completed.payload,
                "final": completed.payload,
            }
            for source, payload in payloads.items():
                quality = payload.quality
                by_source[source].append(
                    {
                        "run_id": completed.record.run_id,
                        "seed": seed,
                        "stop_error_m": abs(quality.metrics.stop_error_m),
                        "time_error_s": quality.metrics.arrival_time_error_s,
                        "abs_time_error_s": abs(quality.metrics.arrival_time_error_s),
                        "total_energy_kwh": quality.metrics.total_energy_kj / 3600.0,
                        "comfort_tav": quality.metrics.comfort_tav_mps2,
                        "success": quality.completed,
                        "safe": quality.safe,
                        "feasible": quality.feasible,
                        "selection_comparison_key": list(selection_key(quality)),
                    }
                )
        metrics = by_source["best"]
        raw[method.id] = metrics
        raw_final[method.id] = by_source["final"]
        count = sum(item["feasible"] for item in metrics)
        feasible[method.id] = {
            "feasible_count": count,
            "total_runs": len(metrics),
            "feasible_rate": count / len(metrics),
        }
        late_rates = []
        first_feasible = []
        for completed in group:
            history = completed.payload.evaluations
            late = history.feasible[history.training_steps > late_start_step]
            if late.size:
                late_rates.append(float(np.mean(late)))
            feasible_steps = history.training_steps[history.feasible]
            if feasible_steps.size:
                first_feasible.append(float(feasible_steps.min()) / total_steps)
        values: dict[str, object] = {
            "late_feasible_rate": _mean_std(late_rates),
            "first_feasible_progress": _mean_std(first_feasible),
        }
        for source, items in by_source.items():
            arrived = [item for item in items if item["success"]]
            source_count = sum(item["feasible"] for item in items)
            values[source] = {
                "feasible_count": source_count,
                **{
                    key: _mean_std([item[key] for item in arrived])
                    for key in TRAJECTORY_METRICS
                },
            }
            process = (
                [
                    _cell(values["late_feasible_rate"], 100.0, 1, ""),
                    _cell(values["first_feasible_progress"], 100.0, 1, ""),
                ]
                if source == "best"
                else ["", ""]
            )
            cells = [
                method.label if source == "best" else "",
                source.capitalize(),
                f"{100 * source_count / len(items):.1f}% ({source_count}/{len(items)})",
                *process,
                *(
                    _cell(values[source][key], 1.0, digits, sign)
                    for key, digits, sign in (
                        ("stop_error_m", 3, ""),
                        ("time_error_s", 2, "+"),
                        ("total_energy_kwh", 1, ""),
                        ("comfort_tav", 3, ""),
                    )
                ),
            ]
            performance_rows.append("| " + " | ".join(cells) + " |")
        performance[method.id] = values
        index_in_group = representative_index(by_source["final"], spec.seeds)
        run_dir = run_dirs[index * len(spec.seeds) + index_in_group]
        representatives[method.id] = {
            "variant_id": method.id,
            **by_source["final"][index_in_group],
            "run_dir": str(run_dir),
            "model_path": str(run_dir / "policy.zip"),
        }
        period_data = {}
        for start, end in periods:
            seed_counts = []
            for completed in group:
                reward = completed.payload.diagnostics.reward
                mask = (
                    reward.episode_complete
                    & (reward.episode_end_step > (start - 1) * rollout_steps)
                    & (reward.episode_end_step <= end * rollout_steps)
                )
                reasons = reward.episode_termination_reason[mask]
                low = int(
                    np.count_nonzero(
                        reasons == int(TerminationReason.UNDER_LOWER_LIMIT)
                    )
                )
                high = int(
                    np.count_nonzero(
                        np.isin(
                            reasons,
                            [
                                int(TerminationReason.OVER_UPPER_LIMIT),
                                int(TerminationReason.OVER_SRTSP),
                            ],
                        )
                    )
                )
                arrived = int(
                    np.count_nonzero(reasons == int(TerminationReason.STOPPED_IN_ZONE))
                )
                seed_counts.append((low, high, arrived, len(reasons)))
            array = np.asarray(seed_counts, dtype=float)
            transitions_per_seed = (end - start + 1) * rollout_steps
            data = {}
            for key, series in zip(
                ("low", "high", "total"),
                (array[:, 0], array[:, 1], array[:, 0] + array[:, 1]),
                strict=True,
            ):
                rates = series / transitions_per_seed * 10000.0
                data[f"count_{key}"] = int(np.sum(series))
                data[f"rate_{key}_mean"] = float(np.mean(rates))
                data[f"rate_{key}_std"] = (
                    float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
                )
            arrival_rates = np.divide(
                array[:, 2],
                array[:, 3],
                out=np.zeros(len(array)),
                where=array[:, 3] > 0,
            )
            data.update(
                {
                    "arrival_ratio_mean": float(np.mean(arrival_rates)),
                    "arrival_ratio_std": float(np.std(arrival_rates, ddof=1))
                    if len(array) > 1
                    else 0.0,
                    "count_arrived": int(np.sum(array[:, 2])),
                    "count_completed": int(np.sum(array[:, 3])),
                    "transitions": transitions_per_seed * len(group),
                }
            )
            data["overall_arrival_pct"] = (
                100 * data["count_arrived"] / data["count_completed"]
                if data["count_completed"]
                else 0.0
            )
            key = f"{start}-{end}"
            period_data[key] = data
            training_rows.append(
                "| "
                + " | ".join(
                    [
                        method.label,
                        key,
                        *(
                            f"{data[f'rate_{name}_mean']:.2f} ± "
                            f"{data[f'rate_{name}_std']:.2f}"
                            for name in ("low", "high", "total")
                        ),
                        f"{data['arrival_ratio_mean'] * 100:.2f}% ± "
                        f"{data['arrival_ratio_std'] * 100:.2f}%",
                        *(
                            str(data[f"count_{name}"])
                            for name in ("low", "high", "total")
                        ),
                        f"{data['transitions']:,}",
                        f"{data['count_arrived']} / {data['count_completed']} "
                        f"({data['overall_arrival_pct']:.2f}%)",
                    ]
                )
                + " |"
            )
        training[method.id] = {"periods": period_data}
        interval = spec.train["evaluation_interval_rollouts"]
        rollout_points = (
            np.arange(interval, spec.train["training_rollouts"], interval, dtype=int)
            if interval is not None
            else np.empty(0, dtype=int)
        )
        rate_matrix = []
        arrival_matrix = []
        for completed in group:
            reward = completed.payload.diagnostics.reward
            rates = []
            arrivals = []
            for end in rollout_points:
                mask = (
                    reward.episode_complete
                    & (reward.episode_end_step > (end - interval) * rollout_steps)
                    & (reward.episode_end_step <= end * rollout_steps)
                )
                reasons = reward.episode_termination_reason[mask]
                violations = np.count_nonzero(
                    np.isin(
                        reasons,
                        [
                            int(TerminationReason.UNDER_LOWER_LIMIT),
                            int(TerminationReason.OVER_UPPER_LIMIT),
                            int(TerminationReason.OVER_SRTSP),
                        ],
                    )
                )
                rates.append(float(violations) / (interval * rollout_steps) * 10000.0)
                arrived = np.count_nonzero(
                    reasons == int(TerminationReason.STOPPED_IN_ZONE)
                )
                arrivals.append(float(arrived) / len(reasons) if len(reasons) else 0.0)
            rate_matrix.append(rates)
            arrival_matrix.append(arrivals)
        rate_mean, rate_std, _ = aggregate_matrix(np.asarray(rate_matrix, dtype=float))
        arrival_mean, arrival_std, _ = aggregate_matrix(
            np.asarray(arrival_matrix, dtype=float)
        )
        training_curves[method.id] = {
            "training_steps": (rollout_points * rollout_steps).tolist(),
            "speed_violation_rate": {
                "mean": rate_mean.tolist(),
                "std": rate_std.tolist(),
            },
            "arrival_ratio": {
                "mean": arrival_mean.tolist(),
                "std": arrival_std.tolist(),
            },
        }
    paired = {}
    if "ppo" in raw:
        for method in spec.variants:
            if method.id == "ppo":
                continue
            differences = {}
            by_seed = {}
            for seed, current, baseline in zip(
                spec.seeds, raw[method.id], raw["ppo"], strict=True
            ):
                by_seed[str(seed)] = {
                    metric: current[metric] - baseline[metric]
                    for metric in (
                        "stop_error_m",
                        "abs_time_error_s",
                        "total_energy_kwh",
                        "comfort_tav",
                    )
                }
            for metric in (
                "stop_error_m",
                "abs_time_error_s",
                "total_energy_kwh",
                "comfort_tav",
            ):
                series = [
                    current[metric] - baseline[metric]
                    for current, baseline in zip(
                        raw[method.id], raw["ppo"], strict=True
                    )
                ]
                differences[metric] = {
                    "mean": float(np.mean(series)),
                    "std": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
                }
            paired[f"{method.id}_minus_ppo"] = {
                "by_seed": by_seed,
                "mean": {metric: data["mean"] for metric, data in differences.items()},
                "std": {metric: data["std"] for metric, data in differences.items()},
            }
    return {
        "budget_mode": spec.train["budget_mode"],
        "training_rollouts": spec.train["training_rollouts"],
        "training_steps": spec.train["training_rollouts"] * rollout_steps,
        "seeds": list(spec.seeds),
        "methods": [method.id for method in spec.variants],
        "method_labels": {method.id: method.label for method in spec.variants},
        "feasible_summary": feasible,
        "raw_seed_metrics": raw,
        "raw_seed_metrics_final": raw_final,
        "paired_differences": paired,
        "representative_policies": representatives,
        "representative_policy": representatives.get("ppo_pirs"),
        "performance": performance,
        "training_curves": training_curves,
        "training": training,
        "training_table": "\n".join(training_rows) + "\n",
        "performance_table": "\n".join(performance_rows) + PERFORMANCE_NOTE + "\n",
    }


def write_summary(summary: dict[str, object], output: Path) -> None:
    """Write the derived data and both tables."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if not key.endswith("_table")
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "method_training_table.md").write_text(
        summary["training_table"], encoding="utf-8"
    )
    (output / "method_performance_table.md").write_text(
        summary["performance_table"], encoding="utf-8"
    )
