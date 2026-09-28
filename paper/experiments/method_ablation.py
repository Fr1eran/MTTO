"""Method ablation execution and artifact-based tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtto.evaluation.quality import selection_key
from mtto.io.artifacts import read_completed_run
from mtto.rl.state import TerminationReason
from paper.analysis import aggregate_matrix, align_exact
from paper.experiments.runner import RunResult, execute_matrix
from paper.experiments.spec import ExperimentSpec, load_experiment_spec


def run(spec_path: str | Path) -> tuple[RunResult, ...]:
    """Execute the full matrix described by a TOML file."""
    return execute_matrix(load_experiment_spec(spec_path))


def summarize(spec: ExperimentSpec, run_dirs: tuple[Path, ...]) -> dict[str, object]:
    """Summarize best-policy quality and training diagnostics across seeds."""
    runs = [read_completed_run(path) for path in run_dirs]
    expected = len(spec.variants) * len(spec.seeds)
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} run directories, got {len(runs)}")
    raw: dict[str, list[dict[str, object]]] = {}
    performance: dict[str, dict[str, object]] = {}
    training: dict[str, dict[str, object]] = {}
    feasible: dict[str, dict[str, object]] = {}
    curves: dict[str, dict[str, object]] = {}
    training_curves: dict[str, dict[str, object]] = {}
    performance_rows = [
        "# Method Ablation Performance Table",
        "",
        "| Method | Strict feasibility rate | Stop error (m) | Time error (s) | "
        "Total energy (kWh) | Cumulative acceleration variation (m/s²) |",
        "| --- | --- | --- | --- | --- | --- |",
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
    periods = tuple(
        (start, min(start + 99, spec.train["training_rollouts"]))
        for start in range(1, spec.train["training_rollouts"] + 1, 100)
    )
    for index, method in enumerate(spec.variants):
        group = runs[index * len(spec.seeds) : (index + 1) * len(spec.seeds)]
        metrics = []
        for seed, completed in zip(spec.seeds, group, strict=True):
            quality = (completed.payload.best or completed.payload).quality
            item = {
                "run_id": completed.record.run_id,
                "seed": seed,
                "stop_error_m": abs(quality.metrics.stop_error_m),
                "time_error_s": abs(quality.metrics.arrival_time_error_s),
                "total_energy_kwh": quality.metrics.total_energy_kj / 3600.0,
                "comfort_tav": quality.metrics.comfort_tav_mps2,
                "success": quality.completed,
                "safe": quality.safe,
                "feasible": quality.feasible,
                "selection_comparison_key": list(selection_key(quality)),
            }
            metrics.append(item)
        raw[method.id] = metrics
        count = sum(item["feasible"] for item in metrics)
        feasible[method.id] = {
            "feasible_count": count,
            "total_runs": len(metrics),
            "feasible_rate": count / len(metrics),
        }
        values = {}
        for key in ("stop_error_m", "time_error_s", "total_energy_kwh", "comfort_tav"):
            series = np.asarray([item[key] for item in metrics], dtype=float)
            mean, std, _ = aggregate_matrix(series[:, None])
            values[key] = {
                "mean": float(mean[0]),
                "std": float(std[0]),
            }
        performance[method.id] = values
        histories = [completed.payload.evaluations for completed in group]
        axis = np.unique(
            np.concatenate([history.training_steps for history in histories])
        ).astype(float)
        curve_data = {"training_steps": axis.tolist()}
        for key, transform in (
            ("stop_error_m", np.abs),
            ("time_error_s", np.abs),
            ("total_energy_j", lambda value: value / 3_600_000.0),
            ("comfort_tav", lambda value: value),
        ):
            matrix = np.stack(
                [
                    align_exact(
                        axis,
                        history.training_steps.astype(float),
                        transform(getattr(history, key)),
                    )
                    for history in histories
                ]
            )
            mean, std, count_by_step = aggregate_matrix(matrix)
            curve_data[key] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "count": count_by_step.tolist(),
            }
        curves[method.id] = curve_data
        performance_rows.append(
            "| "
            + " | ".join(
                [
                    method.label,
                    f"{100 * count / len(metrics):.1f}% ({count}/{len(metrics)})",
                    *(
                        f"{values[key]['mean']:.6f} ± {values[key]['std']:.6f}"
                        for key in values
                    ),
                ]
            )
            + " |"
        )
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
                        "time_error_s",
                        "total_energy_kwh",
                        "comfort_tav",
                    )
                }
            for metric in (
                "stop_error_m",
                "time_error_s",
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
    representative = None
    if "ppo_pirs" in raw:
        index = max(
            range(len(spec.seeds)),
            key=lambda item: (
                raw["ppo_pirs"][item]["selection_comparison_key"],
                -spec.seeds[item],
            ),
        )
        item = raw["ppo_pirs"][index]
        run_dir = run_dirs[
            next(i for i, method in enumerate(spec.variants) if method.id == "ppo_pirs")
            * len(spec.seeds)
            + index
        ]
        representative = {
            "variant_id": "ppo_pirs",
            **item,
            "model_path": str(run_dir / "best" / "policy.zip")
            if (run_dir / "best" / "policy.zip").exists()
            else str(run_dir / "policy.zip"),
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
        "paired_differences": paired,
        "representative_policy": representative,
        "performance": performance,
        "evaluation_curves": curves,
        "training_curves": training_curves,
        "training": training,
        "training_table": "\n".join(training_rows) + "\n",
        "performance_table": "\n".join(performance_rows) + "\n",
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
