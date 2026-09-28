"""Spatial control-step ablation from completed training artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtto.io.artifacts import read_completed_run
from paper.analysis import aggregate_matrix, align_exact
from paper.experiments.runner import RunResult, execute_matrix
from paper.experiments.spec import ExperimentSpec, load_experiment_spec


def run(spec_path: str | Path) -> tuple[RunResult, ...]:
    return execute_matrix(load_experiment_spec(spec_path))


def summarize(spec: ExperimentSpec, run_dirs: tuple[Path, ...]) -> dict[str, object]:
    if len(run_dirs) != len(spec.variants) * len(spec.seeds):
        raise ValueError("Step-distance matrix is incomplete")
    variants = {}
    curves = {}
    rows = [
        "| Step distance | Strict feasibility rate | Stop error (m) | Time error (s) "
        "| Total energy (kWh) | Cumulative acceleration variation (m/s²) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, variant in enumerate(spec.variants):
        paths = run_dirs[index * len(spec.seeds) : (index + 1) * len(spec.seeds)]
        completed = [read_completed_run(path) for path in paths]
        seed_data = []
        for seed, item in zip(spec.seeds, completed, strict=True):
            payload = item.payload.best or item.payload
            quality = payload.quality
            metrics = quality.metrics
            task = item.record.task
            position = float(payload.profile.position_m[-1])
            ratio = min(
                1.0,
                max(
                    0.0,
                    (position - task["start_position_m"])
                    / (task["target_position_m"] - task["start_position_m"]),
                ),
            )
            seed_data.append(
                {
                    "seed": seed,
                    "feasible": quality.feasible,
                    "safe": quality.safe,
                    "success": quality.completed,
                    "completed_training_episodes": (
                        item.payload.result.training.actual_completed_episodes
                    ),
                    "route_completion_ratio": ratio,
                    "stop_error_m": abs(metrics.stop_error_m),
                    "abs_time_error_s": abs(metrics.arrival_time_error_s),
                    "energy_kwh": metrics.total_energy_kj / 3600.0,
                    "comfort_tav": metrics.comfort_tav_mps2,
                }
            )
        feasible = [item for item in seed_data if item["feasible"]]
        metric_stats = {}
        for key in ("stop_error_m", "abs_time_error_s", "energy_kwh", "comfort_tav"):
            values = np.asarray([item[key] for item in seed_data], dtype=float)
            mean, std, _ = aggregate_matrix(values[:, None])
            metric_stats[key] = {"mean": float(mean[0]), "std": float(std[0])}
        data = {
            "variant_id": variant.id,
            "label": variant.label,
            "step_distance": float(completed[0].record.config["step_distance_m"]),
            "feasible_count": len(feasible),
            "feasible_rate": len(feasible) / len(seed_data),
            "safe_count": sum(item["safe"] for item in seed_data),
            "success_count": sum(item["success"] for item in seed_data),
            "mean_route_completion_ratio": float(
                np.mean([item["route_completion_ratio"] for item in seed_data])
            ),
            "mean_feasible_energy_kwh": float(
                np.mean([item["energy_kwh"] for item in feasible])
            )
            if feasible
            else None,
            "mean_feasible_comfort": float(
                np.mean([item["comfort_tav"] for item in feasible])
            )
            if feasible
            else None,
            "metrics": metric_stats,
            "per_seed": seed_data,
        }
        variants[variant.id] = data
        rows.append(
            f"| {variant.label} | {data['feasible_rate'] * 100:.1f}% "
            f"({len(feasible)}/{len(seed_data)}) | "
            + " | ".join(
                f"{metric_stats[key]['mean']:.4f}±{metric_stats[key]['std']:.4f}"
                for key in (
                    "stop_error_m",
                    "abs_time_error_s",
                    "energy_kwh",
                    "comfort_tav",
                )
            )
            + " |"
        )
        histories = [item.payload.evaluations for item in completed]
        axis = np.unique(
            np.concatenate([history.training_steps for history in histories])
        ).astype(float)
        curve = {"training_steps": axis.tolist()}
        for key in ("route_completion_ratio", "feasible"):
            matrix = np.stack(
                [
                    align_exact(
                        axis,
                        history.training_steps.astype(float),
                        getattr(history, key).astype(float),
                    )
                    for history in histories
                ]
            )
            mean, std, counts = aggregate_matrix(matrix)
            curve[key] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "count": counts.tolist(),
            }
        curves[variant.id] = curve
    if all(item["feasible_count"] == 0 for item in variants.values()):
        recommended = None
        status = "本轮不能确定合格步长"
    else:
        ordered = sorted(
            variants.values(),
            key=lambda item: (
                -item["feasible_count"],
                -item["mean_route_completion_ratio"],
                item["mean_feasible_energy_kwh"]
                if item["mean_feasible_energy_kwh"] is not None
                else float("inf"),
                item["mean_feasible_comfort"]
                if item["mean_feasible_comfort"] is not None
                else float("inf"),
                item["step_distance"],
            ),
        )
        recommended = ordered[0]["step_distance"]
        status = f"推荐步长: {recommended:g} m"
    rows.append(
        "\n*Note: Early failures affect energy and errors; interpret them with strict "
        "feasibility. Cumulative acceleration variation is "
        r"$\sum_t |a_t-a_{t-1}|$ (m/s²).*"
    )
    return {
        "matrix_id": "step_distance",
        "recommended_step_distance": recommended,
        "selection_status": status,
        "variants": variants,
        "curves": curves,
        "table": "\n".join(rows) + "\n",
    }


def write_summary(summary: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "step_distance_summary.json").write_text(
        json.dumps(
            {key: value for key, value in summary.items() if key != "table"},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "step_distance_table.md").write_text(summary["table"], encoding="utf-8")
