from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def build_analysis_payload(
    *,
    run_name: str,
    run_directory: str,
    available_tags: list[str],
    regular_metrics: dict[str, Any],
    best_eval_metrics: dict[str, Any] | None = None,
    trajectory_evaluation_metrics: dict[str, Any] | None = None,
    reward_component_analysis: dict[str, Any] | None = None,
    curriculum_distribution_metrics: dict[str, Any] | None = None,
    safety_truncation_position_metrics: dict[str, Any] | None = None,
    step_snapshots: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "meta": {
            "generated_at_utc": datetime.now().isoformat(),
            "run_name": run_name,
            "run_directory": run_directory,
            "available_tags": sorted(available_tags),
            "config": config or {},
        },
        "regular_metrics": regular_metrics,
        "best_eval_metrics": best_eval_metrics or {},
        "trajectory_evaluation_metrics": trajectory_evaluation_metrics or {},
        "reward_component_analysis": reward_component_analysis or {},
        "curriculum_distribution_metrics": curriculum_distribution_metrics or {},
        "safety_truncation_position_metrics": safety_truncation_position_metrics or {},
        "data_quality": data_quality or {},
        "snapshots": {"by_step": step_snapshots or []},
    }


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, np.floating, np.integer))


def _flatten_numeric_fields(
    value: Any,
    *,
    prefix: str = "",
    out: dict[str, float] | None = None,
) -> dict[str, float]:
    if out is None:
        out = {}

    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _ = _flatten_numeric_fields(child, prefix=next_prefix, out=out)
        return out

    if isinstance(value, list):
        # 跳过嵌套列表/矩阵，CSV 无法扁平化表达
        return out

    if _is_scalar(value):
        out[prefix] = float(value)

    return out


def _snapshot_rows(
    snapshots: list[dict[str, Any]],
    *,
    key_fields: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    metric_columns: set[str] = set()

    for snapshot in snapshots:
        metrics = snapshot.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for tag, stats in metrics.items():
            if not isinstance(stats, dict):
                continue
            for stat_name in ("mean", "p05", "p95", "min", "max", "slope", "cv"):
                if stat_name in stats:
                    metric_columns.add(f"{tag}.{stat_name}")

    ordered_metric_columns = sorted(metric_columns)
    columns = key_fields + ordered_metric_columns

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        row: dict[str, Any] = {}
        for key in key_fields:
            row[key] = snapshot.get(key, "")

        metrics = snapshot.get("metrics", {})
        if isinstance(metrics, dict):
            for tag, stats in metrics.items():
                if not isinstance(stats, dict):
                    continue
                for stat_name in ("mean", "p05", "p95", "min", "max", "slope", "cv"):
                    column = f"{tag}.{stat_name}"
                    if column in ordered_metric_columns and stat_name in stats:
                        row[column] = stats[stat_name]

        rows.append(row)

    return columns, rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_number(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return f"{v:.6g}"
    except TypeError, ValueError:
        return default


def _format_percent(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100.0:.2f}%"
    except TypeError, ValueError:
        return default


BEST_EVAL_DISPLAY_METRICS = [
    "success",
    "precise_arrival",
    "punctual_arrival",
    "stop_error_m",
    "time_error_s",
    "total_reward",
    "total_energy_j",
    "comfort_tav",
    "comfort_er_pct",
    "comfort_rms",
]


def _best_eval_metric_final(best_eval: dict[str, Any], key: str) -> Any:
    entry = best_eval.get(key, {})
    if not isinstance(entry, dict):
        return None
    return entry.get("final")


def _ordered_best_eval_keys(best_eval: dict[str, Any], prefix: str) -> list[str]:
    return [
        f"{prefix}_{metric}"
        for metric in BEST_EVAL_DISPLAY_METRICS
        if isinstance(best_eval.get(f"{prefix}_{metric}"), dict)
    ]


def _ascii_bar(value: Any, width: int = 24) -> str:
    try:
        ratio = float(value)
    except TypeError, ValueError:
        ratio = 0.0
    ratio = max(0.0, min(1.0, ratio))
    w = max(8, int(width))
    filled = int(round(ratio * w))
    return f"[{('#' * filled) + ('.' * (w - filled))}] {_format_percent(ratio)}"


def _short_component_name(tag: str) -> str:
    if "/" in tag:
        return tag.split("/")[-1]
    return tag


def _generate_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    config = meta.get("config", {})
    regular = payload.get("regular_metrics", {})
    reward_analysis = payload.get("reward_component_analysis", {})
    trajectory_eval = payload.get("trajectory_evaluation_metrics", {})
    curriculum = payload.get("curriculum_distribution_metrics", {})
    safety_position = payload.get("safety_truncation_position_metrics", {})
    snapshots = payload.get("snapshots", {})

    convergence = regular.get("convergence_speed_quality", {})
    vitality = regular.get("policy_vitality", {})
    critic = regular.get("critic_foresight", {})
    update_safety = regular.get("update_safety", {})

    components = reward_analysis.get("components", {})
    episode_correlation = reward_analysis.get("episode_return_correlation", {})
    strong_negative_pairs = episode_correlation.get("strong_negative_pairs", [])

    bar_width = (
        int(config.get("report_bar_width", 24)) if isinstance(config, dict) else 24
    )

    top_activity = sorted(
        (
            (key, value.get("absolute_activity_share", 0.0))
            for key, value in components.items()
            if isinstance(value, dict)
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# Training Analysis Report")
    lines.append("")
    lines.append("## LLM Core Summary")
    lines.append("")
    lines.append(
        f"- final_ep_rew_mean: {_format_number(convergence.get('final_ep_rew_mean'))}"
    )

    best_eval = payload.get("best_eval_metrics", {})
    if isinstance(best_eval, dict) and best_eval.get("available", False):
        lines.append(
            "- best_eval: "
            + f"arrival_success_rate={
                _format_percent(_best_eval_metric_final(best_eval, 'best_success'))
            }, "
            + f"precise_arrival_rate={
                _format_percent(
                    _best_eval_metric_final(best_eval, 'best_precise_arrival')
                )
            }, "
            + f"punctual_arrival_rate={
                _format_percent(
                    _best_eval_metric_final(best_eval, 'best_punctual_arrival')
                )
            }, "
            + f"best_reward={
                _format_number(_best_eval_metric_final(best_eval, 'best_total_reward'))
            }"
        )

    top3 = top_activity[:3]
    if top3:
        lines.append(
            "- top_reward_activity: "
            + ", ".join(
                f"{_short_component_name(tag)}={_format_percent(val)}"
                for tag, val in top3
            )
        )

    if strong_negative_pairs:
        top_conflict = strong_negative_pairs[0]
        lines.append(
            "- top_objective_conflict: "
            + f"{top_conflict.get('left', '')} vs {top_conflict.get('right', '')}"
            + f" (pearson={_format_number(top_conflict.get('pearson'))})"
        )

    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- run_name: {meta.get('run_name', 'unknown')}")
    lines.append(f"- generated_at_utc: {meta.get('generated_at_utc', 'unknown')}")
    lines.append(f"- run_directory: {meta.get('run_directory', 'unknown')}")
    lines.append(f"- tags_count: {len(meta.get('available_tags', []))}")
    if reward_analysis.get("available", False):
        lines.append(
            "- reward_diagnostics: "
            + f"transitions={reward_analysis.get('transition_count', 0)}, "
            + f"complete_episodes={reward_analysis.get('complete_episode_count', 0)}, "
            + f"partial_episodes={reward_analysis.get('partial_episode_count', 0)}"
        )

    lines.append("")
    lines.append("## Core Training Performance")
    lines.append("")
    lines.append(
        "- convergence: "
        + f"final_ep_rew_mean={_format_number(convergence.get('final_ep_rew_mean'))}, "
        + f"rise_slope_per_step={
            _format_number(convergence.get('rise_slope_per_step'))
        }, "
        + f"volatility_cv={_format_number(convergence.get('volatility_cv'))}"
    )
    lines.append(
        "- policy_vitality: "
        + f"entropy_trend_slope={
            _format_number(vitality.get('entropy_trend_slope_per_step'))
        }, "
        + f"rigidity_risk_score={_format_number(vitality.get('rigidity_risk_score'))}"
    )
    lines.append(
        "- critic_foresight: "
        + f"explained_variance_mean={
            _format_number(critic.get('explained_variance_mean'))
        }, "
        + f"low_explained_variance_ratio={
            _format_percent(critic.get('low_explained_variance_ratio'))
        }"
    )
    lines.append(
        "- update_safety: "
        + f"approx_kl_p95={_format_number(update_safety.get('approx_kl_p95'))}, "
        + "approx_kl_exceed_ratio="
        + f"{_format_percent(update_safety.get('approx_kl_exceed_ratio'))}"
    )

    best_eval = payload.get("best_eval_metrics", {})
    if isinstance(best_eval, dict) and best_eval.get("available", False):
        lines.append("")
        lines.append("## Best Evaluation Performance")
        lines.append("")
        best_keys = _ordered_best_eval_keys(best_eval, "best")
        last_keys = _ordered_best_eval_keys(best_eval, "last")
        if best_keys:
            lines.append("- final_best_values:")
            for key in best_keys:
                entry = (
                    best_eval.get(key, {})
                    if isinstance(best_eval.get(key), dict)
                    else {}
                )
                lines.append(f"  - {key}: {_format_number(entry.get('final'))}")
        if last_keys:
            lines.append("- last_eval_values:")
            for key in last_keys:
                entry = (
                    best_eval.get(key, {})
                    if isinstance(best_eval.get(key), dict)
                    else {}
                )
                lines.append(f"  - {key}: {_format_number(entry.get('final'))}")

    lines.append("")
    lines.append("## Reward Component Diagnostics")
    lines.append("")
    if top_activity:
        lines.append("- absolute_activity_share:")
        for name, value in top_activity:
            metrics = components.get(name, {})
            lines.append(
                f"  - {name}: {_ascii_bar(value, width=bar_width)}, "
                + "signed_return_ratio="
                + _format_number(metrics.get("signed_return_ratio"))
                + ", nonzero_frequency="
                + _format_percent(metrics.get("nonzero_frequency"))
                + ", active_mean_absolute_strength="
                + _format_number(metrics.get("active_mean_absolute_strength"))
            )
    else:
        lines.append(
            "- unavailable: "
            + str(reward_analysis.get("reason", "reward artifact unavailable"))
        )

    if strong_negative_pairs:
        lines.append("- objective_conflicts(top):")
        for pair in strong_negative_pairs[:5]:
            lines.append(
                "  - "
                + f"{pair.get('left', '')} vs {pair.get('right', '')}: "
                + f"pearson={_format_number(pair.get('pearson'))}"
            )
    else:
        lines.append("- objective_conflicts: no strong negative pairs detected")

    lines.append("")
    lines.append("## Evaluation Trend")
    lines.append("")
    if isinstance(trajectory_eval, dict) and trajectory_eval.get("available", False):
        for name, entry in trajectory_eval.get("metrics", {}).items():
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"- {name}: final={_format_number(entry.get('final'))}, "
                + "trend_slope_per_step="
                + f"{_format_number(entry.get('trend_slope_per_step'))}"
            )
    else:
        lines.append("- unavailable")

    lines.append("")
    lines.append("## DSPDL Distribution")
    lines.append("")
    if isinstance(curriculum, dict) and curriculum.get("available", False):
        empirical = curriculum.get("empirical_to_target_kl", {})
        lines.append(
            "- empirical_to_target_kl: "
            + f"final={_format_number(empirical.get('final'))}, "
            + "trend_slope_per_step="
            + f"{_format_number(empirical.get('trend_slope_per_step'))}"
        )
        diagnostics = curriculum.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            for key in (
                "converged",
                "alpha",
                "update_kl",
                "critic_values_duration_s",
                "distribution_solve_duration_s",
                "critic_return_mae",
                "critic_return_pearson",
            ):
                entry = diagnostics.get(key, {})
                if isinstance(entry, dict):
                    lines.append(
                        f"- {key}: final={_format_number(entry.get('final'))}, "
                        + "trend_slope_per_step="
                        + f"{_format_number(entry.get('trend_slope_per_step'))}"
                    )
    else:
        lines.append("- unavailable: no DSPDL distribution KL logged")

    lines.append("")
    lines.append("## Safety Truncation Positions")
    lines.append("")
    highest_truncation = (
        safety_position.get("highest_safety_truncation_bin")
        if isinstance(safety_position, dict)
        else None
    )
    if isinstance(safety_position, dict) and safety_position.get("available"):
        lines.append(
            "- total_safety_truncation_count: "
            + _format_number(safety_position.get("total_safety_truncation_count"))
        )
        if isinstance(highest_truncation, dict):
            lines.append(
                "- highest_safety_truncation_bin: "
                + f"[{_format_number(highest_truncation.get('bin_start_m'))}, "
                + f"{_format_number(highest_truncation.get('bin_end_m'))}) m, "
                + "count="
                + _format_number(highest_truncation.get("safety_truncation_count"))
                + ", global_share="
                + _format_percent(
                    highest_truncation.get("global_safety_truncation_share")
                )
            )
        bins = safety_position.get("bins", [])
        if isinstance(bins, list) and bins:
            lines.append("- bins:")
            for entry in bins:
                if not isinstance(entry, dict):
                    continue
                lines.append(
                    "  - "
                    + f"[{_format_number(entry.get('bin_start_m'))}, "
                    + f"{_format_number(entry.get('bin_end_m'))}) m: "
                    + "count="
                    + _format_number(entry.get("safety_truncation_count"))
                    + ", low_count="
                    + _format_number(entry.get("low_safety_truncation_count"))
                    + ", high_count="
                    + _format_number(entry.get("high_safety_truncation_count"))
                    + ", global_share="
                    + _format_percent(entry.get("global_safety_truncation_share"))
                )
        else:
            lines.append("- no safety truncations observed")
    else:
        reason = (
            safety_position.get("reason") if isinstance(safety_position, dict) else None
        )
        lines.append(f"- unavailable: {reason}" if reason else "- unavailable")

    lines.append("")
    lines.append("## Artifact Summary")
    lines.append("")
    lines.append(f"- step_snapshots_count: {len(snapshots.get('by_step', []))}")
    lines.append(
        f"- export_csv: {
            bool(config.get('export_csv', False)) if isinstance(config, dict) else False
        }"
    )

    return "\n".join(lines) + "\n"


def write_analysis_outputs(
    payload: dict[str, Any],
    output_root: str | Path,
    run_name: str,
) -> dict[str, str]:
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analysis_snapshot.json"
    markdown_path = output_dir / "report.md"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)

    markdown_report = _generate_markdown_report(payload)
    _ = markdown_path.write_text(markdown_report, encoding="utf-8")

    output_paths: dict[str, str] = {
        "output_dir": str(output_dir),
        "json_snapshot": str(json_path),
        "markdown_report": str(markdown_path),
    }

    meta = payload.get("meta", {})
    config = meta.get("config", {}) if isinstance(meta, dict) else {}
    export_csv = (
        bool(config.get("export_csv", False)) if isinstance(config, dict) else False
    )
    include_snapshots = (
        bool(config.get("include_snapshots", False))
        if isinstance(config, dict)
        else False
    )

    if not export_csv:
        return output_paths

    summary_csv_path = output_dir / "summary_metrics.csv"
    summary_metrics = {}
    summary_metrics.update(
        _flatten_numeric_fields(payload.get("regular_metrics", {}), prefix="regular")
    )
    summary_metrics.update(
        _flatten_numeric_fields(
            payload.get("trajectory_evaluation_metrics", {}),
            prefix="trajectory_evaluation",
        )
    )
    summary_metrics.update(
        _flatten_numeric_fields(
            payload.get("curriculum_distribution_metrics", {}),
            prefix="curriculum_distribution",
        )
    )
    summary_metrics.update(
        _flatten_numeric_fields(
            payload.get("safety_truncation_position_metrics", {}),
            prefix="safety_truncation_position",
        )
    )
    summary_metrics.update(
        _flatten_numeric_fields(
            payload.get("reward_component_analysis", {}),
            prefix="reward_component",
        )
    )
    _write_csv(
        summary_csv_path,
        columns=sorted(summary_metrics.keys()),
        rows=[summary_metrics],
    )
    output_paths["summary_metrics_csv"] = str(summary_csv_path)

    if include_snapshots:
        step_snapshots = payload.get("snapshots", {}).get("by_step", [])
        if step_snapshots:
            step_csv_path = output_dir / "step_snapshots.csv"
            step_columns, step_rows = _snapshot_rows(
                step_snapshots,
                key_fields=[
                    "window_type",
                    "window_index",
                    "step_start",
                    "step_end",
                    "sample_count",
                    "severity",
                ],
            )
            _write_csv(step_csv_path, columns=step_columns, rows=step_rows)
            output_paths["step_snapshots_csv"] = str(step_csv_path)

    return output_paths
