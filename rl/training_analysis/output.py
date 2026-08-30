from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

BEST_EVAL_DISPLAY_METRICS = (
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
)

STAT_NAMES = ("mean", "p05", "p95", "min", "max", "slope", "cv")

DSPDL_DIAG_KEYS = (
    "converged",
    "alpha",
    "update_kl",
    "critic_values_duration_s",
    "distribution_solve_duration_s",
    "critic_return_mae",
    "critic_return_pearson",
)


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


def _flatten_numeric_fields(
    value: Any,
    *,
    prefix: str = "",
    out: dict[str, float] | None = None,
) -> dict[str, float]:
    target = {} if out is None else out
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_numeric_fields(
                child, prefix=f"{prefix}.{key}" if prefix else str(key), out=target
            )
    elif isinstance(value, (int, float, bool, np.floating, np.integer)):
        target[prefix] = float(value)
    return target


def _snapshot_rows(
    snapshots: list[dict[str, Any]],
    *,
    key_fields: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    metric_columns = sorted(
        {
            f"{tag}.{stat}"
            for s in snapshots
            for tag, stats in s.get("metrics", {}).items()
            if isinstance(stats, dict)
            for stat in STAT_NAMES
            if stat in stats
        }
    )
    columns = key_fields + metric_columns
    rows = [
        {
            **{k: s.get(k, "") for k in key_fields},
            **{
                f"{tag}.{stat}": stats[stat]
                for tag, stats in s.get("metrics", {}).items()
                if isinstance(stats, dict)
                for stat in STAT_NAMES
                if stat in stats and f"{tag}.{stat}" in metric_columns
            },
        }
        for s in snapshots
    ]
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
        return default if np.isnan(v) or np.isinf(v) else f"{v:.6g}"
    except (TypeError, ValueError):
        return default


def _format_percent(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return default


def _best_eval_metric_final(best_eval: dict[str, Any], key: str) -> Any:
    return best_eval.get(key, {}).get("final")


def _ordered_best_eval_keys(best_eval: dict[str, Any], prefix: str) -> list[str]:
    return [
        f"{prefix}_{m}"
        for m in BEST_EVAL_DISPLAY_METRICS
        if isinstance(best_eval.get(f"{prefix}_{m}"), dict)
    ]


def _ascii_bar(value: Any, width: int = 24) -> str:
    try:
        ratio = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        ratio = 0.0
    w = max(8, int(width))
    filled = int(round(ratio * w))
    return f"[{('#' * filled) + ('.' * (w - filled))}] {_format_percent(ratio)}"


def _short_component_name(tag: str) -> str:
    return tag.split("/")[-1] if "/" in tag else tag


def _render_llm_summary(
    convergence: dict[str, Any],
    best_eval: dict[str, Any],
    top_activity: list[tuple[str, float]],
    strong_negative_pairs: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## LLM Core Summary",
        "",
        f"- final_ep_rew_mean: {_format_number(convergence.get('final_ep_rew_mean'))}",
    ]
    if best_eval.get("available"):
        lines.append(
            "- best_eval: "
            f"arrival_success_rate={_format_percent(_best_eval_metric_final(best_eval, 'best_success'))}, "
            f"precise_arrival_rate={_format_percent(_best_eval_metric_final(best_eval, 'best_precise_arrival'))}, "
            f"punctual_arrival_rate={_format_percent(_best_eval_metric_final(best_eval, 'best_punctual_arrival'))}, "
            f"best_reward={_format_number(_best_eval_metric_final(best_eval, 'best_total_reward'))}"
        )
    if top_activity:
        lines.append(
            "- top_reward_activity: "
            + ", ".join(
                f"{_short_component_name(tag)}={_format_percent(val)}"
                for tag, val in top_activity[:3]
            )
        )
    if strong_negative_pairs:
        top_conflict = strong_negative_pairs[0]
        lines.append(
            f"- top_objective_conflict: {top_conflict.get('left', '')} vs {top_conflict.get('right', '')} "
            f"(pearson={_format_number(top_conflict.get('pearson'))})"
        )
    return lines


def _render_metadata(
    meta: dict[str, Any], reward_analysis: dict[str, Any]
) -> list[str]:
    lines = [
        "## Run Metadata",
        "",
        f"- run_name: {meta.get('run_name', 'unknown')}",
        f"- generated_at_utc: {meta.get('generated_at_utc', 'unknown')}",
        f"- run_directory: {meta.get('run_directory', 'unknown')}",
        f"- tags_count: {len(meta.get('available_tags', []))}",
    ]
    if reward_analysis.get("available"):
        lines.append(
            "- reward_diagnostics: "
            f"transitions={reward_analysis.get('transition_count', 0)}, "
            f"complete_episodes={reward_analysis.get('complete_episode_count', 0)}, "
            f"partial_episodes={reward_analysis.get('partial_episode_count', 0)}"
        )
    return lines


def _render_core_performance(regular: dict[str, Any]) -> list[str]:
    c = regular.get("convergence_speed_quality", {})
    v = regular.get("policy_vitality", {})
    cf = regular.get("critic_foresight", {})
    u = regular.get("update_safety", {})
    return [
        "## Core Training Performance",
        "",
        (
            f"- convergence: final_ep_rew_mean={_format_number(c.get('final_ep_rew_mean'))}, "
            f"rise_slope_per_step={_format_number(c.get('rise_slope_per_step'))}, "
            f"volatility_cv={_format_number(c.get('volatility_cv'))}"
        ),
        (
            f"- policy_vitality: entropy_trend_slope={_format_number(v.get('entropy_trend_slope_per_step'))}, "
            f"rigidity_risk_score={_format_number(v.get('rigidity_risk_score'))}"
        ),
        (
            f"- critic_foresight: explained_variance_mean={_format_number(cf.get('explained_variance_mean'))}, "
            f"low_explained_variance_ratio={_format_percent(cf.get('low_explained_variance_ratio'))}"
        ),
        (
            f"- update_safety: approx_kl_p95={_format_number(u.get('approx_kl_p95'))}, "
            f"approx_kl_exceed_ratio={_format_percent(u.get('approx_kl_exceed_ratio'))}"
        ),
    ]


def _render_best_eval(best_eval: dict[str, Any]) -> list[str]:
    if not best_eval.get("available"):
        return []
    lines = ["## Best Evaluation Performance", ""]
    for group_title, prefix in (
        ("- final_best_values:", "best"),
        ("- last_eval_values:", "last"),
    ):
        keys = _ordered_best_eval_keys(best_eval, prefix)
        if keys:
            lines.append(group_title)
            lines.extend(
                f"  - {k}: {_format_number(best_eval.get(k, {}).get('final'))}"
                for k in keys
            )
    return lines


def _render_reward_diagnostics(
    reward_analysis: dict[str, Any],
    top_activity: list[tuple[str, float]],
    strong_negative_pairs: list[dict[str, Any]],
    bar_width: int,
) -> list[str]:
    lines = ["## Reward Component Diagnostics", ""]
    components = reward_analysis.get("components", {})
    if top_activity:
        lines.append("- absolute_activity_share:")
        for name, value in top_activity:
            m = components.get(name, {})
            lines.append(
                f"  - {name}: {_ascii_bar(value, width=bar_width)}, "
                f"signed_return_ratio={_format_number(m.get('signed_return_ratio'))}, "
                f"nonzero_frequency={_format_percent(m.get('nonzero_frequency'))}, "
                f"active_mean_absolute_strength={_format_number(m.get('active_mean_absolute_strength'))}"
            )
    else:
        lines.append(
            f"- unavailable: {reward_analysis.get('reason', 'reward artifact unavailable')}"
        )

    if strong_negative_pairs:
        lines.append("- objective_conflicts(top):")
        for pair in strong_negative_pairs[:5]:
            lines.append(
                f"  - {pair.get('left', '')} vs {pair.get('right', '')}: "
                f"pearson={_format_number(pair.get('pearson'))}"
            )
    else:
        lines.append("- objective_conflicts: no strong negative pairs detected")
    return lines


def _render_evaluation_trend(trajectory_eval: dict[str, Any]) -> list[str]:
    lines = ["## Evaluation Trend", ""]
    metrics = trajectory_eval.get("metrics", {})
    if trajectory_eval.get("available") and metrics:
        for name, entry in metrics.items():
            if isinstance(entry, dict):
                lines.append(
                    f"- {name}: final={_format_number(entry.get('final'))}, "
                    f"trend_slope_per_step={_format_number(entry.get('trend_slope_per_step'))}"
                )
    else:
        lines.append("- unavailable")
    return lines


def _render_dspdl_distribution(curriculum: dict[str, Any]) -> list[str]:
    lines = ["## DSPDL Distribution", ""]
    if not curriculum.get("available"):
        lines.append("- unavailable: no DSPDL distribution KL logged")
        return lines

    empirical = curriculum.get("empirical_to_target_kl", {})
    lines.append(
        f"- empirical_to_target_kl: final={_format_number(empirical.get('final'))}, "
        f"trend_slope_per_step={_format_number(empirical.get('trend_slope_per_step'))}"
    )
    diagnostics = curriculum.get("diagnostics", {})
    for key in DSPDL_DIAG_KEYS:
        if key in diagnostics:
            entry = diagnostics[key]
            lines.append(
                f"- {key}: final={_format_number(entry.get('final'))}, "
                f"trend_slope_per_step={_format_number(entry.get('trend_slope_per_step'))}"
            )
    return lines


def _render_safety_truncation(safety_position: dict[str, Any]) -> list[str]:
    lines = ["## Safety Truncation Positions", ""]
    if not safety_position.get("available"):
        reason = safety_position.get("reason")
        lines.append(f"- unavailable: {reason}" if reason else "- unavailable")
        return lines

    lines.append(
        f"- total_safety_truncation_count: {_format_number(safety_position.get('total_safety_truncation_count'))}"
    )
    highest = safety_position.get("highest_safety_truncation_bin")
    if highest:
        lines.append(
            f"- highest_safety_truncation_bin: [{_format_number(highest.get('bin_start_m'))}, {_format_number(highest.get('bin_end_m'))}) m, "
            f"count={_format_number(highest.get('safety_truncation_count'))}, "
            f"global_share={_format_percent(highest.get('global_safety_truncation_share'))}"
        )
    bins = safety_position.get("bins", [])
    if bins:
        lines.append("- bins:")
        for entry in bins:
            lines.append(
                f"  - [{_format_number(entry.get('bin_start_m'))}, {_format_number(entry.get('bin_end_m'))}) m: "
                f"count={_format_number(entry.get('safety_truncation_count'))}, "
                f"low_count={_format_number(entry.get('low_safety_truncation_count'))}, "
                f"high_count={_format_number(entry.get('high_safety_truncation_count'))}, "
                f"global_share={_format_percent(entry.get('global_safety_truncation_share'))}"
            )
    else:
        lines.append("- no safety truncations observed")
    return lines


def _render_artifact_summary(
    snapshots: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    return [
        "## Artifact Summary",
        "",
        f"- step_snapshots_count: {len(snapshots.get('by_step', []))}",
        f"- export_csv: {bool(config.get('export_csv', False))}",
    ]


def _generate_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    config = meta.get("config", {})
    regular = payload.get("regular_metrics", {})
    reward_analysis = payload.get("reward_component_analysis", {})
    components = reward_analysis.get("components", {})
    episode_correlation = reward_analysis.get("episode_return_correlation", {})
    strong_negative_pairs = episode_correlation.get("strong_negative_pairs", [])
    bar_width = int(config.get("report_bar_width", 24))

    top_activity = sorted(
        (
            (key, value.get("absolute_activity_share", 0.0))
            for key, value in components.items()
            if isinstance(value, dict)
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    sections = [
        _render_llm_summary(
            regular.get("convergence_speed_quality", {}),
            payload.get("best_eval_metrics", {}),
            top_activity,
            strong_negative_pairs,
        ),
        _render_metadata(meta, reward_analysis),
        _render_core_performance(regular),
        _render_best_eval(payload.get("best_eval_metrics", {})),
        _render_reward_diagnostics(
            reward_analysis, top_activity, strong_negative_pairs, bar_width
        ),
        _render_evaluation_trend(payload.get("trajectory_evaluation_metrics", {})),
        _render_dspdl_distribution(payload.get("curriculum_distribution_metrics", {})),
        _render_safety_truncation(payload.get("safety_truncation_position_metrics", {})),
        _render_artifact_summary(payload.get("snapshots", {}), config),
    ]

    lines = ["# Training Analysis Report", ""]
    for section in sections:
        if section:
            lines.extend(section)
            lines.append("")

    return "\n".join(lines)


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

    _ = markdown_path.write_text(_generate_markdown_report(payload), encoding="utf-8")

    output_paths: dict[str, str] = {
        "output_dir": str(output_dir),
        "json_snapshot": str(json_path),
        "markdown_report": str(markdown_path),
    }

    config = payload.get("meta", {}).get("config", {})
    if not config.get("export_csv", False):
        return output_paths

    summary_csv_path = output_dir / "summary_metrics.csv"
    prefixes = (
        ("regular", "regular_metrics"),
        ("trajectory_evaluation", "trajectory_evaluation_metrics"),
        ("curriculum_distribution", "curriculum_distribution_metrics"),
        ("safety_truncation_position", "safety_truncation_position_metrics"),
        ("reward_component", "reward_component_analysis"),
    )
    summary_metrics: dict[str, float] = {}
    for prefix, key in prefixes:
        _flatten_numeric_fields(
            payload.get(key, {}), prefix=prefix, out=summary_metrics
        )
    _write_csv(
        summary_csv_path,
        columns=sorted(summary_metrics.keys()),
        rows=[summary_metrics],
    )
    output_paths["summary_metrics_csv"] = str(summary_csv_path)

    if config.get("include_snapshots", False):
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

