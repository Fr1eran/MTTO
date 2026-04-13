from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def build_analysis_payload(
    *,
    run_name: str,
    run_directory: str,
    available_tags: list[str],
    regular_metrics: dict[str, Any],
    reward_component_impact: dict[str, Any],
    constraint_diagnostic: dict[str, Any],
    evolution_metrics: dict[str, Any],
    step_snapshots: list[dict[str, Any]],
    episode_snapshots: list[dict[str, Any]],
    config: dict[str, Any],
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "run_directory": run_directory,
            "available_tags": sorted(available_tags),
            "config": config,
        },
        "regular_metrics": regular_metrics,
        "reward_component_impact": reward_component_impact,
        "constraint_diagnostic": constraint_diagnostic,
        "evolution_metrics": evolution_metrics,
        "data_quality": data_quality or {},
        "snapshots": {
            "by_step": step_snapshots,
            "by_episode": episode_snapshots,
        },
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
            _flatten_numeric_fields(child, prefix=next_prefix, out=out)
        return out

    if isinstance(value, list):
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
        return f"{float(value):.6g}"
    except TypeError, ValueError:
        return default


def _format_percent(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100.0:.2f}%"
    except TypeError, ValueError:
        return default


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


def _append_transition_matrix(
    lines: list[str],
    *,
    matrix: Any,
    labels: list[str],
) -> None:
    if not isinstance(matrix, list) or not matrix:
        lines.append("- violation_transition_matrix: unavailable")
        return

    lines.append("| from \\ to | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
    for row_label, row in zip(labels, matrix):
        if not isinstance(row, list):
            continue
        values = [str(int(v)) for v in row[: len(labels)]]
        if len(values) < len(labels):
            values.extend(["0"] * (len(labels) - len(values)))
        lines.append("| " + row_label + " | " + " | ".join(values) + " |")


def _generate_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    config = meta.get("config", {})
    regular = payload.get("regular_metrics", {})
    reward_impact = payload.get("reward_component_impact", {})
    constraint = payload.get("constraint_diagnostic", {})
    evolution = payload.get("evolution_metrics", {})
    snapshots = payload.get("snapshots", {})

    convergence = regular.get("convergence_speed_quality", {})
    vitality = regular.get("policy_vitality", {})
    critic = regular.get("critic_foresight", {})
    update_safety = regular.get("update_safety", {})

    dominance = reward_impact.get("dominance", {})
    stage_component_profile = reward_impact.get("stage_component_profile", [])
    aggregation_order = reward_impact.get("aggregation_order", "unknown")
    strong_negative_pairs = (reward_impact.get("objective_correlation", {}) or {}).get(
        "strong_negative_pairs", []
    )

    gfd = constraint.get("geographic_failure_distribution", {})
    sbt = constraint.get("safety_band_tolerance", {})
    boundary_adhesion = constraint.get("boundary_adhesion", {})
    critical_points = constraint.get("critical_point_risk", {})

    evolution_available = bool(evolution.get("available", False))
    evolution_stage_profiles = evolution.get("stage_profiles", [])
    state_labels = evolution.get(
        "state_labels", ["normal", "low_violation", "high_violation"]
    )
    overall_transition_matrix = evolution.get("overall_transition_matrix", [])

    bar_width = (
        int(config.get("report_bar_width", 24)) if isinstance(config, dict) else 24
    )

    top_dominance = sorted(
        ((key, value) for key, value in dominance.items()),
        key=lambda item: item[1],
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
    lines.append(
        "- boundary_adhesion_ratio_by_distance: "
        f"{_format_percent(boundary_adhesion.get('near_miss_distance_ratio'))}"
    )
    lines.append(
        "- mean_survival_distance_m: "
        f"{_format_number(evolution.get('mean_survival_distance_m'))}"
    )
    lines.append(
        "- overall_normal_to_high_transition_rate: "
        f"{_format_percent((evolution.get('overall_transition_probabilities') or [[0, 0, 0]])[0][2] if evolution.get('overall_transition_probabilities') else 0.0)}"
    )
    sample_count = float(sbt.get("sample_count", 0.0)) if isinstance(sbt, dict) else 0.0
    failure_count = (
        float(gfd.get("truncated_count", 0.0)) if isinstance(gfd, dict) else 0.0
    )
    failure_ratio = (failure_count / sample_count) if sample_count > 0 else 0.0
    lines.append(
        "- safety_signal_ratio(sample): "
        f"failure={_format_percent(failure_ratio)}, "
        f"violation={_format_percent(sbt.get('violation_ratio'))}, "
        f"near_miss={_format_percent(sbt.get('near_miss_ratio'))}"
    )

    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- run_name: {meta.get('run_name', 'unknown')}")
    lines.append(f"- generated_at_utc: {meta.get('generated_at_utc', 'unknown')}")
    lines.append(f"- run_directory: {meta.get('run_directory', 'unknown')}")
    lines.append(f"- tags_count: {len(meta.get('available_tags', []))}")
    lines.append(f"- aggregation_order(reward): {aggregation_order}")
    lines.append(f"- stage_basis(evolution): {evolution.get('stage_basis', 'unknown')}")

    lines.append("")
    lines.append("## Core Training Performance")
    lines.append("")
    lines.append(
        "- convergence: "
        f"final_ep_rew_mean={_format_number(convergence.get('final_ep_rew_mean'))}, "
        f"rise_slope_per_step={_format_number(convergence.get('rise_slope_per_step'))}, "
        f"volatility_cv={_format_number(convergence.get('volatility_cv'))}"
    )
    lines.append(
        "- policy_vitality: "
        f"entropy_trend_slope={_format_number(vitality.get('entropy_trend_slope_per_step'))}, "
        f"rigidity_risk_score={_format_number(vitality.get('rigidity_risk_score'))}"
    )
    lines.append(
        "- critic_foresight: "
        f"explained_variance_mean={_format_number(critic.get('explained_variance_mean'))}, "
        f"low_explained_variance_ratio={_format_percent(critic.get('low_explained_variance_ratio'))}"
    )
    lines.append(
        "- update_safety: "
        f"approx_kl_p95={_format_number(update_safety.get('approx_kl_p95'))}, "
        f"approx_kl_exceed_ratio={_format_percent(update_safety.get('approx_kl_exceed_ratio'))}"
    )

    lines.append("")
    lines.append("## Reward Quality (Episode -> Stage)")
    lines.append("")
    if top_dominance:
        lines.append("- dominance_by_component:")
        for tag, value in top_dominance:
            lines.append(
                f"  - {_short_component_name(tag)}: {_ascii_bar(value, width=bar_width)}"
            )
    else:
        lines.append("- dominance_by_component: unavailable")

    if strong_negative_pairs:
        lines.append("- objective_conflicts(top):")
        for pair in strong_negative_pairs[:5]:
            lines.append(
                "  - "
                f"{pair.get('left', '')} vs {pair.get('right', '')}: "
                f"pearson={_format_number(pair.get('pearson'))}"
            )
    else:
        lines.append("- objective_conflicts: no strong negative pairs detected")

    if isinstance(stage_component_profile, list) and stage_component_profile:
        component_keys = sorted(
            {
                key
                for stage in stage_component_profile
                if isinstance(stage, dict)
                for key in (stage.get("mean_ratio", {}) or {}).keys()
            }
        )
        if component_keys:
            lines.append("")
            header = (
                "| stage | episode_range | "
                + " | ".join(_short_component_name(key) for key in component_keys)
                + " |"
            )
            divider = (
                "| --- | --- | " + " | ".join(["---"] * len(component_keys)) + " |"
            )
            lines.append(header)
            lines.append(divider)
            for stage in stage_component_profile[:10]:
                if not isinstance(stage, dict):
                    continue
                ratio_map = stage.get("mean_ratio", {}) or {}
                ratio_cells = [
                    _format_percent(ratio_map.get(key, 0.0)) for key in component_keys
                ]
                lines.append(
                    "| "
                    f"{stage.get('window_index', 'N/A')} | "
                    f"[{stage.get('episode_start', 'N/A')}, {stage.get('episode_end', 'N/A')}) | "
                    + " | ".join(ratio_cells)
                    + " |"
                )

    lines.append("")
    lines.append("## Physical Trajectory and Compliance")
    lines.append("")
    lines.append(
        "- boundary_adhesion_ratio_by_distance: "
        f"{_ascii_bar(boundary_adhesion.get('near_miss_distance_ratio', 0.0), width=bar_width)}"
    )
    lines.append(
        "- boundary_adhesion_distance: "
        f"near_miss_distance_m={_format_number(boundary_adhesion.get('near_miss_distance_m'))}, "
        f"total_distance_m={_format_number(boundary_adhesion.get('total_distance_m'))}"
    )
    lines.append(
        "- geographic_failure_distribution: "
        f"truncated_count={gfd.get('truncated_count', 0)}"
    )
    lines.append(
        "- safety_band_tolerance: "
        f"avg_distance_to_vmax={_format_number(sbt.get('average_distance_to_vmax_mps'))}, "
        f"avg_distance_to_vmin={_format_number(sbt.get('average_distance_to_vmin_mps'))}, "
        f"near_miss_ratio(sample)={_format_percent(sbt.get('near_miss_ratio'))}"
    )

    top_risk_bins = gfd.get("top_risk_bins", [])
    if isinstance(top_risk_bins, list) and top_risk_bins:
        lines.append("")
        lines.append("### Violation Spatial Distribution (Top Risk Bins)")
        lines.append("")
        lines.append(
            "| bin_m | exposure | near_miss | violation | failure | near_miss_risk | violation_risk | failure_risk | chart |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in top_risk_bins[:10]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"[{_format_number(item.get('bin_start_m'))}, {_format_number(item.get('bin_end_m'))}) | "
                f"{item.get('exposure_count', 0)} | "
                f"{item.get('near_miss_count', 0)} | "
                f"{item.get('violation_count', 0)} | "
                f"{item.get('failure_count', 0)} | "
                f"{_format_percent(item.get('near_miss_risk'))} | "
                f"{_format_percent(item.get('violation_risk'))} | "
                f"{_format_percent(item.get('failure_risk'))} | "
                f"{_ascii_bar(item.get('near_miss_risk', 0.0), width=bar_width)} |"
            )

    top_risky_points = critical_points.get("top_risky_points", [])
    if isinstance(top_risky_points, list) and top_risky_points:
        lines.append("")
        lines.append("### Critical Point Attribution")
        lines.append("")
        lines.append(
            "| type | point_m | exposure | near_miss | violation | failure | near_miss_risk | violation_risk | failure_risk |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in top_risky_points[:10]:
            if not isinstance(item, dict):
                continue
            failure_risk = item.get("failure_risk", item.get("risk"))
            lines.append(
                "| "
                f"{item.get('type', 'N/A')} | "
                f"{_format_number(item.get('point_m'))} | "
                f"{item.get('exposure_count', 0)} | "
                f"{item.get('near_miss_count', 0)} | "
                f"{item.get('violation_count', 0)} | "
                f"{item.get('failure_count', 0)} | "
                f"{_format_percent(item.get('near_miss_risk'))} | "
                f"{_format_percent(item.get('violation_risk'))} | "
                f"{_format_percent(failure_risk)} |"
            )

    lines.append("")
    lines.append("## Evolution Metrics")
    lines.append("")
    lines.append(
        "- survival_distance_slope_per_stage: "
        f"{_format_number(evolution.get('survival_distance_slope_per_stage'))}"
    )
    lines.append(
        "- mean_survival_distance_m: "
        f"{_format_number(evolution.get('mean_survival_distance_m'))}"
    )
    lines.append(
        "- truncated_episode_ratio: "
        f"{_format_percent(evolution.get('truncated_episode_ratio'))}"
    )

    if evolution_available:
        lines.append("")
        lines.append("### Violation Type Transition Matrix (Overall)")
        lines.append("")
        _append_transition_matrix(
            lines,
            matrix=overall_transition_matrix,
            labels=[str(label) for label in state_labels],
        )

    if isinstance(evolution_stage_profiles, list) and evolution_stage_profiles:
        lines.append("")
        lines.append("### Stage Evolution")
        lines.append("")
        lines.append(
            "| stage | episode_range | avg_survival_m | growth_vs_prev | normal->high | low->high | high->low |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for stage in evolution_stage_profiles[:10]:
            if not isinstance(stage, dict):
                continue
            growth = stage.get("survival_growth_rate_vs_prev")
            growth_text = _format_percent(growth) if growth is not None else "N/A"
            lines.append(
                "| "
                f"{stage.get('window_index', 'N/A')} | "
                f"[{stage.get('episode_start', 'N/A')}, {stage.get('episode_end', 'N/A')}) | "
                f"{_format_number(stage.get('avg_survival_distance_m'))} | "
                f"{growth_text} | "
                f"{_format_percent(stage.get('normal_to_high_transition_rate'))} | "
                f"{_format_percent(stage.get('low_to_high_transition_rate'))} | "
                f"{_format_percent(stage.get('high_to_low_transition_rate'))} |"
            )

    lines.append("")
    lines.append("## Artifact Summary")
    lines.append("")
    lines.append(f"- step_snapshots_count: {len(snapshots.get('by_step', []))}")
    lines.append(f"- episode_snapshots_count: {len(snapshots.get('by_episode', []))}")
    lines.append(
        f"- export_csv: {bool(config.get('export_csv', False)) if isinstance(config, dict) else False}"
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
    markdown_path.write_text(markdown_report, encoding="utf-8")

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
            payload.get("reward_component_impact", {}),
            prefix="reward_impact",
        )
    )
    summary_metrics.update(
        _flatten_numeric_fields(
            payload.get("constraint_diagnostic", {}),
            prefix="constraint",
        )
    )
    summary_metrics.update(
        _flatten_numeric_fields(
            payload.get("evolution_metrics", {}),
            prefix="evolution",
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

        episode_snapshots = payload.get("snapshots", {}).get("by_episode", [])
        if episode_snapshots:
            episode_csv_path = output_dir / "episode_snapshots.csv"
            episode_columns, episode_rows = _snapshot_rows(
                episode_snapshots,
                key_fields=[
                    "window_type",
                    "window_index",
                    "episode_start",
                    "episode_end",
                    "step_start",
                    "step_end",
                    "sample_count",
                    "severity",
                ],
            )
            _write_csv(episode_csv_path, columns=episode_columns, rows=episode_rows)
            output_paths["episode_snapshots_csv"] = str(episode_csv_path)

    return output_paths
