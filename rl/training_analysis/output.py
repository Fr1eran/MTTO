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
    step_snapshots: list[dict[str, Any]],
    episode_snapshots: list[dict[str, Any]],
    config: dict[str, Any],
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


def _generate_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    regular = payload.get("regular_metrics", {})
    reward_impact = payload.get("reward_component_impact", {})
    constraint = payload.get("constraint_diagnostic", {})
    snapshots = payload.get("snapshots", {})

    convergence = regular.get("convergence_speed_quality", {})
    vitality = regular.get("policy_vitality", {})
    critic = regular.get("critic_foresight", {})
    update_safety = regular.get("update_safety", {})

    dominance = reward_impact.get("dominance", {})
    correlation = reward_impact.get("objective_correlation", {})
    strong_negative_pairs = correlation.get("strong_negative_pairs", [])

    gfd = constraint.get("geographic_failure_distribution", {})
    sbt = constraint.get("safety_band_tolerance", {})

    top_dominance = sorted(
        ((key, value) for key, value in dominance.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    lines: list[str] = []
    lines.append("# Training Data Full-Dimension Quantitative Analysis")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- run_name: {meta.get('run_name', 'unknown')}")
    lines.append(f"- generated_at_utc: {meta.get('generated_at_utc', 'unknown')}")
    lines.append(f"- run_directory: {meta.get('run_directory', 'unknown')}")
    lines.append(f"- tags_count: {len(meta.get('available_tags', []))}")
    lines.append("")
    lines.append("## Regular Training Metrics")
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
        f"low_explained_variance_ratio={_format_number(critic.get('low_explained_variance_ratio'))}"
    )
    lines.append(
        "- update_safety: "
        f"approx_kl_p95={_format_number(update_safety.get('approx_kl_p95'))}, "
        f"approx_kl_exceed_ratio={_format_number(update_safety.get('approx_kl_exceed_ratio'))}"
    )
    lines.append("")
    lines.append("## Reward Component Impact")
    lines.append("")
    if top_dominance:
        for key, value in top_dominance:
            lines.append(f"- dominance {key}: {_format_number(value)}")
    else:
        lines.append("- dominance: no component series available")

    if strong_negative_pairs:
        lines.append("- objective_conflicts:")
        for pair in strong_negative_pairs[:10]:
            left = pair.get("left", "")
            right = pair.get("right", "")
            pearson = _format_number(pair.get("pearson"))
            lines.append(f"  - {left} vs {right}: pearson={pearson}")
    else:
        lines.append("- objective_conflicts: no strong negative pairs detected")

    lines.append("")
    lines.append("## Constraint Diagnostic")
    lines.append("")
    lines.append(
        "- geographic_failure_distribution: "
        f"truncated_count={gfd.get('truncated_count', 0)}"
    )
    lines.append(
        "- safety_band_tolerance: "
        f"avg_distance_to_vmax={_format_number(sbt.get('average_distance_to_vmax_mps'))}, "
        f"avg_distance_to_vmin={_format_number(sbt.get('average_distance_to_vmin_mps'))}, "
        f"near_miss_ratio={_format_number(sbt.get('near_miss_ratio'))}"
    )
    lines.append("")
    lines.append("## LLM-Ready Snapshot Summary")
    lines.append("")
    lines.append(f"- step_snapshots_count: {len(snapshots.get('by_step', []))}")
    lines.append(f"- episode_snapshots_count: {len(snapshots.get('by_episode', []))}")

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
    summary_csv_path = output_dir / "summary_metrics.csv"
    step_csv_path = output_dir / "step_snapshots.csv"
    episode_csv_path = output_dir / "episode_snapshots.csv"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)

    markdown_report = _generate_markdown_report(payload)
    markdown_path.write_text(markdown_report, encoding="utf-8")

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
    _write_csv(
        summary_csv_path,
        columns=sorted(summary_metrics.keys()),
        rows=[summary_metrics],
    )

    step_snapshots = payload.get("snapshots", {}).get("by_step", [])
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

    episode_snapshots = payload.get("snapshots", {}).get("by_episode", [])
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

    return {
        "output_dir": str(output_dir),
        "json_snapshot": str(json_path),
        "markdown_report": str(markdown_path),
        "summary_metrics_csv": str(summary_csv_path),
        "step_snapshots_csv": str(step_csv_path),
        "episode_snapshots_csv": str(episode_csv_path),
    }
