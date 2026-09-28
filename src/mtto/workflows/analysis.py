"""Workflow for training log analysis and report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtto.io.artifacts import (
    read_completed_run,
    write_analysis_report,
)
from mtto.io.tensorboard import load_scalar_series_from_run, resolve_run_directory
from mtto.rl.diagnostics import RewardDiagnostics, SafetyTruncationHistogram
from mtto.rl.training_analysis.collect import ScalarSeries
from mtto.rl.training_analysis.output import (
    render_markdown_report,
    snapshot_csv_table,
    summary_csv_table,
)
from mtto.rl.training_analysis.pipeline import AnalysisConfig, analyze

__all__ = ["AnalysisResult", "analyze_training"]


@dataclass(frozen=True, slots=True, eq=False)
class AnalysisResult:
    payload: dict[str, Any]
    output_paths: dict[str, str]


def analyze_training(
    *,
    config: AnalysisConfig,
    output_root: str | Path = "mtto_train_reports",
    run_name: str | None = None,
    train_run_dir: str | Path | None = None,
    tensorboard_log_root: str | Path | None = None,
    tensorboard_run_name: str | None = None,
) -> AnalysisResult:
    """Orchestrate training log analysis, metric aggregation, and report generation."""
    reward_diagnostics: RewardDiagnostics | None = None
    safety_histogram: SafetyTruncationHistogram | None = None

    if train_run_dir is not None:
        completed = read_completed_run(train_run_dir)
        if completed.payload.diagnostics is not None:
            reward_diagnostics = completed.payload.diagnostics.reward
            safety_histogram = completed.payload.diagnostics.safety

    series_map: dict[str, ScalarSeries] = {}
    tb_dir: Path | None = None
    if tensorboard_log_root is not None:
        try:
            tb_dir = resolve_run_directory(
                log_root=tensorboard_log_root, run_name=tensorboard_run_name
            )
        except FileNotFoundError:
            if train_run_dir is None:
                raise
        else:
            series_map = load_scalar_series_from_run(tb_dir)

    resolved_run_name = run_name
    if resolved_run_name is None:
        if tb_dir is not None:
            resolved_run_name = tb_dir.name
        elif train_run_dir is not None:
            resolved_run_name = Path(train_run_dir).name
        else:
            resolved_run_name = "training_run"

    target_output_dir = Path(output_root) / resolved_run_name

    payload = analyze(
        series_map=series_map,
        reward_diagnostics=reward_diagnostics,
        safety_histogram=safety_histogram,
        config=config,
        run_name=resolved_run_name,
        run_directory=str(tb_dir) if tb_dir is not None else "",
    )

    markdown = render_markdown_report(payload)
    csv_tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    if config.export_csv:
        summary_cols, summary_rows = summary_csv_table(payload)
        csv_tables["summary_metrics"] = (summary_cols, summary_rows)
        if config.include_snapshots:
            snap_cols, snap_rows = snapshot_csv_table(payload)
            if snap_cols:
                csv_tables["step_snapshots"] = (snap_cols, snap_rows)

    output_paths = write_analysis_report(
        target_output_dir,
        payload=payload,
        markdown=markdown,
        csv_tables=csv_tables if csv_tables else None,
    )

    return AnalysisResult(payload=payload, output_paths=output_paths)
