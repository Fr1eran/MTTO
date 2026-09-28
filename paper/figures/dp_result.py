from __future__ import annotations

import argparse
import functools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from mtto.domain.scenario import Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.domain.srtsp import min_operation_time
from mtto.evaluation.quality import QualityReport
from mtto.io.artifacts import RunKind, read_completed_run, task_from_json
from paper.figures import load_paper_scenario
from paper.plotting.profiles import render_dp_curve_on_axes
from paper.plotting.style import (
    apply_sci_grid,
    save_sci_figure,
    set_global_plot_style,
)

FIGURE_FILENAME = "dp_result.pdf"


def _print_metrics(
    *,
    quality: QualityReport,
    task: Task,
    created_at: str,
) -> None:
    display_metrics: dict[str, Any] = {
        "target_time_s": task.schedule_time_s,
        "total_time_s": quality.metrics.run_time_s,
        "time_error_s": quality.metrics.arrival_time_error_s,
        "start_position_m": task.start_position_m,
        "target_position_m": task.target_position_m,
        "total_energy_kj": quality.metrics.total_energy_kj,
        "total_energy_j": quality.metrics.total_energy_kj * 1000.0,
        "comfort_tav": quality.metrics.comfort_tav_mps2,
        "comfort_er_pct": quality.metrics.comfort_exceedance_pct,
        "comfort_rms": quality.metrics.comfort_rms_mps2,
        "created_at": created_at,
    }
    print("Loaded metrics:")
    for key, val in display_metrics.items():
        if val is not None:
            print(f"  {key}: {val}")


def _calc_redundant_operation_time_arr(
    *,
    pos_arr: NDArray[np.float64],
    speed_arr: NDArray[np.float64],
    cum_time_arr: NDArray[np.float64],
    task: Task,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> NDArray[np.float64]:
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)
    cum_time = np.asarray(cum_time_arr, dtype=np.float64)

    if pos.ndim != 1 or speed.ndim != 1 or cum_time.ndim != 1:
        raise ValueError("pos_arr, speed_arr, and cum_time_arr must be 1-D arrays")
    if not (pos.size == speed.size == cum_time.size):
        raise ValueError(
            "pos_arr, speed_arr, and cum_time_arr must have the same length"
        )
    if pos.size == 0:
        raise ValueError("DP curve must contain at least one point")
    if task.schedule_time_s is None:
        raise ValueError("Task schedule_time_s must not be None")

    target_position = task.target_position_m
    target_speed = 0.0

    min_remaining_arr = np.asarray(
        [
            min_remaining_time_fn(
                float(pos_val),
                float(speed_val),
                float(target_position),
                float(target_speed),
            )
            for pos_val, speed_val in zip(pos, speed, strict=False)
        ],
        dtype=np.float64,
    )
    return float(task.schedule_time_s) - cum_time - min_remaining_arr


def _render_redundant_operation_time_on_axes(
    *,
    ax: Axes,
    pos_arr: NDArray[np.float64],
    redundant_operation_time_arr: NDArray[np.float64],
) -> None:
    _ = ax.plot(
        pos_arr,
        redundant_operation_time_arr,
        color="#16a34a",
        linewidth=1.5,
        label="DP redundant operation time",
    )
    _ = ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label="No redundancy",
    )
    _ = ax.set_xlabel("Position (m)")
    _ = ax.set_ylabel("Redundant operation time (s)")
    apply_sci_grid(ax)
    _ = ax.legend(loc="best")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and display saved DP optimized speed curve."
    )
    _ = parser.add_argument(
        "--dp-run",
        type=Path,
        required=True,
        help="DP solve run directory containing run.json, profile.npz, quality.json",
    )
    _ = parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="Do not draw safeguard background.",
    )
    _ = parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="Safeguard factor used for rendering when safeguard is enabled.",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Optional figure directory; saves {FIGURE_FILENAME}.",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Do not display the interactive plot window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        completed_run = read_completed_run(args.dp_run)
    except Exception as exc:
        parser.error(f"Failed to read DP run from {args.dp_run}: {exc}")

    if completed_run.record.kind != RunKind.DP_SOLVE:
        parser.error(f"Expected DP solve run, got {completed_run.record.kind.value}")

    scenario = load_paper_scenario()
    if completed_run.record.scenario_hash != scenario.scenario_hash:
        parser.error(
            f"Scenario hash mismatch for run at {args.dp_run}: "
            f"expected {scenario.scenario_hash}, "
            f"got {completed_run.record.scenario_hash}"
        )

    task = task_from_json(completed_run.record.task)
    profile: SpeedProfile = completed_run.payload.profile
    quality: QualityReport = completed_run.payload.quality

    _print_metrics(
        quality=quality,
        task=task,
        created_at=completed_run.record.created_at,
    )

    min_remaining_time_fn = functools.partial(
        min_operation_time,
        scenario.vehicle,
        scenario.line,
        scenario.safeguard.params.factor,
    )

    try:
        redundant_operation_time_arr = _calc_redundant_operation_time_arr(
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            cum_time_arr=profile.time_s,
            task=task,
            min_remaining_time_fn=min_remaining_time_fn,
        )
    except ValueError as exc:
        parser.error(str(exc))

    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Arial",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
    )

    fig, (ax_speed, ax_redundant) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    render_dp_curve_on_axes(
        ax=ax_speed,
        profile=profile,
        task=task,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color="blue",
    )

    ax_speed.legend(loc="upper right")
    _render_redundant_operation_time_on_axes(
        ax=ax_redundant,
        pos_arr=profile.position_m,
        redundant_operation_time_arr=redundant_operation_time_arr,
    )
    _ = ax_speed.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_speed.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    _ = ax_redundant.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_redundant.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout()

    if args.output_dir is not None:
        saved_path = save_sci_figure(fig, args.output_dir / FIGURE_FILENAME)
        print(f"Saved figure to: {saved_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
