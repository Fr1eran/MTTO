from __future__ import annotations

import argparse
import csv
import functools
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from mtto.domain.srtsp import min_operation_time
from mtto.io.artifacts import RunKind, read_completed_run, task_from_json
from paper.figures import load_paper_scenario
from paper.plotting.style import (
    apply_sci_figure_layout,
    apply_sci_grid,
    save_sci_figure,
    set_global_plot_style,
)

FIGURE_FILENAME = "dp_redundancy_error.pdf"


def _validate_same_length(
    named_arrays: Sequence[tuple[str, NDArray[np.float64]]],
) -> None:
    sizes = {arr.size for _name, arr in named_arrays}
    if len(sizes) != 1:
        names = ", ".join(name for name, _arr in named_arrays)
        raise ValueError(f"{names} must have the same length")


def reconstruct_redundant_operation_time(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    speed_arr: Sequence[float] | NDArray[Any],
    cum_time_arr: Sequence[float] | NDArray[Any],
    schedule_time_s: float,
    target_position: float,
    target_speed: float,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> NDArray[np.float64]:
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)
    cum_time = np.asarray(cum_time_arr, dtype=np.float64)

    if pos.ndim != 1 or speed.ndim != 1 or cum_time.ndim != 1:
        raise ValueError("pos_arr, speed_arr, and cum_time_arr must be 1-D arrays")
    _validate_same_length(
        (("pos_arr", pos), ("speed_arr", speed), ("cum_time_arr", cum_time))
    )

    if not math.isfinite(float(schedule_time_s)):
        raise ValueError("schedule_time_s must be finite")

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
    if min_remaining_arr.ndim != 1 or min_remaining_arr.size != pos.size:
        raise ValueError(
            "min_remaining_time_fn must return one finite scalar per sample"
        )
    if not np.all(np.isfinite(min_remaining_arr)):
        raise ValueError("min_remaining_time_fn returned non-finite values")
    return float(schedule_time_s) - cum_time - min_remaining_arr


def compute_expected_redundant_operation_time(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    start_position: float,
    target_position: float,
    initial_redundant_s: float,
) -> NDArray[np.float64]:
    pos = np.asarray(pos_arr, dtype=np.float64)
    if pos.ndim != 1:
        raise ValueError("pos_arr must be a 1-D array")
    denominator = float(target_position) - float(start_position)
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise ValueError("start_position and target_position must be distinct")
    if not math.isfinite(float(initial_redundant_s)):
        raise ValueError("initial_redundant_s must be finite")

    progress = np.clip((pos - float(start_position)) / denominator, 0.0, 1.0)
    return float(initial_redundant_s) * (1.0 - progress)


def _series_stats(values: NDArray[np.float64]) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "sample_count": 0,
            "mean_s": None,
            "std_s": None,
            "min_s": None,
            "max_s": None,
        }
    return {
        "sample_count": int(values.size),
        "mean_s": float(np.mean(values)),
        "std_s": float(np.std(values)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def summarize_error_statistics(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    cum_time_arr: Sequence[float] | NDArray[Any],
    error_arr: Sequence[float] | NDArray[Any],
    zero_eps: float = 1e-9,
) -> dict[str, Any]:
    pos = np.asarray(pos_arr, dtype=np.float64)
    cum_time = np.asarray(cum_time_arr, dtype=np.float64)
    error = np.asarray(error_arr, dtype=np.float64)
    if pos.ndim != 1 or cum_time.ndim != 1 or error.ndim != 1:
        raise ValueError("Arrays must be 1-D")
    _validate_same_length(
        (("pos_arr", pos), ("cum_time_arr", cum_time), ("error_arr", error))
    )
    if zero_eps < 0.0 or not math.isfinite(float(zero_eps)):
        raise ValueError("zero_eps must be finite and non-negative")

    abs_error = np.abs(error)
    max_abs_idx = int(np.argmax(abs_error))
    positive_mask = error > float(zero_eps)
    negative_mask = error < -float(zero_eps)
    near_zero_mask = ~(positive_mask | negative_mask)

    def subset_summary(mask: NDArray[np.bool_], *, kind: str) -> dict[str, Any]:
        idx = np.flatnonzero(mask)
        values = error[idx]
        summary: dict[str, Any] = _series_stats(values)
        summary["fraction"] = float(values.size / error.size)
        if values.size == 0:
            if kind == "positive":
                summary.update({"max_position_m": None, "max_cum_time_s": None})
            elif kind == "negative":
                summary.update(
                    {
                        "min_position_m": None,
                        "min_cum_time_s": None,
                        "max_abs_s": None,
                        "max_abs_position_m": None,
                        "max_abs_cum_time_s": None,
                    }
                )
            return summary

        if kind == "positive":
            local_idx = int(idx[int(np.argmax(values))])
            summary.update(
                {
                    "max_position_m": float(pos[local_idx]),
                    "max_cum_time_s": float(cum_time[local_idx]),
                }
            )
        elif kind == "negative":
            local_idx = int(idx[int(np.argmin(values))])
            summary.update(
                {
                    "min_position_m": float(pos[local_idx]),
                    "min_cum_time_s": float(cum_time[local_idx]),
                    "max_abs_s": float(abs(error[local_idx])),
                    "max_abs_position_m": float(pos[local_idx]),
                    "max_abs_cum_time_s": float(cum_time[local_idx]),
                }
            )
        return summary

    return {
        "overall": {
            "sample_count": int(error.size),
            "mean_s": float(np.mean(error)),
            "std_s": float(np.std(error)),
            "min_s": float(np.min(error)),
            "max_s": float(np.max(error)),
            "mae_s": float(np.mean(abs_error)),
            "rmse_s": float(np.sqrt(np.mean(error**2))),
            "max_abs_s": float(abs_error[max_abs_idx]),
            "max_abs_position_m": float(pos[max_abs_idx]),
            "max_abs_cum_time_s": float(cum_time[max_abs_idx]),
        },
        "positive": subset_summary(positive_mask, kind="positive"),
        "negative": subset_summary(negative_mask, kind="negative"),
        "near_zero": {
            "sample_count": int(np.count_nonzero(near_zero_mask)),
            "fraction": float(np.count_nonzero(near_zero_mask) / error.size),
        },
    }


def plot_redundancy_error_series(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    actual_redundant_arr: Sequence[float] | NDArray[Any],
    expected_redundant_arr: Sequence[float] | NDArray[Any],
    error_arr: Sequence[float] | NDArray[Any],
) -> Figure:
    pos = np.asarray(pos_arr, dtype=np.float64)
    actual = np.asarray(actual_redundant_arr, dtype=np.float64)
    expected = np.asarray(expected_redundant_arr, dtype=np.float64)
    error = np.asarray(error_arr, dtype=np.float64)
    _validate_same_length(
        (
            ("pos_arr", pos),
            ("actual_redundant_arr", actual),
            ("expected_redundant_arr", expected),
            ("error_arr", error),
        )
    )

    fig, (ax_redundant, ax_error) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_redundant.plot(
        pos,
        actual,
        color="#2563eb",
        linewidth=1.6,
        label="Actual redundant time",
    )
    ax_redundant.plot(
        pos,
        expected,
        color="#f97316",
        linewidth=1.6,
        linestyle="--",
        label="Expected redundant time",
    )
    ax_redundant.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
        alpha=0.7,
        label="No redundancy",
    )
    ax_redundant.set_ylabel("Redundant time (s)")
    apply_sci_grid(ax_redundant)
    ax_redundant.legend(loc="best")

    ax_error.plot(
        pos,
        error,
        color="#9333ea",
        linewidth=1.4,
        label="Actual - expected",
    )
    ax_error.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
        alpha=0.7,
        label="Zero error",
    )
    ax_error.set_xlabel("Position (m)")
    ax_error.set_ylabel("Error (s)")
    apply_sci_grid(ax_error)
    ax_error.legend(loc="best")
    _ = ax_redundant.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_redundant.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    _ = ax_error.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_error.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    apply_sci_figure_layout(
        fig,
        columns=1,
        height_in=4.2,
        left=0.20,
        bottom=0.13,
        top=0.96,
        hspace=0.22,
    )
    return fig


def save_compact_figure(
    fig: Figure,
    output_dir: Path,
) -> Path:
    return save_sci_figure(fig, output_dir / FIGURE_FILENAME)


def write_point_csv(
    *,
    output_path: Path,
    pos_arr: Sequence[float] | NDArray[Any],
    speed_arr: Sequence[float] | NDArray[Any],
    cum_time_arr: Sequence[float] | NDArray[Any],
    actual_redundant_arr: Sequence[float] | NDArray[Any],
    expected_redundant_arr: Sequence[float] | NDArray[Any],
    error_arr: Sequence[float] | NDArray[Any],
) -> Path:
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)
    cum_time = np.asarray(cum_time_arr, dtype=np.float64)
    actual = np.asarray(actual_redundant_arr, dtype=np.float64)
    expected = np.asarray(expected_redundant_arr, dtype=np.float64)
    error = np.asarray(error_arr, dtype=np.float64)
    _validate_same_length(
        (
            ("pos_arr", pos),
            ("speed_arr", speed),
            ("cum_time_arr", cum_time),
            ("actual_redundant_arr", actual),
            ("expected_redundant_arr", expected),
            ("error_arr", error),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "position_m",
                "speed_mps",
                "cum_time_s",
                "actual_redundant_time_s",
                "expected_redundant_time_s",
                "error_s",
            ],
        )
        writer.writeheader()
        for values in zip(pos, speed, cum_time, actual, expected, error, strict=True):
            writer.writerow(
                {
                    "position_m": float(values[0]),
                    "speed_mps": float(values[1]),
                    "cum_time_s": float(values[2]),
                    "actual_redundant_time_s": float(values[3]),
                    "expected_redundant_time_s": float(values[4]),
                    "error_s": float(values[5]),
                }
            )
    return output_path


def write_summary_json(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _fmt_optional(value: object, *, precision: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return str(value)


def print_error_statistics(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    positive = summary["positive"]
    negative = summary["negative"]
    near_zero = summary["near_zero"]

    print("Redundant-time error summary:")
    print("  overall:")
    for key in [
        "sample_count",
        "mean_s",
        "std_s",
        "min_s",
        "max_s",
        "mae_s",
        "rmse_s",
        "max_abs_s",
        "max_abs_position_m",
        "max_abs_cum_time_s",
    ]:
        print(f"    {key}: {_fmt_optional(overall.get(key))}")

    print("  positive errors:")
    for key in [
        "sample_count",
        "fraction",
        "max_s",
        "mean_s",
        "std_s",
        "max_position_m",
        "max_cum_time_s",
    ]:
        print(f"    {key}: {_fmt_optional(positive.get(key))}")

    print("  negative errors:")
    for key in [
        "sample_count",
        "fraction",
        "min_s",
        "mean_s",
        "std_s",
        "max_abs_s",
        "max_abs_position_m",
        "max_abs_cum_time_s",
    ]:
        print(f"    {key}: {_fmt_optional(negative.get(key))}")

    print("  near-zero errors:")
    print(f"    sample_count: {_fmt_optional(near_zero.get('sample_count'))}")
    print(f"    fraction: {_fmt_optional(near_zero.get('fraction'))}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DP redundant-time error against a linearly decreasing "
            "expected redundant-time curve."
        )
    )
    _ = parser.add_argument(
        "--dp-run",
        type=Path,
        required=True,
        help="DP solve run directory containing run.json, profile.npz, quality.json",
    )
    _ = parser.add_argument(
        "--initial-redundant-s",
        type=float,
        help="Initial expected redundant time. Defaults to the first actual sample.",
    )
    _ = parser.add_argument(
        "--zero-eps",
        type=float,
        default=1e-9,
        help="Absolute error tolerance treated as near zero.",
    )
    _ = parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for saving summary statistics as JSON.",
    )
    _ = parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for saving per-sample analysis rows as CSV.",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Optional figure directory; saves {FIGURE_FILENAME}.",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive display window.",
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
    profile = completed_run.payload.profile

    if task.schedule_time_s is None:
        parser.error("Task schedule_time_s must not be None")

    schedule_time_s = float(task.schedule_time_s)
    vehicle = scenario.vehicle
    track = scenario.line
    safeguard = scenario.safeguard
    start_position = float(task.start_position_m)
    target_position = float(task.target_position_m)
    target_speed = 0.0

    min_remaining_time_fn = functools.partial(
        min_operation_time,
        vehicle,
        track,
        safeguard.params.factor,
    )

    try:
        redundant_arr = reconstruct_redundant_operation_time(
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            cum_time_arr=profile.time_s,
            schedule_time_s=schedule_time_s,
            target_position=target_position,
            target_speed=target_speed,
            min_remaining_time_fn=min_remaining_time_fn,
        )
        initial_redundant_s = (
            float(args.initial_redundant_s)
            if args.initial_redundant_s is not None
            else float(redundant_arr[0])
        )
        expected_redundant_arr = compute_expected_redundant_operation_time(
            pos_arr=profile.position_m,
            start_position=start_position,
            target_position=target_position,
            initial_redundant_s=initial_redundant_s,
        )
        error_arr = redundant_arr - expected_redundant_arr
        summary = summarize_error_statistics(
            pos_arr=profile.position_m,
            cum_time_arr=profile.time_s,
            error_arr=error_arr,
            zero_eps=args.zero_eps,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    print(f"Using DP run: {args.dp_run}")
    print("Redundancy source: reconstructed_from_cum_time")
    print(f"Initial expected redundant time: {initial_redundant_s:.6f} s")
    print(f"Schedule time: {schedule_time_s:.6f} s")
    print_error_statistics(summary)

    payload = {
        "run_dir": str(args.dp_run),
        "redundancy_source": "reconstructed_from_cum_time",
        "initial_redundant_s": initial_redundant_s,
        "schedule_time_s": schedule_time_s,
        "start_position_m": start_position,
        "target_position_m": target_position,
        "target_speed_mps": target_speed,
        "zero_eps": float(args.zero_eps),
        "statistics": summary,
    }

    if args.output_json is not None:
        output_json = write_summary_json(args.output_json, payload)
        print(f"Saved JSON summary to {output_json}")

    if args.output_csv is not None:
        output_csv = write_point_csv(
            output_path=args.output_csv,
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            cum_time_arr=profile.time_s,
            actual_redundant_arr=redundant_arr,
            expected_redundant_arr=expected_redundant_arr,
            error_arr=error_arr,
        )
        print(f"Saved per-sample CSV to {output_csv}")

    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Arial",
        title_font_size=9.0,
        axis_label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
    )
    fig = plot_redundancy_error_series(
        pos_arr=profile.position_m,
        actual_redundant_arr=redundant_arr,
        expected_redundant_arr=expected_redundant_arr,
        error_arr=error_arr,
    )

    if args.output_dir is not None:
        output_file = save_compact_figure(fig, args.output_dir)
        print(f"Saved compact figure to {output_file}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
