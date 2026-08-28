from __future__ import annotations

import argparse
import csv
import functools
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    resolve_dp_curve_artifact,
)
from model.common import min_operation_time
from utils.plot_utils import (
    SCI_EXPORT_PAD_INCHES,
    apply_sci_figure_layout,
    save_sci_figure,
    set_global_plot_style,
)
from utils.scenario import build_scenario

DEFAULT_REDUNDANT_ARRAY_KEYS: tuple[str, ...] = (
    "redundant_operation_time_s",
    "redundant_operation_time",
    "ref_redundant_operation_time_s",
    "ref_redundant_operation_time",
    "redundant_time_s",
)


@dataclass(frozen=True)
class RedundancySeries:
    values: NDArray[np.float64]
    source: str
    key: str | None


def _as_1d_float_array(
    values: Sequence[float] | NDArray[Any], name: str
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_same_length(
    named_arrays: Sequence[tuple[str, NDArray[np.float64]]],
) -> None:
    sizes = {arr.size for _name, arr in named_arrays}
    if len(sizes) != 1:
        names = ", ".join(name for name, _arr in named_arrays)
        raise ValueError(f"{names} must have the same length")


def _metric_as_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _require_metric_float(metrics: dict[str, Any], key: str) -> float:
    value = _metric_as_float(metrics, key)
    if value is None:
        raise ValueError(f"DP metrics must contain numeric '{key}'")
    return value


def load_redundant_operation_time_from_npz(
    npz_path: str | Path,
    *,
    expected_size: int,
    redundant_key: str | None = None,
) -> RedundancySeries | None:
    """Load a redundant-time array from a DP NPZ when one is present."""
    with np.load(npz_path, allow_pickle=False) as npz_data:
        available_keys = tuple(npz_data.files)
        if redundant_key is not None:
            if redundant_key not in npz_data.files:
                raise ValueError(
                    "DP curve does not contain redundant array key "
                    + f"{redundant_key!r}. Available keys: {available_keys!r}"
                )
            selected_key = redundant_key
        else:
            selected_key = next(
                (key for key in DEFAULT_REDUNDANT_ARRAY_KEYS if key in npz_data.files),
                None,
            )
            if selected_key is None:
                return None

        values = _as_1d_float_array(npz_data[selected_key], selected_key)

    if values.size != int(expected_size):
        raise ValueError(
            f"Redundant array '{selected_key}' length {values.size} does not match "
            + f"trajectory length {int(expected_size)}"
        )
    return RedundancySeries(
        values=values,
        source=f"npz:{selected_key}",
        key=selected_key,
    )


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
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    speed = _as_1d_float_array(speed_arr, "speed_arr")
    cum_time = _as_1d_float_array(cum_time_arr, "cum_time_arr")
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


def load_or_reconstruct_redundant_operation_time(
    *,
    npz_path: str | Path,
    pos_arr: Sequence[float] | NDArray[Any],
    speed_arr: Sequence[float] | NDArray[Any],
    cum_time_arr: Sequence[float] | NDArray[Any],
    schedule_time_s: float,
    target_position: float,
    target_speed: float,
    redundant_key: str | None,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> RedundancySeries:
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    loaded = load_redundant_operation_time_from_npz(
        npz_path,
        expected_size=pos.size,
        redundant_key=redundant_key,
    )
    if loaded is not None:
        return loaded

    reconstructed = reconstruct_redundant_operation_time(
        pos_arr=pos,
        speed_arr=speed_arr,
        cum_time_arr=cum_time_arr,
        schedule_time_s=schedule_time_s,
        target_position=target_position,
        target_speed=target_speed,
        min_remaining_time_fn=min_remaining_time_fn,
    )
    return RedundancySeries(
        values=reconstructed,
        source="reconstructed_from_cum_time",
        key=None,
    )


def compute_expected_redundant_operation_time(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    start_position: float,
    target_position: float,
    initial_redundant_s: float,
) -> NDArray[np.float64]:
    pos = _as_1d_float_array(pos_arr, "pos_arr")
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
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    cum_time = _as_1d_float_array(cum_time_arr, "cum_time_arr")
    error = _as_1d_float_array(error_arr, "error_arr")
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
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    actual = _as_1d_float_array(actual_redundant_arr, "actual_redundant_arr")
    expected = _as_1d_float_array(expected_redundant_arr, "expected_redundant_arr")
    error = _as_1d_float_array(error_arr, "error_arr")
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
    ax_redundant.grid(True, alpha=0.3)
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
    ax_error.grid(True, alpha=0.3)
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
    output_file: Path,
    dpi: float,
    pad_inches: float,
) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return save_sci_figure(fig, output_file, dpi=dpi, pad_inches=pad_inches)


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
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    speed = _as_1d_float_array(speed_arr, "speed_arr")
    cum_time = _as_1d_float_array(cum_time_arr, "cum_time_arr")
    actual = _as_1d_float_array(actual_redundant_arr, "actual_redundant_arr")
    expected = _as_1d_float_array(expected_redundant_arr, "expected_redundant_arr")
    error = _as_1d_float_array(error_arr, "error_arr")
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
        "--curve-dir",
        default=DP_DEFAULT_SEARCH_DIR,
        help="Directory to recursively search for the optimized DP curve.",
    )
    _ = parser.add_argument(
        "--redundant-key",
        help="NPZ key to read as the redundant-time array.",
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
        "--output-file",
        type=Path,
        help="Optional path for saving the figure. A .png suffix is added if omitted.",
    )
    _ = parser.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="DPI used when saving the figure.",
    )
    _ = parser.add_argument(
        "--pad-inches",
        type=float,
        default=SCI_EXPORT_PAD_INCHES,
        help="Padding around the tight saved figure.",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive display window.",
    )
    return parser


def _resolve_target_context(
    metrics: dict[str, Any],
    *,
    train_service: Any,
) -> tuple[float, float, float]:
    start_position = _metric_as_float(metrics, "start_position_m")
    if start_position is None:
        start_position = float(train_service.start_position)
    target_position = _metric_as_float(metrics, "target_position_m")
    if target_position is None:
        target_position = float(train_service.target_position)
    target_speed = _metric_as_float(metrics, "target_speed_mps")
    if target_speed is None:
        target_speed = 0.0
    return start_position, target_position, target_speed


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        artifact = resolve_dp_curve_artifact(curve_dir=args.curve_dir)
        pos_arr, speed_arr, cum_time_arr, metrics = load_dp_curve_artifact(artifact)
        schedule_time_s = _require_metric_float(metrics, "target_time_s")
        vehicle, track, safeguard_utility, train_service = build_scenario(
            schedule_time_s=schedule_time_s
        )
        start_position, target_position, target_speed = _resolve_target_context(
            metrics,
            train_service=train_service,
        )
        min_remaining_time_fn = functools.partial(
            min_operation_time,
            vehicle=vehicle,
            track=track,
            factor=safeguard_utility.gamma,
        )
        redundancy_series = load_or_reconstruct_redundant_operation_time(
            npz_path=artifact.npz_path,
            pos_arr=pos_arr,
            speed_arr=speed_arr,
            cum_time_arr=cum_time_arr,
            schedule_time_s=schedule_time_s,
            target_position=target_position,
            target_speed=target_speed,
            redundant_key=args.redundant_key,
            min_remaining_time_fn=min_remaining_time_fn,
        )
        initial_redundant_s = (
            float(args.initial_redundant_s)
            if args.initial_redundant_s is not None
            else float(redundancy_series.values[0])
        )
        expected_redundant_arr = compute_expected_redundant_operation_time(
            pos_arr=pos_arr,
            start_position=start_position,
            target_position=target_position,
            initial_redundant_s=initial_redundant_s,
        )
        error_arr = redundancy_series.values - expected_redundant_arr
        summary = summarize_error_statistics(
            pos_arr=pos_arr,
            cum_time_arr=cum_time_arr,
            error_arr=error_arr,
            zero_eps=args.zero_eps,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Using DP curve file: {artifact.npz_path}")
    print(f"Using DP metrics file: {artifact.metrics_path}")
    print(f"Redundancy source: {redundancy_series.source}")
    print(f"Initial expected redundant time: {initial_redundant_s:.6f} s")
    print(f"Schedule time: {schedule_time_s:.6f} s")
    print_error_statistics(summary)

    payload = {
        "curve_file": artifact.npz_path,
        "metrics_file": artifact.metrics_path,
        "redundancy_source": redundancy_series.source,
        "redundant_key": redundancy_series.key,
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
            pos_arr=pos_arr,
            speed_arr=speed_arr,
            cum_time_arr=cum_time_arr,
            actual_redundant_arr=redundancy_series.values,
            expected_redundant_arr=expected_redundant_arr,
            error_arr=error_arr,
        )
        print(f"Saved per-sample CSV to {output_csv}")

    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=9.0,
        axis_label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )
    fig = plot_redundancy_error_series(
        pos_arr=pos_arr,
        actual_redundant_arr=redundancy_series.values,
        expected_redundant_arr=expected_redundant_arr,
        error_arr=error_arr,
    )

    if args.output_file is not None:
        output_file = save_compact_figure(
            fig,
            args.output_file,
            args.dpi,
            args.pad_inches,
        )
        print(f"Saved compact figure to {output_file}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
