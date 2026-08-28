import argparse
import functools
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    render_dp_curve_on_axes,
)
from model.common import min_operation_time
from utils.plot_utils import apply_sci_figure_layout, set_global_plot_style
from utils.scenario import build_scenario
from utils.trajectory import OptimizedCurveArtifact


def _resolve_curve_and_metrics_paths(curve_dir: str) -> tuple[str, str]:
    """
    Recursively find the newest optimized_speed_curve.npz
    and its sibling metrics file.
    """
    search_root = Path(curve_dir)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Curve directory does not exist: {curve_dir}")
    matches = sorted(
        (p for p in search_root.rglob("optimized_speed_curve.npz") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, str(p)),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find optimized_speed_curve.npz under: {curve_dir}"
        )
    curve_path = matches[0]
    metrics_path = curve_path.with_name("optimized_speed_curve_metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find optimized_speed_curve_metrics.json in: {curve_path.parent}"
        )
    return str(curve_path), str(metrics_path)


def _print_metrics(metrics: dict[str, object]) -> None:
    if not metrics:
        print("No metrics file found.")
        return

    print("Loaded metrics:")
    for key in [
        "target_time_s",
        "total_time_s",
        "time_error_s",
        "start_position_m",
        "target_position_m",
        "total_energy_kj",
        "total_energy_j",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
        "created_at",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def _metric_as_float(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _require_metric_float(metrics: dict[str, object], key: str) -> float:
    value = _metric_as_float(metrics, key)
    if value is None:
        raise ValueError(f"DP metrics must contain numeric '{key}'")
    return value


def _calc_redundant_operation_time_arr(
    *,
    pos_arr: NDArray[np.float64],
    speed_arr: NDArray[np.float64],
    cum_time_arr: NDArray[np.float64],
    schedule_time_s: float,
    target_position: float,
    target_speed: float,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> np.ndarray:
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
    return float(schedule_time_s) - cum_time - min_remaining_arr


def _build_dp_redundant_operation_time_arr(
    *,
    pos_arr: NDArray[np.float64],
    speed_arr: NDArray[np.float64],
    cum_time_arr: NDArray[np.float64],
    metrics: dict[str, object],
) -> np.ndarray:
    schedule_time_s = _require_metric_float(metrics, "target_time_s")
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )
    min_remaining_time_fn = functools.partial(
        min_operation_time,
        vehicle=vehicle,
        track=track,
        gamma=safeguard_utility.gamma,
    )

    target_position = _metric_as_float(metrics, "target_position_m")
    if target_position is None:
        target_position = train_service.target_position
    target_speed = _metric_as_float(metrics, "target_speed_mps")
    if target_speed is None:
        target_speed = 0.0

    return _calc_redundant_operation_time_arr(
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        cum_time_arr=cum_time_arr,
        schedule_time_s=schedule_time_s,
        target_position=target_position,
        target_speed=target_speed,
        min_remaining_time_fn=min_remaining_time_fn,
    )


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
    _ = ax.grid(True, alpha=0.3)
    _ = ax.legend(loc="best")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and display saved DP optimized speed curve."
    )
    _ = parser.add_argument(
        "--curve-dir",
        default=DP_DEFAULT_SEARCH_DIR,
        help="Directory to recursively search for the optimized speed curve.",
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
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        curve_path, metrics_path = _resolve_curve_and_metrics_paths(
            curve_dir=args.curve_dir
        )
        artifact = OptimizedCurveArtifact(
            npz_path=curve_path, metrics_path=metrics_path
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Using curve file: {artifact.npz_path}")
    print(f"Using metrics file: {artifact.metrics_path}")

    pos_arr, speed_arr, cum_time_arr, metrics = load_dp_curve_artifact(artifact)

    _print_metrics(metrics)
    try:
        redundant_operation_time_arr = _build_dp_redundant_operation_time_arr(
            pos_arr=pos_arr,
            speed_arr=speed_arr,
            cum_time_arr=cum_time_arr,
            metrics=metrics,
        )
    except ValueError as exc:
        parser.error(str(exc))

    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )

    fig, (ax_speed, ax_redundant) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    render_dp_curve_on_axes(
        ax=ax_speed,
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        metrics=metrics,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color="blue",
    )

    ax_speed.legend(loc="upper right")
    _render_redundant_operation_time_on_axes(
        ax=ax_redundant,
        pos_arr=pos_arr,
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

    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=4.4,
        left=0.10,
        bottom=0.12,
        top=0.96,
        hspace=0.22,
    )
    plt.show()


if __name__ == "__main__":
    main()
