import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    render_dp_curve_on_axes,
)
from utils.plot_utils import set_global_plot_style
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


def _print_metrics(metrics: dict) -> None:
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and display saved DP optimized speed curve."
    )
    parser.add_argument(
        "--curve-dir",
        default=DP_DEFAULT_SEARCH_DIR,
        help="Directory to recursively search for the optimized speed curve.",
    )
    parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="Do not draw safeguard background.",
    )
    parser.add_argument(
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

    pos_arr, speed_arr, _cum_time_arr, metrics = load_dp_curve_artifact(artifact)

    _print_metrics(metrics)

    set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    render_dp_curve_on_axes(
        ax=ax,
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        metrics=metrics,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color="blue",
    )

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
