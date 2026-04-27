import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model.ocs import SafeGuardUtility
from utils.io_utils import load_optimized_curve_and_metrics
from utils.plot_utils import set_global_plot_style
from utils.data_loader import load_speed_limits, load_safeguard_curves


_DP_CURVE_FILENAME = "optimized_speed_curve.npz"
_DP_METRICS_FILENAME = "optimized_speed_curve_metrics.json"
_DP_DEFAULT_SEARCH_DIR = "output/optimal/dp"


def _build_safeguard_utility(factor: float = 0.99) -> SafeGuardUtility:
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
        load_safeguard_curves(
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        )
    )

    return SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=factor,
    )


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
        default=_DP_DEFAULT_SEARCH_DIR,
        help=(
            "Directory to recursively search for both "
            f"{_DP_CURVE_FILENAME} and {_DP_METRICS_FILENAME}."
        ),
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


def _find_latest_named_file(*, search_dir: str, file_name: str) -> Path:
    search_root = Path(search_dir)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Search directory does not exist: {search_dir}")

    matches = sorted(
        (path for path in search_root.rglob(file_name) if path.is_file()),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{file_name}' under directory: {search_dir}"
        )

    if len(matches) > 1:
        print(
            f"Found {len(matches)} '{file_name}' files under '{search_dir}', "
            f"using latest: {matches[0]}"
        )
    return matches[0]


def _resolve_curve_and_metrics_paths(*, curve_dir: str) -> tuple[str, str]:
    curve_path = _find_latest_named_file(
        search_dir=curve_dir,
        file_name=_DP_CURVE_FILENAME,
    )

    metrics_path = curve_path.with_name(_DP_METRICS_FILENAME)
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find '{_DP_METRICS_FILENAME}' in the same directory as "
            f"curve file: {curve_path}"
        )

    return str(curve_path), str(metrics_path)


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        npz_path, metrics_path = _resolve_curve_and_metrics_paths(
            curve_dir=args.curve_dir,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Using curve file: {npz_path}")
    print(f"Using metrics file: {metrics_path}")

    pos_arr, speed_arr, metrics = load_optimized_curve_and_metrics(
        npz_path=npz_path,
        metrics_path=metrics_path,
        dtype=np.float32,
        use_metrics_cache=True,
    )

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

    if not args.no_safeguard:
        safeguard = _build_safeguard_utility(factor=args.factor)
        safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

    ax.plot(
        pos_arr,
        speed_arr * 3.6,
        color="blue",
        alpha=0.8,
        linewidth=1.5,
        label="DP optimized speed curve",
    )

    if "start_position_m" in metrics:
        ax.scatter(
            metrics["start_position_m"],
            0.0,
            marker="o",
            color="green",
            s=40,
            alpha=0.85,
            label="start",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
    if "target_position_m" in metrics:
        ax.scatter(
            metrics["target_position_m"],
            0.0,
            marker="o",
            color="red",
            s=40,
            alpha=0.85,
            label="end",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_ylim((0, 500.0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
