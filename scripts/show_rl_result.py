import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model.ocs import SafeGuardUtility
from utils.io_utils import load_optimized_curve_and_metrics
from utils.plot_utils import set_global_plot_style
from utils.data_loader import load_speed_limits, load_safeguard_curves

_RL_CURVE_FILENAME = "best_trajectory.npz"
_RL_METRICS_FILENAME = "best_trajectory_metrics.json"
_RL_DEFAULT_SEARCH_DIR = "output/optimal/rl/best"


def _build_safeguard_utility(factor: float = 0.95) -> SafeGuardUtility:
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


def _print_metrics(metrics: dict[str, object]) -> None:
    if not metrics:
        print("No metrics file found.")
        return

    print("Loaded metrics:")
    for key in [
        "total_reward",
        "target_time_s",
        "total_time_s",
        "time_error_s",
        "start_position_m",
        "target_position_m",
        "final_position_m",
        "stop_error_m",
        "total_energy_kj",
        "total_energy_j",
        "final_speed_mps",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
        "episode_steps",
        "success",
        "num_timesteps",
        "eval_trigger_mode",
        "eval_trigger_interval",
        "created_at",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and display saved RL best trajectory."
    )
    parser.add_argument(
        "--curve-dir",
        default=_RL_DEFAULT_SEARCH_DIR,
        help=(
            "Directory to recursively search for both "
            f"{_RL_CURVE_FILENAME} and {_RL_METRICS_FILENAME}."
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
        file_name=_RL_CURVE_FILENAME,
    )

    metrics_path = curve_path.with_name(_RL_METRICS_FILENAME)
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find '{_RL_CURVE_FILENAME}' in the same directory as "
            f"curve file: {curve_path}"
        )

    return str(curve_path), str(metrics_path)


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        npz_path, metrics_path = _resolve_curve_and_metrics_paths(
            curve_dir=args.curve_dir
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
        alpha=0.85,
        linewidth=1.5,
        label="RL best optimized speed curve",
    )

    start_position = _as_float(metrics.get("start_position_m"))
    target_position = _as_float(metrics.get("target_position_m"))
    # final_position = _as_float(metrics.get("final_position_m"))
    # final_speed_mps = _as_float(metrics.get("final_speed_mps"))

    if start_position is not None:
        ax.scatter(
            start_position,
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
    if target_position is not None:
        ax.scatter(
            target_position,
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
    # if final_position is not None and final_speed_mps is not None:
    #     should_draw_final_point = (
    #         target_position is None
    #         or abs(final_position - target_position) > 1e-6
    #         or abs(final_speed_mps) > 1e-6
    #     )
    #     if should_draw_final_point:
    #         ax.scatter(
    #             final_position,
    #             final_speed_mps * 3.6,
    #             marker="X",
    #             color="orange",
    #             s=90,
    #             alpha=0.9,
    #             label="终止点",
    #             zorder=6,
    #             edgecolors="black",
    #             linewidths=1.0,
    #         )

    success_value = metrics.get("success")
    if isinstance(success_value, bool):
        print(
            "RL 最优轨迹（成功到达）" if success_value else "RL 最优轨迹（未成功到达）"
        )

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_xlim((0.0, 30000.0))
    ax.set_ylim((0.0, 500.0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
