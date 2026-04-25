import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np

from model.ocs import SafeGuardUtility
from utils.io_utils import load_optimized_curve_and_metrics
from utils.plot_utils import set_chinese_font


def _build_safeguard_utility(factor: float = 0.95) -> SafeGuardUtility:
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        speedlimit_data = json.load(f)
        speed_limits = (
            np.asarray(speedlimit_data["speed_limits"], dtype=np.float64) / 3.6
        )
        speed_limit_intervals = speedlimit_data["intervals"]

    with open("output/safeguardcurves/min_curves_list.pkl", "rb") as f:
        min_curves_list = pickle.load(f)
    with open("output/safeguardcurves/max_curves_list.pkl", "rb") as f:
        max_curves_list = pickle.load(f)

    return SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
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
        "success",
        "total_reward",
        "target_time_s",
        "total_time_s",
        "time_error_s",
        "stop_error_m",
        "total_energy_kj",
        "total_energy_j",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
        "num_timesteps",
        "trigger_mode",
        "trigger_value",
        "episode_steps",
        "terminated",
        "truncated",
        "start_position_m",
        "target_position_m",
        "final_position_m",
        "final_speed_mps",
        "created_at",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and display saved RL best trajectory."
    )
    parser.add_argument(
        "--npz",
        default="output/optimal/rl/best/best_trajectory.npz",
        help="Path to RL best trajectory npz file.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional path to metrics json file. If omitted, infer from npz name.",
    )
    parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="Do not draw safeguard background.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.95,
        help="Safeguard factor used for rendering when safeguard is enabled.",
    )
    args = parser.parse_args()

    pos_arr, speed_arr, metrics = load_optimized_curve_and_metrics(
        npz_path=args.npz,
        metrics_path=args.metrics,
        dtype=np.float32,
        use_metrics_cache=True,
    )

    _print_metrics(metrics)

    set_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 7))

    if not args.no_safeguard:
        safeguard = _build_safeguard_utility(factor=args.factor)
        safeguard.render(ax=ax)

    ax.plot(
        pos_arr,
        speed_arr * 3.6,
        color="darkorange",
        alpha=0.85,
        linewidth=2,
        label="RL best trajectory",
    )

    start_position = _as_float(metrics.get("start_position_m"))
    target_position = _as_float(metrics.get("target_position_m"))
    final_position = _as_float(metrics.get("final_position_m"))
    final_speed_mps = _as_float(metrics.get("final_speed_mps"))

    if start_position is not None:
        ax.scatter(
            start_position,
            0.0,
            marker="o",
            color="green",
            s=80,
            alpha=0.85,
            label="起点",
            zorder=5,
            edgecolors="black",
            linewidths=1.2,
        )
    if target_position is not None:
        ax.scatter(
            target_position,
            0.0,
            marker="o",
            color="red",
            s=80,
            alpha=0.85,
            label="终点",
            zorder=5,
            edgecolors="black",
            linewidths=1.2,
        )
    if final_position is not None and final_speed_mps is not None:
        should_draw_final_point = (
            target_position is None
            or abs(final_position - target_position) > 1e-6
            or abs(final_speed_mps) > 1e-6
        )
        if should_draw_final_point:
            ax.scatter(
                final_position,
                final_speed_mps * 3.6,
                marker="X",
                color="orange",
                s=90,
                alpha=0.9,
                label="终止点",
                zorder=6,
                edgecolors="black",
                linewidths=1.0,
            )

    success_value = metrics.get("success")
    if isinstance(success_value, bool):
        title = (
            "RL 最优轨迹（成功到达）" if success_value else "RL 最优轨迹（未成功到达）"
        )
        ax.set_title(title)

    ax.set_xlabel(r"里程 $s\left( m \right)$")
    ax.set_ylabel(r"速度 $v\left( km/h \right)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
