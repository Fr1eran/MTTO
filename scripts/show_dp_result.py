import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np

from model.safe_guard_utility import SafeGuardUtility
from utils.io_utils import load_optimized_curve_and_metrics
from utils.plot_utils import set_chinese_font


def _build_safeguard_utility(factor: float = 0.99) -> SafeGuardUtility:
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        speedlimit_data = json.load(f)
        speed_limits = (
            np.asarray(speedlimit_data["speed_limits"], dtype=np.float64) / 3.6
        )
        speed_limit_intervals = speedlimit_data["intervals"]

    with open("data/rail/safeguard/min_curves_list.pkl", "rb") as f:
        min_curves_list = pickle.load(f)
    with open("data/rail/safeguard/max_curves_list.pkl", "rb") as f:
        max_curves_list = pickle.load(f)

    return SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
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
        "total_energy_kj",
        "total_energy_j",
        "start_position_m",
        "target_position_m",
        "created_at",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and display saved DP optimized speed curve."
    )
    parser.add_argument(
        "--npz",
        default="data/optimal/DP/optimized_speed_curve.npz",
        help="Path to optimized curve npz file.",
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
        default=0.99,
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
        color="blue",
        alpha=0.8,
        linewidth=2,
        label="DP optimized speed curve",
    )

    if "start_position_m" in metrics:
        ax.scatter(
            metrics["start_position_m"],
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
    if "target_position_m" in metrics:
        ax.scatter(
            metrics["target_position_m"],
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

    ax.set_xlabel(r"里程 $s\left( m \right)$")
    ax.set_ylabel(r"速度 $v\left( km/h \right)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
