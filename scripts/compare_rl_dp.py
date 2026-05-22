from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    render_dp_curve_on_axes,
    resolve_dp_curve_artifact,
)
from model.common import ECC
from rl.experiment_utils import (
    DEFAULT_SCHEDULE_TIME_S,
    RL_DEFAULT_SEARCH_DIR,
    RL_TRAJECTORY_SOURCE_CHOICES,
    apply_rl_curve_plot_style,
    get_rl_trajectory_status_text,
    load_rl_curve_artifact,
    render_rl_curve_on_axes,
    resolve_rl_curve_artifact,
)
from utils.scenario import build_safeguard_utility, build_scenario
from utils.trajectory import (
    OptimizedCurveArtifact,
    compute_cumulative_energy_from_trajectory,
    compute_segment_accelerations,
)


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _resolve_target_schedule_time(
    *,
    dp_metrics: dict[str, object],
    rl_metrics: dict[str, object],
) -> float:
    rl_target_time_s = _metric_as_float(rl_metrics.get("target_time_s"))
    if rl_target_time_s is not None and rl_target_time_s > 0.0:
        return rl_target_time_s

    dp_target_time_s = _metric_as_float(dp_metrics.get("target_time_s"))
    if dp_target_time_s is not None and dp_target_time_s > 0.0:
        return dp_target_time_s

    return DEFAULT_SCHEDULE_TIME_S


def _compute_segment_midpoints(
    pos_arr: np.ndarray | list[float],
) -> np.ndarray:
    pos = np.asarray(pos_arr, dtype=np.float64)
    if pos.ndim != 1:
        raise ValueError("pos_arr must be a 1-D array")
    if pos.size < 2:
        return np.asarray([], dtype=np.float64)
    return 0.5 * (pos[:-1] + pos[1:])


def _build_ecc() -> ECC:
    return ECC(
        R_m=0.2796,
        L_d=0.0002,
        R_k=50.0,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.8,
    )


def _resolve_curve_artifacts(
    *,
    dp_curve_dir: str,
    rl_curve_dir: str,
    trajectory_source: str,
) -> tuple[OptimizedCurveArtifact, OptimizedCurveArtifact]:
    dp_artifact = resolve_dp_curve_artifact(curve_dir=dp_curve_dir)
    rl_artifact = resolve_rl_curve_artifact(
        curve_dir=rl_curve_dir,
        trajectory_source=trajectory_source,
    )
    return dp_artifact, rl_artifact


def _print_metrics(*, title: str, metrics: dict[str, object], keys: list[str]) -> None:
    print(title)
    if not metrics:
        print("  No metrics file found.")
        return

    for key in keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def _deduplicate_legend(ax: Any, *, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    filtered_handles: list[Any] = []
    filtered_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels, strict=False):
        if not label or label.startswith("_"):
            continue
        if label in seen:
            continue
        seen.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)

    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc=loc)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected DP and RL optimization trajectories with a 3-panel "
            "view (speed, acceleration, cumulative energy)."
        )
    )
    parser.add_argument(
        "--dp-curve-dir",
        default=DP_DEFAULT_SEARCH_DIR,
        help="Directory used to recursively search DP optimized trajectory artifacts.",
    )
    parser.add_argument(
        "--rl-curve-dir",
        default=RL_DEFAULT_SEARCH_DIR,
        help="Directory used to recursively search RL trajectory artifacts.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=RL_TRAJECTORY_SOURCE_CHOICES,
        default="best",
        help="RL trajectory source: best, best_steps, best_episodes, final.",
    )
    parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="Do not render safeguard background on the speed panel.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="Safeguard factor used for rendering when safeguard is enabled.",
    )
    return parser


def _build_rl_curve_label(trajectory_source: str) -> str:
    if trajectory_source == "final":
        return "RL final trajectory"
    return "RL best trajectory"


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        dp_artifact, rl_artifact = _resolve_curve_artifacts(
            dp_curve_dir=args.dp_curve_dir,
            rl_curve_dir=args.rl_curve_dir,
            trajectory_source=args.trajectory_source,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Using DP curve file: {dp_artifact.npz_path}")
    print(f"Using DP metrics file: {dp_artifact.metrics_path}")
    print(f"Using RL curve file: {rl_artifact.npz_path}")
    print(f"Using RL metrics file: {rl_artifact.metrics_path}")

    dp_pos_arr, dp_speed_arr, dp_metrics = load_dp_curve_artifact(dp_artifact)
    rl_pos_arr, rl_speed_arr, rl_metrics = load_rl_curve_artifact(rl_artifact)

    _print_metrics(
        title="DP metrics:",
        metrics=dp_metrics,
        keys=[
            "target_time_s",
            "total_time_s",
            "time_error_s",
            "total_energy_kj",
            "total_energy_j",
            "comfort_tav",
            "comfort_er_pct",
            "comfort_rms",
        ],
    )
    _print_metrics(
        title="RL metrics:",
        metrics=rl_metrics,
        keys=[
            "trajectory_source",
            "success",
            "target_time_s",
            "total_time_s",
            "time_error_s",
            "stop_error_m",
            "total_reward",
            "total_energy_kj",
            "total_energy_j",
            "comfort_tav",
            "comfort_er_pct",
            "comfort_rms",
        ],
    )
    rl_status_text = get_rl_trajectory_status_text(rl_metrics)
    if rl_status_text is not None:
        print(rl_status_text)

    schedule_time_s = _resolve_target_schedule_time(
        dp_metrics=dp_metrics,
        rl_metrics=rl_metrics,
    )
    print(f"Energy context schedule_time_s: {schedule_time_s:.3f}")

    vehicle, track, _, _ = build_scenario(schedule_time_s=schedule_time_s)
    ecc = _build_ecc()

    dp_acc_arr = compute_segment_accelerations(dp_pos_arr, dp_speed_arr)
    rl_acc_arr = compute_segment_accelerations(rl_pos_arr, rl_speed_arr)
    dp_acc_pos_arr = _compute_segment_midpoints(dp_pos_arr)
    rl_acc_pos_arr = _compute_segment_midpoints(rl_pos_arr)

    dp_cum_energy_arr = compute_cumulative_energy_from_trajectory(
        pos_arr=dp_pos_arr,
        speed_arr=dp_speed_arr,
        vehicle=vehicle,
        track=track,
        ecc=ecc,
    )
    rl_cum_energy_arr = compute_cumulative_energy_from_trajectory(
        pos_arr=rl_pos_arr,
        speed_arr=rl_speed_arr,
        vehicle=vehicle,
        track=track,
        ecc=ecc,
    )

    apply_rl_curve_plot_style()
    fig, (ax_speed, ax_acc, ax_energy) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(6, 8),
        sharex=False,
    )

    safeguard = None
    if not args.no_safeguard:
        safeguard = build_safeguard_utility(args.factor)

    render_dp_curve_on_axes(
        ax=ax_speed,
        pos_arr=dp_pos_arr,
        speed_arr=dp_speed_arr,
        metrics=dp_metrics,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color="tab:red",
        curve_label="DP optimized speed curve",
        safeguard=safeguard,
    )
    render_rl_curve_on_axes(
        ax=ax_speed,
        pos_arr=rl_pos_arr,
        speed_arr=rl_speed_arr,
        metrics=rl_metrics,
        no_safeguard=True,
        factor=args.factor,
        curve_color="tab:blue",
        curve_label=_build_rl_curve_label(args.trajectory_source),
        safeguard=safeguard,
    )
    ax_speed.set_title("Optimized trajectory comparison")
    _deduplicate_legend(ax_speed)

    ax_acc.plot(
        dp_acc_pos_arr,
        dp_acc_arr,
        color="tab:red",
        linewidth=1.5,
        alpha=0.9,
        label="DP acceleration",
    )
    ax_acc.plot(
        rl_acc_pos_arr,
        rl_acc_arr,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.9,
        label="RL acceleration",
    )
    ax_acc.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_acc.set_xlabel("Position (m)")
    ax_acc.set_ylabel("Acceleration (m/s^2)")
    ax_acc.set_title("Acceleration profile comparison")
    ax_acc.grid(True, alpha=0.3)
    _deduplicate_legend(ax_acc)

    ax_energy.plot(
        dp_pos_arr,
        dp_cum_energy_arr,
        color="tab:red",
        linewidth=1.5,
        alpha=0.9,
        label="DP cumulative energy",
    )
    ax_energy.plot(
        rl_pos_arr,
        rl_cum_energy_arr,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.9,
        label="RL cumulative energy",
    )
    ax_energy.set_xlabel("Position (m)")
    ax_energy.set_ylabel("Cumulative energy (kJ)")
    ax_energy.set_title("Cumulative energy profile (traction + levitation)")
    ax_energy.grid(True, alpha=0.3)
    _deduplicate_legend(ax_energy)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
