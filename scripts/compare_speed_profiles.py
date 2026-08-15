from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
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
    RL_DEFAULT_SEARCH_DIR,
    RL_TRAJECTORY_SOURCE_CHOICES,
    apply_rl_curve_plot_style,
    load_rl_curve_artifact,
    render_rl_curve_on_axes,
    resolve_rl_curve_artifact,
)
from utils.scenario import build_safeguard_utility, build_scenario
from utils.trajectory import (
    OptimizedCurveArtifact,
    compute_comfort_metrics_from_trajectory,
    compute_cumulative_energy_from_trajectory,
    compute_segment_accelerations,
)

DEFAULT_REAL_CURVE_PATH = "output/real_operation/aligned_real_operation_curve.npz"
_REAL_CURVE_REQUIRED_KEYS = ("position_m", "speed_mps", "time_s", "target_position_m")
_TARGET_TIME_TOLERANCE_S = 1e-6
_TARGET_POSITION_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class SpeedProfile:
    label: str
    position_m: np.ndarray
    speed_mps: np.ndarray
    time_s: np.ndarray
    target_position_m: float


@dataclass(frozen=True)
class ProfileMetrics:
    time_error_s: float
    stop_error_m: float
    total_energy_kj: float
    comfort_tav: float


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _compute_segment_midpoints(pos_arr: np.ndarray | list[float]) -> np.ndarray:
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


def _resolve_target_schedule_time(
    *, dp_metrics: dict[str, object], rl_metrics: dict[str, object]
) -> float:
    dp_target_time_s = _metric_as_float(dp_metrics.get("target_time_s"))
    rl_target_time_s = _metric_as_float(rl_metrics.get("target_time_s"))
    if dp_target_time_s is not None and dp_target_time_s <= 0.0:
        raise ValueError("DP target_time_s must be positive")
    if rl_target_time_s is not None and rl_target_time_s <= 0.0:
        raise ValueError("RL target_time_s must be positive")
    if dp_target_time_s is not None and rl_target_time_s is not None:
        if abs(dp_target_time_s - rl_target_time_s) > _TARGET_TIME_TOLERANCE_S:
            raise ValueError(
                "DP and RL target_time_s differ; select artifacts from the same task."
            )
        return dp_target_time_s
    if rl_target_time_s is not None:
        return rl_target_time_s
    if dp_target_time_s is not None:
        return dp_target_time_s
    raise ValueError(
        "Both DP and RL metrics are missing target_time_s; "
        "cannot compute a common time error."
    )


def _as_valid_trajectory_array(name: str, values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a 1-D array with at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _build_uniform_time_array(*, total_time_s: float, sample_count: int) -> np.ndarray:
    if total_time_s <= 0.0:
        raise ValueError("total_time_s must be positive")
    return np.linspace(0.0, total_time_s, sample_count, dtype=np.float64)


def _resolve_target_position(
    *, metrics: dict[str, object], position_m: np.ndarray, source_name: str
) -> float:
    target_position_m = _metric_as_float(metrics.get("target_position_m"))
    if target_position_m is None:
        raise ValueError(f"{source_name} metrics are missing target_position_m")
    if not np.isfinite(target_position_m):
        raise ValueError(f"{source_name} target_position_m must be finite")
    return target_position_m


def load_real_operation_profile(curve_path: str | Path) -> SpeedProfile:
    path = Path(curve_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Real operation curve does not exist: {path}. "
            "Run 'python -m scripts.transform_real_operation_curve' first, "
            "or provide --real-curve."
        )
    with np.load(path, allow_pickle=False) as curve_data:
        missing_keys = [
            key for key in _REAL_CURVE_REQUIRED_KEYS if key not in curve_data
        ]
        if missing_keys:
            raise ValueError(
                "Real operation curve is missing required arrays: "
                + ", ".join(missing_keys)
            )
        position_m = _as_valid_trajectory_array("position_m", curve_data["position_m"])
        speed_mps = _as_valid_trajectory_array("speed_mps", curve_data["speed_mps"])
        time_s = _as_valid_trajectory_array("time_s", curve_data["time_s"])
        target_values = np.asarray(curve_data["target_position_m"], dtype=np.float64)

    if not (position_m.size == speed_mps.size == time_s.size):
        raise ValueError(
            "Real operation position_m, speed_mps, and time_s must match length"
        )
    if np.any(np.diff(position_m) < 0.0):
        raise ValueError("Real operation position_m must be non-decreasing")
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError("Real operation time_s must be non-decreasing")
    if target_values.size != 1 or not np.isfinite(float(target_values.reshape(-1)[0])):
        raise ValueError("Real operation target_position_m must be one finite scalar")

    return SpeedProfile(
        label="Actual operation",
        position_m=position_m,
        speed_mps=speed_mps,
        time_s=time_s,
        target_position_m=float(target_values.reshape(-1)[0]),
    )


def _validate_common_target_position(profiles: list[SpeedProfile]) -> float:
    target_position_m = profiles[0].target_position_m
    mismatched = [
        profile.label
        for profile in profiles[1:]
        if abs(profile.target_position_m - target_position_m)
        > _TARGET_POSITION_TOLERANCE_M
    ]
    if mismatched:
        raise ValueError(
            "Trajectory target positions differ; select curves aligned to the same "
            "station: "
            + ", ".join(mismatched)
        )
    return target_position_m


def compute_profile_metrics(
    *,
    profile: SpeedProfile,
    target_schedule_time_s: float,
    vehicle: Any,
    track: Any,
    ecc: ECC,
    max_acc_change: float,
) -> ProfileMetrics:
    cumulative_energy_kj = compute_cumulative_energy_from_trajectory(
        pos_arr=profile.position_m,
        speed_arr=profile.speed_mps,
        vehicle=vehicle,
        track=track,
        ecc=ecc,
    )
    comfort_metrics = compute_comfort_metrics_from_trajectory(
        pos_arr=profile.position_m,
        speed_arr=profile.speed_mps,
        max_acc_change=max_acc_change,
    )
    total_time_s = float(profile.time_s[-1] - profile.time_s[0])
    return ProfileMetrics(
        time_error_s=abs(total_time_s - target_schedule_time_s),
        stop_error_m=abs(float(profile.position_m[-1]) - profile.target_position_m),
        total_energy_kj=float(cumulative_energy_kj[-1]),
        comfort_tav=float(comfort_metrics["comfort_tav"]),
    )


def format_comparison_table(
    profile_metrics: list[tuple[str, ProfileMetrics]],
) -> str:
    rows = [
        ("Time error (s)", "time_error_s", ".3f"),
        ("Stop error (m)", "stop_error_m", ".3f"),
        ("Total energy (kJ)", "total_energy_kj", ".3f"),
        ("comfort_tav (m/s^2)", "comfort_tav", ".6f"),
    ]
    headers = ["Metric", *(label for label, _ in profile_metrics)]
    values = [
        [
            title,
            *(format(getattr(metrics, attr), spec) for _, metrics in profile_metrics),
        ]
        for title, attr, spec in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in values))
        for column in range(len(headers))
    ]

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    rendered_rows = [
        render_row(headers),
        separator,
        *(render_row(row) for row in values),
    ]
    return "\n".join(rendered_rows)


def _deduplicate_legend(ax: Any, *, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=False)
        if label
        and not label.startswith("_")
        and not (label in seen or seen.add(label))
    ]
    if filtered:
        ax.legend(*zip(*filtered, strict=False), loc=loc)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DP, RL, and actual operation speed profiles with speed, "
            "acceleration, cumulative-energy plots and a terminal metric table."
        )
    )
    parser.add_argument("--dp-curve-dir", default=DP_DEFAULT_SEARCH_DIR)
    parser.add_argument("--rl-curve-dir", default=RL_DEFAULT_SEARCH_DIR)
    parser.add_argument(
        "--real-curve",
        default=DEFAULT_REAL_CURVE_PATH,
        help="Aligned actual curve NPZ path.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=RL_TRAJECTORY_SOURCE_CHOICES,
        default="best",
    )
    parser.add_argument("--no-safeguard", action="store_true")
    parser.add_argument("--factor", type=float, default=0.99)
    return parser


def _build_rl_curve_label(trajectory_source: str) -> str:
    return (
        "RL final trajectory"
        if trajectory_source == "final"
        else "RL best trajectory"
    )


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    try:
        dp_artifact, rl_artifact = _resolve_curve_artifacts(
            dp_curve_dir=args.dp_curve_dir,
            rl_curve_dir=args.rl_curve_dir,
            trajectory_source=args.trajectory_source,
        )
        dp_pos, dp_speed, dp_time, dp_metadata = load_dp_curve_artifact(dp_artifact)
        rl_pos, rl_speed, rl_metadata = load_rl_curve_artifact(rl_artifact)
        target_time_s = _resolve_target_schedule_time(
            dp_metrics=dp_metadata, rl_metrics=rl_metadata
        )
        dp_profile = SpeedProfile(
            label="DP optimization",
            position_m=_as_valid_trajectory_array("DP position", dp_pos),
            speed_mps=_as_valid_trajectory_array("DP speed", dp_speed),
            time_s=_as_valid_trajectory_array("DP cumulative time", dp_time),
            target_position_m=_resolve_target_position(
                metrics=dp_metadata, position_m=dp_pos, source_name="DP"
            ),
        )
        rl_total_time_s = _metric_as_float(rl_metadata.get("total_time_s"))
        if rl_total_time_s is None:
            raise ValueError("RL metrics are missing total_time_s")
        rl_profile = SpeedProfile(
            label="Proposed RL",
            position_m=_as_valid_trajectory_array("RL position", rl_pos),
            speed_mps=_as_valid_trajectory_array("RL speed", rl_speed),
            time_s=_build_uniform_time_array(
                total_time_s=rl_total_time_s, sample_count=len(rl_pos)
            ),
            target_position_m=_resolve_target_position(
                metrics=rl_metadata, position_m=rl_pos, source_name="RL"
            ),
        )
        real_profile = load_real_operation_profile(args.real_curve)
        profiles = [dp_profile, rl_profile, real_profile]
        _ = _validate_common_target_position(profiles)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    vehicle, track, train_service, _ = build_scenario(schedule_time_s=target_time_s)
    ecc = _build_ecc()
    metrics_by_label = [
        (
            profile.label,
            compute_profile_metrics(
                profile=profile,
                target_schedule_time_s=target_time_s,
                vehicle=vehicle,
                track=track,
                ecc=ecc,
                max_acc_change=train_service.max_acc_change,
            ),
        )
        for profile in profiles
    ]

    print(f"DP curve: {dp_artifact.npz_path}")
    print(f"RL curve: {rl_artifact.npz_path}")
    print(f"Actual operation curve: {args.real_curve}")
    print(f"Common target running time: {target_time_s:.3f} s")
    print("\nTrajectory comparison metrics:")
    print(format_comparison_table(metrics_by_label))

    apply_rl_curve_plot_style()
    _, (ax_speed, ax_acc, ax_energy) = plt.subplots(3, 1, figsize=(6, 8))
    safeguard = None if args.no_safeguard else build_safeguard_utility(args.factor)
    render_dp_curve_on_axes(
        ax=ax_speed,
        pos_arr=dp_profile.position_m,
        speed_arr=dp_profile.speed_mps,
        metrics=dp_metadata,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color="tab:red",
        curve_label="DP optimized speed curve",
        safeguard=safeguard,
    )
    render_rl_curve_on_axes(
        ax=ax_speed,
        pos_arr=rl_profile.position_m,
        speed_arr=rl_profile.speed_mps,
        metrics=rl_metadata,
        no_safeguard=True,
        factor=args.factor,
        curve_color="tab:blue",
        curve_label=_build_rl_curve_label(args.trajectory_source),
        safeguard=safeguard,
    )
    ax_speed.plot(
        real_profile.position_m,
        real_profile.speed_mps * 3.6,
        color="tab:green",
        linewidth=1.5,
        label="Actual operation speed curve",
    )
    ax_speed.set_title("Speed-position profile comparison")
    _deduplicate_legend(ax_speed)

    colors = ("tab:red", "tab:blue", "tab:green")
    for profile, color in zip(profiles, colors, strict=True):
        ax_acc.plot(
            _compute_segment_midpoints(profile.position_m),
            compute_segment_accelerations(profile.position_m, profile.speed_mps),
            color=color,
            linewidth=1.5,
            label=f"{profile.label} acceleration",
        )
    ax_acc.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_acc.set_xlabel("Position (m)")
    ax_acc.set_ylabel("Acceleration (m/s^2)")
    ax_acc.set_title("Acceleration-position profile comparison")
    ax_acc.grid(True, alpha=0.3)
    _deduplicate_legend(ax_acc)

    for profile, color in zip(profiles, colors, strict=True):
        cumulative_energy = compute_cumulative_energy_from_trajectory(
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            vehicle=vehicle,
            track=track,
            ecc=ecc,
        )
        ax_energy.plot(
            profile.position_m,
            cumulative_energy,
            color=color,
            linewidth=1.5,
            label=f"{profile.label} cumulative energy",
        )
    ax_energy.set_xlabel("Position (m)")
    ax_energy.set_ylabel("Cumulative energy (kJ)")
    ax_energy.set_title("Cumulative energy-position profile comparison")
    ax_energy.grid(True, alpha=0.3)
    _deduplicate_legend(ax_energy)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
