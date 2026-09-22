from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    render_dp_curve_on_axes,
    resolve_dp_curve_artifact,
)
from model.common import ECC
from rl.experiment_utils import (
    load_rl_curve_artifact,
    render_rl_curve_on_axes,
    resolve_rl_curve_artifact,
)
from utils.plot_utils import (
    VIS_ACTUAL_PURPLE,
    VIS_DP_BLACK,
    VIS_PROPOSED_ORANGE,
    add_panel_label,
    apply_sci_curve_style,
    apply_sci_figure_layout,
    apply_sci_grid,
    save_sci_figure,
)
from utils.scenario import build_safeguard_utility, build_scenario
from utils.trajectory import (
    OptimizedCurveArtifact,
    compute_comfort_metrics_from_trajectory,
    compute_cumulative_energy_from_trajectory,
    compute_segment_accelerations,
    recover_time_axis_from_trajectory,
)
from utils.type_utils import as_1d_float_array, as_float

FIGURE_FILENAME = "dp_rl_actual_comparison.pdf"
DEFAULT_REAL_CURVE_PATH = "output/real_operation/aligned_real_operation_curve.npz"
_REAL_CURVE_REQUIRED_KEYS = ("position_m", "speed_mps", "time_s", "target_position_m")
_TARGET_TIME_TOLERANCE_S = 1e-6
_TARGET_POSITION_TOLERANCE_M = 1e-3
_TRAJECTORY_COLORS = (VIS_DP_BLACK, VIS_PROPOSED_ORANGE, VIS_ACTUAL_PURPLE)
_TRAJECTORY_LINESTYLES = ("-", "--", "-.")
_TRAJECTORY_LEGEND_LABELS = (
    "DP optimization",
    "Proposed Method",
    "Actual operation",
)


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
    total_energy_kwh: float
    comfort_tav: float | None = None

    @property
    def total_energy_kj(self) -> float:
        return self.total_energy_kwh * 3600.0


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
        L_d=0.00292,
        R_k=0.0736,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.5,
    )


def _resolve_curve_artifacts(
    *,
    dp_curve_dir: str,
    rl_model_dir: str,
) -> tuple[OptimizedCurveArtifact, OptimizedCurveArtifact]:
    dp_artifact = resolve_dp_curve_artifact(curve_dir=dp_curve_dir)
    rl_artifact = resolve_rl_curve_artifact(
        curve_dir=rl_model_dir,
    )
    return dp_artifact, rl_artifact


def _resolve_target_schedule_time(
    *, dp_metrics: dict[str, object], rl_metrics: dict[str, object]
) -> float:
    dp_target_time_s = as_float(dp_metrics.get("target_time_s"))
    rl_target_time_s = as_float(rl_metrics.get("target_time_s"))
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


def _resolve_target_position(
    *, metrics: dict[str, object], position_m: np.ndarray, source_name: str
) -> float:
    target_position_m = as_float(metrics.get("target_position_m"))
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
        position_m = as_1d_float_array(
            curve_data["position_m"], "position_m", min_length=2, check_finite=True
        )
        speed_mps = as_1d_float_array(
            curve_data["speed_mps"], "speed_mps", min_length=2, check_finite=True
        )
        time_s = as_1d_float_array(
            curve_data["time_s"], "time_s", min_length=2, check_finite=True
        )
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
            "station: " + ", ".join(mismatched)
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
    total_energy_kwh = float(cumulative_energy_kj[-1]) / 3600.0
    if profile.label == "Actual operation":
        comfort_tav = None
    else:
        comfort_metrics = compute_comfort_metrics_from_trajectory(
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            max_acc_change=max_acc_change,
        )
        comfort_tav = float(comfort_metrics["comfort_tav"])

    total_time_s = float(profile.time_s[-1] - profile.time_s[0])
    return ProfileMetrics(
        time_error_s=abs(total_time_s - target_schedule_time_s),
        stop_error_m=abs(float(profile.position_m[-1]) - profile.target_position_m),
        total_energy_kwh=total_energy_kwh,
        comfort_tav=comfort_tav,
    )


def format_comparison_table(
    profile_metrics: list[tuple[str, ProfileMetrics]],
) -> str:
    def _fmt_tav(m: ProfileMetrics) -> str:
        return f"{m.comfort_tav:.6f}" if m.comfort_tav is not None else "—"

    rows: list[tuple[str, Any]] = [
        ("Time error (s)", lambda m: f"{m.time_error_s:.3f}"),
        ("Stop error (m)", lambda m: f"{m.stop_error_m:.3f}"),
        ("Total energy (kWh)", lambda m: f"{m.total_energy_kwh:.3f}"),
        ("Cumulative acceleration variation (m/s²)", _fmt_tav),
    ]
    headers = ["Metric", *(label for label, _ in profile_metrics)]
    values = [
        [
            title,
            *(formatter(metrics) for _, metrics in profile_metrics),
        ]
        for title, formatter in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in values))
        for column in range(len(headers))
    ]

    def render_row(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    rendered_rows = [
        render_row(headers),
        separator,
        *(render_row(row) for row in values),
    ]
    note = (
        "\n*Note: Actual operation acceleration calculation differs "
        "(estimated by differencing discrete operational data) and is not "
        "directly comparable with DP/RL. The cumulative acceleration variation "
        r"formula is $\sum_t |a_t - a_{t-1}|$, with unit $\mathrm{m/s^2}$.*"
    )
    return "\n".join(rendered_rows) + note


def _finalize_comparison_figure(
    figure: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
) -> None:
    """Apply the paper layout and a trajectory-only shared legend."""
    for axis in axes:
        axis.set_title("")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.8)
        for color, linestyle in zip(
            _TRAJECTORY_COLORS, _TRAJECTORY_LINESTYLES, strict=True
        )
    ]
    figure.legend(
        handles,
        _TRAJECTORY_LEGEND_LABELS,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0.0,
    )
    apply_sci_figure_layout(
        figure,
        columns=2,
        height_in=5.5,
        left=0.11,
        right=0.96,
        bottom=0.10,
        top=0.92,
        hspace=0.30,
    )


def _create_comparison_axes() -> tuple[
    plt.Figure,
    tuple[plt.Axes, plt.Axes, plt.Axes],
]:
    figure, axes = plt.subplots(3, 1, sharex=True)
    return figure, (axes[0], axes[1], axes[2])


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DP, RL, and actual operation speed profiles with speed, "
            "acceleration, cumulative-energy plots and a terminal metric table."
        )
    )
    parser.add_argument("--dp-curve-dir", default=DP_DEFAULT_SEARCH_DIR)
    parser.add_argument("--rl-model-dir", required=True)
    parser.add_argument(
        "--real-curve",
        default=DEFAULT_REAL_CURVE_PATH,
        help="Aligned actual curve NPZ path.",
    )
    parser.add_argument("--no-safeguard", action="store_true")
    parser.add_argument("--factor", type=float, default=0.99)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Optional figure directory; saves {FIGURE_FILENAME}.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the interactive plot window.",
    )
    return parser


def _build_rl_curve_label() -> str:
    return "Proposed Method speed curve"


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    try:
        dp_artifact, rl_artifact = _resolve_curve_artifacts(
            dp_curve_dir=args.dp_curve_dir,
            rl_model_dir=args.rl_model_dir,
        )
        dp_pos, dp_speed, dp_time, dp_metadata = load_dp_curve_artifact(dp_artifact)
        rl_pos, rl_speed, rl_metadata = load_rl_curve_artifact(rl_artifact)
        target_time_s = _resolve_target_schedule_time(
            dp_metrics=dp_metadata, rl_metrics=rl_metadata
        )
        dp_profile = SpeedProfile(
            label="DP optimization",
            position_m=as_1d_float_array(
                dp_pos, "DP position", min_length=2, check_finite=True
            ),
            speed_mps=as_1d_float_array(
                dp_speed, "DP speed", min_length=2, check_finite=True
            ),
            time_s=as_1d_float_array(
                dp_time, "DP cumulative time", min_length=2, check_finite=True
            ),
            target_position_m=_resolve_target_position(
                metrics=dp_metadata, position_m=dp_pos, source_name="DP"
            ),
        )
        rl_position_m = as_1d_float_array(
            rl_pos, "RL position", min_length=2, check_finite=True
        )
        rl_speed_mps = as_1d_float_array(
            rl_speed, "RL speed", min_length=2, check_finite=True
        )
        rl_profile = SpeedProfile(
            label="Proposed Method",
            position_m=rl_position_m,
            speed_mps=rl_speed_mps,
            time_s=recover_time_axis_from_trajectory(
                rl_position_m,
                rl_speed_mps,
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

    vehicle, track, _, train_service = build_scenario(schedule_time_s=target_time_s)
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

    apply_sci_curve_style()
    fig, (ax_speed, ax_acc, ax_energy) = _create_comparison_axes()
    safeguard = None if args.no_safeguard else build_safeguard_utility(args.factor)
    render_dp_curve_on_axes(
        ax=ax_speed,
        pos_arr=dp_profile.position_m,
        speed_arr=dp_profile.speed_mps,
        metrics=dp_metadata,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color=_TRAJECTORY_COLORS[0],
        curve_label="DP optimized speed curve",
        safeguard=safeguard,
        render_endpoints=False,
    )
    render_rl_curve_on_axes(
        ax=ax_speed,
        pos_arr=rl_profile.position_m,
        speed_arr=rl_profile.speed_mps,
        metrics=rl_metadata,
        no_safeguard=True,
        factor=args.factor,
        curve_color=_TRAJECTORY_COLORS[1],
        curve_label=_build_rl_curve_label(),
        safeguard=safeguard,
        render_endpoints=False,
    )
    for line, linestyle in zip(
        (
            next(
                line
                for line in ax_speed.lines
                if line.get_label() == "DP optimized speed curve"
            ),
            next(
                line
                for line in ax_speed.lines
                if line.get_label() == _build_rl_curve_label()
            ),
        ),
        _TRAJECTORY_LINESTYLES[:2],
        strict=True,
    ):
        line.set_linestyle(linestyle)
    ax_speed.plot(
        real_profile.position_m,
        real_profile.speed_mps * 3.6,
        color=_TRAJECTORY_COLORS[2],
        linestyle=_TRAJECTORY_LINESTYLES[2],
        linewidth=1.5,
        label="Actual operation speed curve",
    )
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_xlabel("")
    add_panel_label(ax_speed, "(a)")

    acc_profiles = [p for p in profiles if p.label != "Actual operation"]
    for profile, color, linestyle in zip(
        acc_profiles, _TRAJECTORY_COLORS, _TRAJECTORY_LINESTYLES, strict=False
    ):
        ax_acc.plot(
            _compute_segment_midpoints(profile.position_m),
            compute_segment_accelerations(profile.position_m, profile.speed_mps),
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{profile.label} acceleration",
        )
    ax_acc.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax_acc.set_xlabel("")
    ax_acc.set_ylabel(r"Acceleration ($\mathrm{m/s^2}$)")
    ax_acc.set_ylim(-1.5, 1.6)
    add_panel_label(ax_acc, "(b)")
    apply_sci_grid(ax_acc)

    for profile, color, linestyle in zip(
        profiles, _TRAJECTORY_COLORS, _TRAJECTORY_LINESTYLES, strict=True
    ):
        cumulative_energy = compute_cumulative_energy_from_trajectory(
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            vehicle=vehicle,
            track=track,
            ecc=ecc,
        )
        ax_energy.plot(
            profile.position_m,
            cumulative_energy / 3600.0,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{profile.label} cumulative energy",
        )
    ax_energy.set_xlabel("Position (m)")
    ax_energy.set_ylabel("Energy (kWh)")
    add_panel_label(ax_energy, "(c)")
    apply_sci_grid(ax_energy)
    _finalize_comparison_figure(fig, (ax_speed, ax_acc, ax_energy))

    if args.output_dir is not None:
        saved_path = save_sci_figure(fig, args.output_dir / FIGURE_FILENAME)
        print(f"Saved comparison figure to: {saved_path}")
        table_path = args.output_dir / "dp_rl_actual_comparison_table.md"
        table_path.write_text(
            "# Trajectory Comparison Table\n\n"
            + format_comparison_table(metrics_by_label)
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved comparison table to: {table_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
