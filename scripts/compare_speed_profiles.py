from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    render_dp_curve_on_axes,
    resolve_dp_curve_artifact,
)
from model.common import ECC
from model.ocs import SafeGuardUtility
from rl.experiment_utils import (
    load_rl_curve_artifact,
    render_rl_curve_on_axes,
    resolve_rl_curve_artifact,
)
from utils.plot_utils import (
    VIS_ACTUAL_PURPLE,
    VIS_DP_BLACK,
    VIS_DSPL_MAGENTA,
    VIS_PPO_GRAY,
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
VIS_BASELINE_GREEN = "#009E73"  # Okabe-Ito bluish green
DP_LABEL = "DP"
PROPOSED_LABEL = "PPO-PIRS (proposed)"
ACTUAL_LABEL = "Recorded operation"
_DP_STYLE = (VIS_DP_BLACK, "-")
_PROPOSED_STYLE = (VIS_PROPOSED_ORANGE, "--")
# Dash-dot-dot keeps the recorded curve distinct from the dash-dot track
# speed limit drawn by the safeguard renderer, also in greyscale.
_ACTUAL_STYLE = (VIS_ACTUAL_PURPLE, (0, (3, 1, 1, 1, 1, 1)))
# Baseline colours avoid the envelope palette (blue minimum-speed curves,
# red limits) so that no trajectory can be mistaken for a protection curve.
_BASELINE_STYLES: tuple[tuple[str, Any], ...] = (
    (VIS_BASELINE_GREEN, ":"),
    (VIS_DSPL_MAGENTA, (0, (5, 1, 1, 1))),
    (VIS_PPO_GRAY, (0, (1, 1))),
)
_TRAJECTORY_LINEWIDTH = 1.8
_DEFAULT_LEGEND_ENTRIES: tuple[tuple[str, str, Any], ...] = (
    (DP_LABEL, *_DP_STYLE),
    (PROPOSED_LABEL, *_PROPOSED_STYLE),
    (ACTUAL_LABEL, *_ACTUAL_STYLE),
)
_MARGIN_SAMPLE_STEP_M = 10.0
_MARGIN_MIN_SPEED_MPS = 5.0


@dataclass(frozen=True)
class SpeedProfile:
    label: str
    position_m: np.ndarray
    speed_mps: np.ndarray
    time_s: np.ndarray
    target_position_m: float


@dataclass(frozen=True)
class ProfileMetrics:
    time_error_s: float  # signed: actual running time minus planned time
    stop_error_m: float
    total_energy_kwh: float
    comfort_tav: float | None = None
    within_tolerance: bool | None = None
    min_limit_margin_kmh: float | None = None

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


def compute_min_limit_margin_kmh(
    profile: SpeedProfile, safeguard: SafeGuardUtility
) -> float:
    """Minimum gap (km/h) between the line speed limit x gamma and the profile.

    Evaluated on a uniform position grid while the train is moving, so the
    standstill end points do not dominate the statistic.
    """
    limits = np.asarray(safeguard.speed_limits, dtype=np.float64)
    intervals = np.asarray(safeguard.speed_limit_intervals, dtype=np.float64)
    grid = np.arange(
        profile.position_m[0], profile.position_m[-1], _MARGIN_SAMPLE_STEP_M
    )
    speed = np.interp(grid, profile.position_m, profile.speed_mps)
    index = np.clip(np.searchsorted(intervals, grid, side="right") - 1, 0, None)
    limit = limits[np.minimum(index, limits.size - 1)] * float(safeguard.gamma)
    moving = speed > _MARGIN_MIN_SPEED_MPS
    return float(np.min((limit - speed)[moving]) * 3.6)


def _parse_baseline_spec(raw: str) -> tuple[str, str]:
    label, _, model_dir = raw.partition("=")
    label, model_dir = label.strip(), model_dir.strip()
    if not label or not model_dir:
        raise ValueError(f"--baseline-rl expects LABEL=DIR, got '{raw}'")
    return label, model_dir


def _build_rl_profile(
    label: str, pos: np.ndarray, speed: np.ndarray, metadata: dict[str, object]
) -> SpeedProfile:
    position_m = as_1d_float_array(
        pos, f"{label} position", min_length=2, check_finite=True
    )
    speed_mps = as_1d_float_array(
        speed, f"{label} speed", min_length=2, check_finite=True
    )
    return SpeedProfile(
        label=label,
        position_m=position_m,
        speed_mps=speed_mps,
        time_s=recover_time_axis_from_trajectory(position_m, speed_mps),
        target_position_m=_resolve_target_position(
            metrics=metadata, position_m=position_m, source_name=label
        ),
    )


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
        label=ACTUAL_LABEL,
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
    safeguard: SafeGuardUtility | None = None,
    max_stop_error_m: float | None = None,
    max_time_error_s: float | None = None,
) -> ProfileMetrics:
    cumulative_energy_kj = compute_cumulative_energy_from_trajectory(
        pos_arr=profile.position_m,
        speed_arr=profile.speed_mps,
        vehicle=vehicle,
        track=track,
        ecc=ecc,
    )
    total_energy_kwh = float(cumulative_energy_kj[-1]) / 3600.0
    if profile.label == ACTUAL_LABEL:
        comfort_tav = None
    else:
        comfort_metrics = compute_comfort_metrics_from_trajectory(
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            max_acc_change=max_acc_change,
        )
        comfort_tav = float(comfort_metrics["comfort_tav"])

    total_time_s = float(profile.time_s[-1] - profile.time_s[0])
    time_error_s = total_time_s - target_schedule_time_s
    stop_error_m = abs(float(profile.position_m[-1]) - profile.target_position_m)
    within_tolerance = (
        abs(time_error_s) <= max_time_error_s and stop_error_m <= max_stop_error_m
        if max_time_error_s is not None and max_stop_error_m is not None
        else None
    )
    return ProfileMetrics(
        time_error_s=time_error_s,
        stop_error_m=stop_error_m,
        total_energy_kwh=total_energy_kwh,
        comfort_tav=comfort_tav,
        within_tolerance=within_tolerance,
        min_limit_margin_kmh=(
            compute_min_limit_margin_kmh(profile, safeguard)
            if safeguard is not None
            else None
        ),
    )


def format_comparison_table(
    profile_metrics: list[tuple[str, ProfileMetrics]],
) -> str:
    energy_by_label = {label: m.total_energy_kwh for label, m in profile_metrics}
    dp_energy = energy_by_label.get(DP_LABEL)
    actual_energy = energy_by_label.get(ACTUAL_LABEL)

    def _optional(value: float | None, spec: str) -> str:
        return "—" if value is None else format(value, spec)

    def _relative(
        label: str, m: ProfileMetrics, reference: float | None, reference_label: str
    ) -> str:
        if reference is None or label == reference_label:
            return "—"
        if reference_label == ACTUAL_LABEL:
            value = (reference - m.total_energy_kwh) / reference * 100.0
        else:
            value = (m.total_energy_kwh - reference) / reference * 100.0
        marker = "^a" if m.within_tolerance is False else ""
        return f"{value:.2f}{marker}"

    def _tolerance(m: ProfileMetrics) -> str:
        if m.within_tolerance is None:
            return "—"
        return "Yes" if m.within_tolerance else "No"

    rows: list[tuple[str, Any]] = [
        ("Within stop/time tolerance", lambda label, m: _tolerance(m)),
        ("Time error Δt (s)", lambda label, m: f"{m.time_error_s:+.3f}"),
        ("Stop error (m)", lambda label, m: f"{m.stop_error_m:.3f}"),
        ("Total energy (kWh)", lambda label, m: f"{m.total_energy_kwh:.3f}"),
        (
            "Energy saving vs recorded (%)",
            lambda label, m: _relative(label, m, actual_energy, ACTUAL_LABEL),
        ),
        (
            "Energy difference vs DP (%)",
            lambda label, m: _relative(label, m, dp_energy, DP_LABEL),
        ),
        (
            "Cumulative acceleration variation (m/s²)",
            lambda label, m: _optional(m.comfort_tav, ".6f"),
        ),
        (
            "Min. margin to line speed limit (km/h)",
            lambda label, m: _optional(m.min_limit_margin_kmh, ".1f"),
        ),
    ]
    headers = ["Metric", *(label for label, _ in profile_metrics)]
    values = [
        [
            title,
            *(formatter(label, metrics) for label, metrics in profile_metrics),
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
        "\n*Note: Δt is the actual running time minus the planned time "
        "(positive = late). ^a marks energy figures obtained while violating the "
        "stop/time tolerance; they are not energy savings. The speed-limit margin "
        "is the minimum of (line limit x safety factor - speed) while moving. "
        "Recorded operation acceleration calculation differs "
        "(estimated by differencing discrete operational data) and is not "
        "directly comparable with DP/RL. The cumulative acceleration variation "
        r"formula is $\sum_t |a_t - a_{t-1}|$, with unit $\mathrm{m/s^2}$.*"
    )
    return "\n".join(rendered_rows) + note


def _finalize_comparison_figure(
    figure: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    legend_entries: Sequence[tuple[str, str, Any]] | None = None,
) -> None:
    """Apply the paper layout and a trajectory-only shared legend."""
    entries = _DEFAULT_LEGEND_ENTRIES if legend_entries is None else legend_entries
    for axis in axes:
        axis.set_title("")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.8)
        for _, color, linestyle in entries
    ]
    figure.legend(
        handles,
        [label for label, _, _ in entries],
        loc="upper center",
        ncol=min(len(entries), 4),
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
            "Compare DP, the proposed RL policy, optional RL baselines, and actual "
            "operation speed profiles with speed, acceleration, cumulative-energy "
            "plots and a terminal metric table."
        )
    )
    parser.add_argument("--dp-curve-dir", default=DP_DEFAULT_SEARCH_DIR)
    parser.add_argument(
        "--rl-model-dir",
        required=True,
        help="Proposed-method model directory (best/ or final/).",
    )
    parser.add_argument(
        "--baseline-rl",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help=(
            "RL baseline to overlay, e.g. 'PPO-CR=output/.../best'. "
            "Repeat for several baselines; they are drawn in the given order."
        ),
    )
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


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    try:
        baseline_specs = [_parse_baseline_spec(raw) for raw in args.baseline_rl]
        if len(baseline_specs) > len(_BASELINE_STYLES):
            raise ValueError(
                f"At most {len(_BASELINE_STYLES)} --baseline-rl entries are supported"
            )
        dp_artifact, rl_artifact = _resolve_curve_artifacts(
            dp_curve_dir=args.dp_curve_dir,
            rl_model_dir=args.rl_model_dir,
        )
        dp_pos, dp_speed, dp_time, dp_metadata = load_dp_curve_artifact(dp_artifact)
        rl_entries: list[tuple[str, OptimizedCurveArtifact, str, Any]] = [
            (label, resolve_rl_curve_artifact(curve_dir=model_dir), *style)
            for (label, model_dir), style in zip(
                baseline_specs, _BASELINE_STYLES, strict=False
            )
        ]
        rl_entries.append((PROPOSED_LABEL, rl_artifact, *_PROPOSED_STYLE))

        rl_profiles: list[tuple[SpeedProfile, dict[str, object], str, Any]] = []
        target_time_s: float | None = None
        for label, artifact, color, linestyle in rl_entries:
            pos, speed, metadata = load_rl_curve_artifact(artifact)
            task_time_s = _resolve_target_schedule_time(
                dp_metrics=dp_metadata, rl_metrics=metadata
            )
            if (
                target_time_s is not None
                and abs(task_time_s - target_time_s) > _TARGET_TIME_TOLERANCE_S
            ):
                raise ValueError(
                    f"{label} target_time_s differs from the other RL artifacts"
                )
            target_time_s = task_time_s
            profile = _build_rl_profile(label, pos, speed, metadata)
            rl_profiles.append((profile, metadata, color, linestyle))
        assert target_time_s is not None
        dp_profile = SpeedProfile(
            label=DP_LABEL,
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
        real_profile = load_real_operation_profile(args.real_curve)
        styled_profiles: list[tuple[SpeedProfile, str, Any]] = [
            (dp_profile, *_DP_STYLE),
            *((profile, color, ls) for profile, _, color, ls in rl_profiles),
            (real_profile, *_ACTUAL_STYLE),
        ]
        profiles = [profile for profile, _, _ in styled_profiles]
        _ = _validate_common_target_position(profiles)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    vehicle, track, _, train_service = build_scenario(schedule_time_s=target_time_s)
    ecc = _build_ecc()
    margin_safeguard = build_safeguard_utility(args.factor)
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
                safeguard=margin_safeguard,
                max_stop_error_m=train_service.max_stop_error,
                max_time_error_s=train_service.max_arr_time_error_s,
            ),
        )
        for profile in profiles
    ]

    print(f"DP curve: {dp_artifact.npz_path}")
    for label, artifact, _, _ in rl_entries:
        print(f"{label} curve: {artifact.npz_path}")
    print(f"{ACTUAL_LABEL} curve: {args.real_curve}")
    print(f"Common target running time: {target_time_s:.3f} s")
    print("\nTrajectory comparison metrics:")
    print(format_comparison_table(metrics_by_label))

    apply_sci_curve_style()
    fig, (ax_speed, ax_acc, ax_energy) = _create_comparison_axes()
    safeguard = None if args.no_safeguard else margin_safeguard
    render_dp_curve_on_axes(
        ax=ax_speed,
        pos_arr=dp_profile.position_m,
        speed_arr=dp_profile.speed_mps,
        metrics=dp_metadata,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
        curve_color=_DP_STYLE[0],
        curve_label="DP optimized speed curve",
        safeguard=safeguard,
        render_endpoints=False,
    )
    ax_speed.lines[-1].set_linewidth(_TRAJECTORY_LINEWIDTH)
    for profile, metadata, color, linestyle in rl_profiles:
        render_rl_curve_on_axes(
            ax=ax_speed,
            pos_arr=profile.position_m,
            speed_arr=profile.speed_mps,
            metrics=metadata,
            no_safeguard=True,
            factor=args.factor,
            curve_color=color,
            curve_label=f"{profile.label} speed curve",
            safeguard=safeguard,
            render_endpoints=False,
        )
        ax_speed.lines[-1].set_linestyle(linestyle)
        ax_speed.lines[-1].set_linewidth(_TRAJECTORY_LINEWIDTH)
    ax_speed.plot(
        real_profile.position_m,
        real_profile.speed_mps * 3.6,
        color=_ACTUAL_STYLE[0],
        linestyle=_ACTUAL_STYLE[1],
        linewidth=_TRAJECTORY_LINEWIDTH,
        label=f"{ACTUAL_LABEL} speed curve",
    )
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_xlabel("")
    add_panel_label(ax_speed, "(a)")

    for profile, color, linestyle in styled_profiles:
        if profile.label == ACTUAL_LABEL:
            continue
        ax_acc.plot(
            _compute_segment_midpoints(profile.position_m),
            compute_segment_accelerations(profile.position_m, profile.speed_mps),
            color=color,
            linestyle=linestyle,
            linewidth=_TRAJECTORY_LINEWIDTH,
            label=f"{profile.label} acceleration",
        )
    ax_acc.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax_acc.set_xlabel("")
    ax_acc.set_ylabel("Acceleration (m/s²)")
    ax_acc.set_ylim(-1.5, 1.6)
    add_panel_label(ax_acc, "(b)")
    apply_sci_grid(ax_acc)

    for profile, color, linestyle in styled_profiles:
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
            linewidth=_TRAJECTORY_LINEWIDTH,
            label=f"{profile.label} cumulative energy",
        )
    ax_energy.set_xlabel("Position (km)")
    ax_energy.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    ax_energy.set_ylabel("Cumulative energy (kWh)")
    add_panel_label(ax_energy, "(c)")
    apply_sci_grid(ax_energy)
    _finalize_comparison_figure(
        fig,
        (ax_speed, ax_acc, ax_energy),
        [(profile.label, color, ls) for profile, color, ls in styled_profiles],
    )

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
