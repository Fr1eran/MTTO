from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from mtto.domain.safeguard import Safeguard, build_safeguard
from mtto.domain.scenario import Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import QualityReport, assess
from mtto.io.artifacts import RunKind, read_completed_run, task_from_json
from paper.figures import load_paper_scenario
from paper.plotting.profiles import (
    render_dp_curve_on_axes,
    render_rl_curve_on_axes,
)
from paper.plotting.style import (
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
from paper.real_operation import real_operation_profile

FIGURE_FILENAME = "dp_rl_actual_comparison.pdf"
VIS_BASELINE_GREEN = "#009E73"
DP_LABEL = "DP"
PROPOSED_LABEL = "PPO-PIRS (proposed)"
ACTUAL_LABEL = "Recorded operation"
_DP_STYLE = (VIS_DP_BLACK, "-")
_PROPOSED_STYLE = (VIS_PROPOSED_ORANGE, "--")
_ACTUAL_STYLE = (VIS_ACTUAL_PURPLE, (0, (3, 1, 1, 1, 1, 1)))
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
class ProfileMetrics:
    time_error_s: float
    stop_error_m: float
    total_energy_kwh: float
    comfort_tav: float | None = None
    within_tolerance: bool = True
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


def compute_min_limit_margin_kmh(profile: SpeedProfile, safeguard: Safeguard) -> float:
    """Minimum gap (km/h) between the line speed limit x gamma and the profile."""
    limits = np.asarray(safeguard.speed_limits, dtype=np.float64)
    intervals = np.asarray(safeguard.speed_limit_intervals, dtype=np.float64)
    grid = np.arange(
        profile.position_m[0], profile.position_m[-1], _MARGIN_SAMPLE_STEP_M
    )
    speed = np.interp(grid, profile.position_m, profile.speed_mps)
    index = np.clip(np.searchsorted(intervals, grid, side="right") - 1, 0, None)
    limit = limits[np.minimum(index, limits.size - 1)] * float(safeguard.params.factor)
    moving = speed > _MARGIN_MIN_SPEED_MPS
    if not np.any(moving):
        return float("nan")
    return float(np.min((limit - speed)[moving]) * 3.6)


def _parse_baseline_spec(raw: str) -> tuple[str, str]:
    label, _, model_dir = raw.partition("=")
    label, model_dir = label.strip(), model_dir.strip()
    if not label or not model_dir:
        raise ValueError(f"--baseline-rl expects LABEL=DIR, got '{raw}'")
    return label, model_dir


def _validate_same_task(task_a: Task, task_b: Task, label_b: str) -> None:
    if (
        task_a.schedule_time_s != task_b.schedule_time_s
        or task_a.target_position_m != task_b.target_position_m
        or task_a.start_position_m != task_b.start_position_m
    ):
        raise ValueError(
            f"Task mismatch between DP and {label_b}; select runs from the same task."
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
    parser.add_argument(
        "--dp-run",
        type=Path,
        required=True,
        help="DP solve run directory containing run.json, profile.npz, quality.json",
    )
    parser.add_argument(
        "--rl-run",
        type=Path,
        required=True,
        help="Proposed-method RL run directory (RL_TRAIN or EVALUATION).",
    )
    parser.add_argument(
        "--rl-best",
        action="store_true",
        default=False,
        help="Use best/ payload for RL training run.",
    )
    parser.add_argument(
        "--baseline-rl",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help=(
            "RL baseline to overlay, e.g. 'PPO-CR=runs/ppo_cr'. "
            "Repeat for several baselines; they are drawn in the given order."
        ),
    )
    parser.add_argument("--no-safeguard", action="store_true")
    parser.add_argument("--factor", type=float, default=None)
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        baseline_specs = [_parse_baseline_spec(raw) for raw in args.baseline_rl]
        if len(baseline_specs) > len(_BASELINE_STYLES):
            raise ValueError(
                f"At most {len(_BASELINE_STYLES)} --baseline-rl entries are supported"
            )

        dp_run = read_completed_run(args.dp_run)
        if dp_run.record.kind != RunKind.DP_SOLVE:
            raise ValueError(f"Expected DP solve run, got {dp_run.record.kind.value}")

        rl_run = read_completed_run(args.rl_run)
        if rl_run.record.kind not in (RunKind.RL_TRAIN, RunKind.EVALUATION):
            raise ValueError(f"Expected RL run, got {rl_run.record.kind.value}")
        baseline_runs: list[tuple[str, Any, str, Any]] = []
        for (label, dir_str), style in zip(
            baseline_specs, _BASELINE_STYLES, strict=False
        ):
            b_run = read_completed_run(dir_str)
            if b_run.record.kind not in (RunKind.RL_TRAIN, RunKind.EVALUATION):
                raise ValueError(
                    f"Baseline {label} must be RL run, got {b_run.record.kind.value}"
                )
            baseline_runs.append((label, b_run, *style))

        if args.rl_best:
            if rl_run.record.kind != RunKind.RL_TRAIN:
                raise ValueError("--rl-best is only valid for RL training runs")
            if rl_run.payload.best is None:
                raise ValueError(
                    f"--rl-best specified but {args.rl_run} has no best/ artifacts"
                )
            for label, b_run, _, _ in baseline_runs:
                if b_run.record.kind != RunKind.RL_TRAIN:
                    raise ValueError(
                        "--rl-best is only valid for RL training runs, "
                        f"baseline {label} is {b_run.record.kind.value}"
                    )
                if b_run.payload.best is None:
                    raise ValueError(
                        f"--rl-best specified but baseline {label} has no "
                        "best/ artifacts"
                    )

        scenario = load_paper_scenario()
        for run_path, run_obj in [(args.dp_run, dp_run), (args.rl_run, rl_run)]:
            if run_obj.record.scenario_hash != scenario.scenario_hash:
                raise ValueError(
                    f"Scenario hash mismatch for run at {run_path}: "
                    f"expected {scenario.scenario_hash}, "
                    f"got {run_obj.record.scenario_hash}"
                )
        for label, b_run, _, _ in baseline_runs:
            if b_run.record.scenario_hash != scenario.scenario_hash:
                raise ValueError(
                    f"Scenario hash mismatch for baseline {label}: "
                    f"expected {scenario.scenario_hash}, "
                    f"got {b_run.record.scenario_hash}"
                )

        dp_task = task_from_json(dp_run.record.task)
        if dp_task.schedule_time_s is None:
            raise ValueError(
                "Speed profile comparison requires runs with a scheduled arrival time, "
                "but task schedule_time_s is None"
            )
        rl_task = task_from_json(rl_run.record.task)
        _validate_same_task(dp_task, rl_task, "RL")
        for label, b_run, _, _ in baseline_runs:
            b_task = task_from_json(b_run.record.task)
            _validate_same_task(dp_task, b_task, label)

        target_rl_payload = rl_run.payload.best if args.rl_best else rl_run.payload

        real_profile = real_operation_profile(scenario, dp_task)
        real_quality = assess(real_profile, scenario, dp_task)

        styled_profiles: list[tuple[SpeedProfile, QualityReport, str, str, Any]] = [
            (
                dp_run.payload.profile,
                dp_run.payload.quality,
                DP_LABEL,
                *_DP_STYLE,
            )
        ]
        for label, b_run, color, ls in baseline_runs:
            b_payload = b_run.payload.best if args.rl_best else b_run.payload
            styled_profiles.append(
                (b_payload.profile, b_payload.quality, label, color, ls)
            )

        styled_profiles.append(
            (
                target_rl_payload.profile,
                target_rl_payload.quality,
                PROPOSED_LABEL,
                *_PROPOSED_STYLE,
            )
        )
        styled_profiles.append(
            (real_profile, real_quality, ACTUAL_LABEL, *_ACTUAL_STYLE)
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    effective_factor = (
        scenario.safeguard.params.factor if args.factor is None else args.factor
    )
    if effective_factor != scenario.safeguard.params.factor:
        margin_safeguard = build_safeguard(
            params=dataclasses.replace(
                scenario.safeguard.params, factor=effective_factor
            ),
            line=scenario.line,
            levi_curves=scenario.safeguard.levi_curves,
            brake_curves=scenario.safeguard.brake_curves,
            min_curves=scenario.safeguard.min_curves,
            max_curves=scenario.safeguard.max_curves,
        )
    else:
        margin_safeguard = scenario.safeguard

    metrics_by_label: list[tuple[str, ProfileMetrics]] = []
    for profile, quality, label, _color, _linestyle in styled_profiles:
        time_err = quality.metrics.arrival_time_error_s
        stop_err = quality.metrics.stop_error_m
        tot_energy_kwh = quality.metrics.total_energy_kj / 3600.0
        c_tav = None if label == ACTUAL_LABEL else quality.metrics.comfort_tav_mps2
        within_tol = (
            abs(time_err) <= dp_task.max_arr_time_error_s
            and stop_err <= dp_task.max_stop_error_m
        )
        margin_kmh = compute_min_limit_margin_kmh(profile, margin_safeguard)

        metrics_by_label.append(
            (
                label,
                ProfileMetrics(
                    time_error_s=time_err,
                    stop_error_m=stop_err,
                    total_energy_kwh=tot_energy_kwh,
                    comfort_tav=c_tav,
                    within_tolerance=within_tol,
                    min_limit_margin_kmh=margin_kmh,
                ),
            )
        )

    print(f"DP run: {args.dp_run}")
    for label, b_run, _, _ in baseline_runs:
        print(f"{label} run: {b_run.record.run_id}")
    print(f"{PROPOSED_LABEL} run: {args.rl_run}")
    print(f"{ACTUAL_LABEL} (computed on the fly)")
    print(f"Common target running time: {dp_task.schedule_time_s:.3f} s")
    print("\nTrajectory comparison metrics:")
    print(format_comparison_table(metrics_by_label))

    apply_sci_curve_style()
    fig, (ax_speed, ax_acc, ax_energy) = _create_comparison_axes()
    safeguard = None if args.no_safeguard else margin_safeguard

    dp_profile, dp_quality, _, _, _ = styled_profiles[0]
    render_dp_curve_on_axes(
        ax=ax_speed,
        profile=dp_profile,
        task=dp_task,
        no_safeguard=args.no_safeguard,
        factor=effective_factor,
        curve_color=_DP_STYLE[0],
        curve_label="DP optimized speed curve",
        safeguard=safeguard,
        render_endpoints=False,
    )
    ax_speed.lines[-1].set_linewidth(_TRAJECTORY_LINEWIDTH)

    for profile, _quality, label, color, linestyle in styled_profiles[1:-1]:
        render_rl_curve_on_axes(
            ax=ax_speed,
            profile=profile,
            task=dp_task,
            no_safeguard=True,
            factor=effective_factor,
            curve_color=color,
            curve_label=f"{label} speed curve",
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

    for profile, _quality, label, color, linestyle in styled_profiles:
        if label == ACTUAL_LABEL:
            continue
        ax_acc.plot(
            _compute_segment_midpoints(profile.position_m),
            profile.segment_acceleration_mps2,
            color=color,
            linestyle=linestyle,
            linewidth=_TRAJECTORY_LINEWIDTH,
            label=f"{label} acceleration",
        )
    ax_acc.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax_acc.set_xlabel("")
    ax_acc.set_ylabel("Acceleration (m/s²)")
    ax_acc.set_ylim(-1.5, 1.6)
    add_panel_label(ax_acc, "(b)")
    apply_sci_grid(ax_acc)

    for profile, _quality, label, color, linestyle in styled_profiles:
        cumulative_energy = profile.propulsion_energy_kj + profile.levitation_energy_kj
        ax_energy.plot(
            profile.position_m,
            cumulative_energy / 3600.0,
            color=color,
            linestyle=linestyle,
            linewidth=_TRAJECTORY_LINEWIDTH,
            label=f"{label} cumulative energy",
        )
    ax_energy.set_xlabel("Position (km)")
    ax_energy.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    ax_energy.set_ylabel("Cumulative energy (kWh)")
    add_panel_label(ax_energy, "(c)")
    apply_sci_grid(ax_energy)

    _finalize_comparison_figure(
        fig,
        (ax_speed, ax_acc, ax_energy),
        [(label, color, ls) for _, _, label, color, ls in styled_profiles],
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
