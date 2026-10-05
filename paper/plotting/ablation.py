"""Ablation and schedule-change figures from new run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from mtto.domain.scenario import Scenario
from mtto.io.artifacts import read_completed_run
from mtto.io.scenario import load_scenario, load_tasks
from paper.experiments.spec import ExperimentSpec
from paper.plotting.profiles import (
    DANGER_VIEW_LAYERS,
    render_safeguard,
)
from paper.plotting.style import (
    PAPER_LEGEND_FONT_SIZE,
    SCI_LINE_WIDTH,
    SCI_SERIES_LINE_STYLES,
    VIS_DP_BLACK,
    VIS_DSPL_MAGENTA,
    VIS_HARD_LIMIT_RED,
    VIS_PPO_GRAY,
    VIS_PROPOSED_ORANGE,
    VIS_SAFE_BLUE,
    add_panel_label,
    apply_paper_style,
    apply_sci_figure_layout,
    apply_sci_grid,
    sci_tint_color,
)

METHOD_COLORS = (VIS_PPO_GRAY, VIS_SAFE_BLUE, VIS_DSPL_MAGENTA, VIS_PROPOSED_ORANGE)
# Zoom windows of the representative-profile figure: departure acceleration
# through the first ASA transitions, and the station approach.
DEPARTURE_ZOOM_M = 6000.0
ARRIVAL_ZOOM_M = 4000.0


def require_clean(run_dirs: tuple[Path, ...]) -> None:
    """Reject publication figures built from dirty worktree results."""
    dirty = [
        path
        for path in run_dirs
        if json.loads((path / "paper.json").read_text(encoding="utf-8"))["dirty"]
    ]
    if dirty:
        raise ValueError(
            "Dirty runs cannot produce figures: " + ", ".join(map(str, dirty))
        )


def _plot_bands(
    axis: plt.Axes,
    x: np.ndarray,
    series: dict,
    label: str,
    color: str,
    style: dict,
    *,
    highlight: bool = False,
    markevery: int = 2,
    clip: tuple[float, float] | None = None,
) -> None:
    mean = np.asarray(series["mean"], dtype=float)
    std = np.asarray(series["std"], dtype=float)
    plotted = np.clip(mean, *clip) if clip is not None else mean
    lower = np.clip(mean - std, *clip) if clip is not None else mean - std
    upper = np.clip(mean + std, *clip) if clip is not None else mean + std
    axis.plot(
        x,
        plotted,
        label=label,
        color=color,
        linewidth=SCI_LINE_WIDTH + 0.4 if highlight else SCI_LINE_WIDTH,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=markevery,
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.7,
    )
    axis.fill_between(
        x,
        lower,
        upper,
        color=sci_tint_color(color),
        linewidth=0,
        where=np.isfinite(mean) & np.isfinite(std),
    )


def _format_transition_axis(axis: plt.Axes) -> None:
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)


def method_figures(
    summary: dict[str, object], spec: ExperimentSpec, output: Path
) -> tuple[Path, Path]:
    apply_paper_style()
    output.mkdir(parents=True, exist_ok=True)
    methods = summary["methods"]
    labels = summary["method_labels"]
    rollout_steps = spec.train["num_envs"] * spec.train["n_steps_per_env"]
    budget_rollouts = spec.train["training_rollouts"]
    interval = spec.train["evaluation_interval_rollouts"] or budget_rollouts
    last_evaluation_rollout = ((budget_rollouts - 1) // interval) * interval
    if last_evaluation_rollout == 0:
        last_evaluation_rollout = budget_rollouts
    axis_end = budget_rollouts * rollout_steps
    inset_end = last_evaluation_rollout * rollout_steps
    fig, axes = plt.subplots(1, 2)
    for index, method in enumerate(methods):
        data = summary["training_curves"][method]
        x = np.asarray(data["training_steps"], dtype=float)
        style = SCI_SERIES_LINE_STYLES[index % len(SCI_SERIES_LINE_STYLES)]
        for axis, key in zip(
            axes, ("speed_violation_rate", "arrival_ratio"), strict=True
        ):
            _plot_bands(
                axis,
                x,
                data[key],
                labels[method],
                METHOD_COLORS[index % 4],
                style,
                highlight=method == "ppo_pirs",
                clip=(0.0, np.inf) if key == "speed_violation_rate" else (0.0, 1.0),
            )
    inset = axes[0].inset_axes((0.48, 0.46, 0.48, 0.48))
    for index, method in enumerate(methods):
        data = summary["training_curves"][method]
        x = np.asarray(data["training_steps"], dtype=float)
        _plot_bands(
            inset,
            x,
            data["speed_violation_rate"],
            "_nolegend_",
            METHOD_COLORS[index % 4],
            SCI_SERIES_LINE_STYLES[index % len(SCI_SERIES_LINE_STYLES)],
            highlight=method == "ppo_pirs",
            clip=(0.0, np.inf),
        )
    training_inset_start = 0.5 * axis_end
    if training_inset_start >= inset_end:
        training_inset_start = 0.5 * inset_end
    inset.set_xlim(training_inset_start, inset_end)
    inset.set_ylim(0, 5)
    inset.tick_params(labelsize=PAPER_LEGEND_FONT_SIZE)
    apply_sci_grid(inset)
    _format_transition_axis(inset)
    for axis, ylabel, panel in zip(
        axes,
        ("Speed violations / 10⁴ transitions", "Training arrival rate"),
        ("(a)", "(b)"),
        strict=True,
    ):
        axis.set_xlabel("Environment transitions")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, axis_end)
        _format_transition_axis(axis)
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=panel)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(-0.03, 1.03)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    apply_sci_figure_layout(
        fig,
        columns="text",
        height_in=2.9,
        left=0.11,
        right=0.98,
        bottom=0.17,
        top=0.89,
        wspace=0.30,
    )
    training_path = output / "method_training_curves.pdf"
    fig.savefig(training_path)
    plt.close(fig)

    profiles_path = representative_profiles_figure(summary, spec, output)
    return training_path, profiles_path


def representative_profiles_figure(
    summary: dict[str, object], spec: ExperimentSpec, output: Path
) -> Path:
    """Overlay the representative speed profile of every ablation variant."""
    apply_paper_style()
    output.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(spec.scenario, spec.line_dir)
    task = load_tasks(spec.tasks)[spec.task]
    methods = summary["methods"]
    labels = summary["method_labels"]
    profiles = []
    for method in methods:
        profiles.append(
            read_completed_run(
                Path(summary["representative_policies"][method]["run_dir"])
            ).payload.profile
        )
    fig = plt.figure()
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    overview = fig.add_subplot(grid[0, :])
    departure = fig.add_subplot(grid[1, 0])
    arrival = fig.add_subplot(grid[1, 1])
    windows = (
        (overview, (task.start_position_m, task.target_position_m)),
        (
            departure,
            (task.start_position_m, task.start_position_m + DEPARTURE_ZOOM_M),
        ),
        (arrival, (task.target_position_m - ARRIVAL_ZOOM_M, task.target_position_m)),
    )
    for panel_index, (axis, (left, right)) in enumerate(windows):
        render_safeguard(scenario.safeguard, ax=axis, layers=DANGER_VIEW_LAYERS)
        top = 0.0
        for index, (method, profile) in enumerate(zip(methods, profiles, strict=True)):
            speed = profile.speed_mps * 3.6
            axis.plot(
                profile.position_m,
                speed,
                color=METHOD_COLORS[index % 4],
                linestyle=SCI_SERIES_LINE_STYLES[index % 4]["linestyle"],
                linewidth=SCI_LINE_WIDTH + (0.4 if method == "ppo_pirs" else 0.0),
                label=labels[method],
            )
            inside = (profile.position_m >= left) & (profile.position_m <= right)
            if np.any(inside):
                top = max(top, float(np.max(speed[inside])))
        axis.set_xlim(left, right)
        axis.set_ylim(0.0, top * 1.15 if top > 0.0 else None)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
        axis.set(xlabel="Position (km)", ylabel="Speed (km/h)")
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=f"({chr(97 + panel_index)})")
    handles, legend_labels = overview.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    apply_sci_figure_layout(
        fig,
        columns="text",
        height_in=4.9,
        left=0.11,
        right=0.98,
        bottom=0.10,
        top=0.89,
        wspace=0.30,
        hspace=0.40,
    )
    path = output / "method_representative_profiles.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def schedule_change_figure(
    summary: dict[str, object], scenario: Scenario, output: Path
) -> Path:
    """Speed profiles after the schedule change: PPO-PIRS solid, DP dashed."""
    apply_paper_style()
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots()
    render_safeguard(scenario.safeguard, ax=axis, layers=DANGER_VIEW_LAYERS)
    case_colors = {}
    # Each case also gets a marker (offset along the line) so the figure reads
    # in greyscale; PPO-PIRS markers are filled, DP markers hollow.
    case_markers = {0: ("o", 0.0), 1: ("^", 0.027), -1: ("v", 0.053)}
    extent = [np.inf, -np.inf]
    for entry in summary["entries"]:
        delta = entry["case"]["delta_time_s"]
        color = case_colors.setdefault(
            delta,
            VIS_DP_BLACK
            if delta == 0
            else VIS_PROPOSED_ORANGE
            if delta > 0
            else VIS_SAFE_BLUE,
        )
        profile = read_completed_run(Path(entry["run_dir"])).payload.profile
        extent = [
            min(extent[0], float(profile.position_m[0])),
            max(extent[1], float(profile.position_m[-1])),
        ]
        marker, offset = case_markers[int(np.sign(delta))]
        axis.plot(
            profile.position_m,
            profile.speed_mps * 3.6,
            color=color,
            linestyle="-" if entry["method"] == "PPO-PIRS" else "--",
            linewidth=SCI_LINE_WIDTH,
            marker=marker,
            markersize=4.5,
            markevery=(offset, 0.08),
            markerfacecolor=color if entry["method"] == "PPO-PIRS" else "white",
            markeredgecolor=color,
        )
    axis.axvline(
        summary["change_distance_m"],
        color=VIS_HARD_LIMIT_RED,
        linestyle=":",
        linewidth=1.0,
    )
    axis.set_xlim(*extent)
    axis.set_ylim(0.0, float(np.nanmax(scenario.safeguard.speed_limits) * 3.6) * 1.08)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    axis.set(xlabel="Position (km)", ylabel="Speed (km/h)")
    apply_sci_grid(axis)
    handles = [
        *(
            Line2D(
                [],
                [],
                color=color,
                linewidth=SCI_LINE_WIDTH,
                marker=case_markers[int(np.sign(delta))][0],
                markersize=4.5,
            )
            for delta, color in case_colors.items()
        ),
        Line2D([], [], color="#555555", linestyle="-", linewidth=SCI_LINE_WIDTH),
        Line2D([], [], color="#555555", linestyle="--", linewidth=SCI_LINE_WIDTH),
        Line2D([], [], color=VIS_HARD_LIMIT_RED, linestyle=":", linewidth=1.0),
    ]
    labels = [
        *(
            "Unchanged"
            if delta == 0
            else f"{'+' if delta > 0 else '−'}{abs(delta):g} s"
            for delta in case_colors
        ),
        "PPO-PIRS",
        "DP",
        "Timetable change",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        handlelength=2.0,
        columnspacing=0.8,
    )
    apply_sci_figure_layout(
        fig, columns="text", height_in=3.4, left=0.11, right=0.98, bottom=0.14, top=0.88
    )
    path = output / "schedule_time_change_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path
