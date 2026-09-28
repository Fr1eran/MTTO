"""Ablation and schedule-change figures from new run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mtto.domain.scenario import Scenario
from mtto.io.artifacts import read_completed_run
from paper.experiments.spec import ExperimentSpec
from paper.plotting.profiles import (
    DANGER_VIEW_LAYERS,
    render_safeguard,
)
from paper.plotting.style import (
    SCI_LINE_WIDTH,
    SCI_SERIES_LINE_STYLES,
    VIS_ACTUAL_PURPLE,
    VIS_DP_BLACK,
    VIS_DSPL_MAGENTA,
    VIS_HARD_LIMIT_RED,
    VIS_PPO_GRAY,
    VIS_PROPOSED_ORANGE,
    VIS_SAFE_BLUE,
    add_panel_label,
    apply_sci_curve_style,
    apply_sci_figure_layout,
    apply_sci_grid,
    sci_tint_color,
)

METHOD_COLORS = (VIS_PPO_GRAY, VIS_SAFE_BLUE, VIS_DSPL_MAGENTA, VIS_PROPOSED_ORANGE)
STEP_COLORS = (VIS_PPO_GRAY, VIS_PROPOSED_ORANGE, VIS_SAFE_BLUE, VIS_ACTUAL_PURPLE)


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
    apply_sci_curve_style()
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
    inset.tick_params(labelsize=7)
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
        fig, columns=2, height_in=2.85, left=0.11, bottom=0.18, top=0.90, wspace=0.24
    )
    training_path = output / "method_training_curves.pdf"
    fig.savefig(training_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2)
    panels = (
        (axes[0, 0], "stop_error_m", "Absolute stop error (m)"),
        (axes[0, 1], "time_error_s", "Absolute time error (s)"),
        (axes[1, 0], "total_energy_j", "Trajectory energy (kWh)"),
        (axes[1, 1], "comfort_tav", "Cumulative acceleration variation (m/s²)"),
    )
    for panel_index, (axis, key, ylabel) in enumerate(panels):
        for index, method in enumerate(methods):
            curve = summary["evaluation_curves"][method]
            _plot_bands(
                axis,
                np.asarray(curve["training_steps"], dtype=float),
                curve[key],
                labels[method],
                METHOD_COLORS[index % 4],
                SCI_SERIES_LINE_STYLES[index % len(SCI_SERIES_LINE_STYLES)],
                highlight=method == "ppo_pirs",
            )
        axis.set(xlabel="Environment transitions", ylabel=ylabel)
        axis.set_xlim(0, axis_end)
        _format_transition_axis(axis)
        axis.set_ylim(bottom=0)
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=f"({chr(97 + panel_index)})")
    axes[0, 0].axhline(0.3, color="#666666", linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(10.0, color="#666666", linestyle="--", linewidth=0.8)
    for axis, key, upper in (
        (axes[0, 0], "stop_error_m", 1.5),
        (axes[0, 1], "time_error_s", 30.0),
    ):
        inset = axis.inset_axes((0.50, 0.48, 0.46, 0.46))
        for index, method in enumerate(methods):
            curve = summary["evaluation_curves"][method]
            x = np.asarray(curve["training_steps"], dtype=float)
            _plot_bands(
                inset,
                x,
                curve[key],
                "_nolegend_",
                METHOD_COLORS[index % 4],
                SCI_SERIES_LINE_STYLES[index % len(SCI_SERIES_LINE_STYLES)],
                highlight=method == "ppo_pirs",
            )
        evaluation_inset_start = 0.75 * axis_end
        if evaluation_inset_start >= inset_end:
            evaluation_inset_start = 0.5 * inset_end
        inset.set_xlim(evaluation_inset_start, inset_end)
        inset.set_ylim(0, upper)
        inset.axhline(
            0.3 if key == "stop_error_m" else 10.0,
            color="#666666",
            linestyle="--",
            linewidth=0.8,
        )
        inset.tick_params(labelsize=7)
        apply_sci_grid(inset)
        _format_transition_axis(inset)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        frameon=False,
    )
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=4.6,
        left=0.11,
        bottom=0.12,
        top=0.92,
        wspace=0.30,
        hspace=0.32,
    )
    metrics_path = output / "method_trajectory_metrics.pdf"
    fig.savefig(metrics_path)
    plt.close(fig)
    return training_path, metrics_path


def step_distance_figure(
    summary: dict[str, object], spec: ExperimentSpec, output: Path
) -> Path:
    apply_sci_curve_style()
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2)
    axis_end = (
        spec.train["training_rollouts"]
        * spec.train["num_envs"]
        * spec.train["n_steps_per_env"]
    )
    for axis in axes:
        axis.set_box_aspect(3 / 4)
    for index, (variant_id, variant) in enumerate(summary["variants"].items()):
        curve = summary["curves"][variant_id]
        x = np.asarray(curve["training_steps"], dtype=float)
        style = SCI_SERIES_LINE_STYLES[index % len(SCI_SERIES_LINE_STYLES)]
        for axis, key in zip(axes, ("route_completion_ratio", "feasible"), strict=True):
            _plot_bands(
                axis,
                x,
                curve[key],
                variant["label"],
                STEP_COLORS[index % 4],
                style,
                highlight=variant_id == "30p0",
                markevery=3,
                clip=(0.0, 1.0) if key == "feasible" else None,
            )
    for axis, ylabel, panel in zip(
        axes,
        ("Route completion ratio", "Strict feasibility rate"),
        ("(a)", "(b)"),
        strict=True,
    ):
        axis.set(
            xlabel="Environment transitions",
            ylabel=ylabel,
            xlim=(0, axis_end),
            ylim=(0.0, 1.0) if panel == "(a)" else (-0.03, 1.03),
        )
        _format_transition_axis(axis)
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=panel)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=min(4, len(summary["variants"])),
        borderaxespad=0,
        handlelength=1.8,
        columnspacing=1.2,
        frameon=False,
    )
    apply_sci_figure_layout(
        fig, columns=2, height_in=3.0, left=0.10, bottom=0.19, top=0.90, wspace=0.24
    )
    path = output / "step_distance_learning_curves.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def schedule_change_figure(
    summary: dict[str, object], scenario: Scenario, output: Path
) -> Path:
    apply_sci_curve_style()
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots()
    render_safeguard(scenario.safeguard, ax=axis, layers=DANGER_VIEW_LAYERS)
    original_profile = None
    trigger = None
    positions = []
    speeds = []
    case_handles = []
    case_labels = []
    for case in sorted(
        summary["cases"],
        key=lambda item: (
            item["case"]["delta_time_s"] < 0,
            item["case"]["delta_time_s"] != 0,
            abs(item["case"]["delta_time_s"]),
        ),
    ):
        delta = case["case"]["delta_time_s"]
        completed = read_completed_run(Path(case["run_dir"]))
        profile = completed.payload.profile
        if delta == 0:
            original_profile = profile
        else:
            trigger_position = completed.record.task["schedule_change"][
                "trigger_position_m"
            ]
            reached = profile.position_m[profile.position_m >= trigger_position]
            if reached.size:
                trigger = float(reached[0])
        label = (
            "Original" if delta == 0 else f"{'+' if delta > 0 else '−'}{abs(delta):g} s"
        )
        color = (
            VIS_DP_BLACK
            if delta == 0
            else VIS_PROPOSED_ORANGE
            if delta > 0
            else VIS_ACTUAL_PURPLE
        )
        (handle,) = axis.plot(
            profile.position_m,
            profile.speed_mps * 3.6,
            color=color,
            linestyle="--" if delta < 0 else "-",
            linewidth=1.7 if delta == 0 else 1.5,
        )
        case_handles.append(handle)
        case_labels.append(label)
        positions.append(np.asarray(profile.position_m, dtype=float))
        speeds.append(np.asarray(profile.speed_mps, dtype=float) * 3.6)
    trigger_handle = None
    if trigger is not None and original_profile is not None:
        axis.scatter(
            [trigger],
            [
                np.interp(
                    trigger,
                    original_profile.position_m,
                    original_profile.speed_mps * 3.6,
                )
            ],
            marker="*",
            s=80,
            color=VIS_HARD_LIMIT_RED,
            zorder=8,
        )
        trigger_handle = Line2D(
            [],
            [],
            marker="*",
            markersize=9,
            color=VIS_HARD_LIMIT_RED,
            linestyle="None",
        )
    if positions:
        position_min = min(float(np.nanmin(position)) for position in positions)
        position_max = max(float(np.nanmax(position)) for position in positions)
        margin = max((position_max - position_min) * 0.03, 1.0)
        axis.set_xlim(position_min - margin, position_max + margin)
    if speeds:
        curve_max = max(float(np.nanmax(speed)) for speed in speeds)
        limit_max = float(np.nanmax(scenario.safeguard.speed_limits) * 3.6)
        axis.set_ylim(0.0, max(curve_max, limit_max) * 1.08)
    axis.set(xlabel="Position (m)", ylabel="Speed (km/h)")
    apply_sci_grid(axis)
    if trigger_handle is not None:
        case_handles.append(trigger_handle)
        case_labels.append("Schedule change")
    fig.legend(
        case_handles,
        case_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.8,
    )
    apply_sci_figure_layout(
        fig, columns=2, height_in=3.8, left=0.09, right=0.97, bottom=0.15, top=0.88
    )
    path = output / "schedule_time_change_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path
