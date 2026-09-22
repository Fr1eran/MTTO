"""Train and display the method ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from contracts.ablation import AblationManifest
from rl.experiment_statistics import assess_constraints
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_STEP_DISTANCE,
    evaluate_final_training_run,
    learning_rate_schedule_parameters,
    resolve_reward_preset,
    reward_config_parameters,
    train_single_experiment,
)
from utils.ablation import (
    AblationDriver,
    AblationRun,
    AblationSpec,
    ArgRef,
    ArgumentSpec,
    CLIConfig,
    CurveAggregate,
    CurveAggregationSpec,
    CurveMetricSpec,
    FinalAggregationSpec,
    FinalMetricAggregate,
    FinalMetricSpec,
    SeedValues,
    VariantPayloads,
    VariantSpec,
    manifest_run_complete,
)
from utils.ablation.plotting import save_ablation_figure
from utils.io_utils import load_evaluation_history, load_evaluation_metrics
from utils.plot_utils import (
    SCI_LINE_WIDTH,
    SCI_SERIES_LINE_STYLES,
    VIS_DSPL_MAGENTA,
    VIS_PPO_GRAY,
    VIS_PROPOSED_ORANGE,
    VIS_SAFE_BLUE,
    add_panel_label,
    apply_sci_curve_style,
    apply_sci_figure_layout,
    apply_sci_grid,
    sci_tint_color,
)

METHOD_ABLATION_MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 2
PROTOCOL_VERSION = 11
DEFAULT_OUTPUT_ROOT = "output/paper_experiment/02_method_ablation"
METHOD_FIGURE_FILENAMES = (
    "method_training_curves.pdf",
    "method_trajectory_metrics.pdf",
)
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
METHOD_TRAINING_ROLLOUTS = 400
METHOD_TRAINING_STEPS = METHOD_TRAINING_ROLLOUTS * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
EVALUATION_SMOOTHING_WINDOW = 5
_METHOD_COLORS = {
    "ppo": VIS_PPO_GRAY,
    "ppo_safety": VIS_SAFE_BLUE,
    "ppo_punctuality": VIS_DSPL_MAGENTA,
    "ppo_pirs": VIS_PROPOSED_ORANGE,
}
_METHOD_STYLE_BY_ID = {
    method_id: {"color": _METHOD_COLORS[method_id], **SCI_SERIES_LINE_STYLES[index]}
    for index, method_id in enumerate(_METHOD_COLORS)
}


def _plot_method_curve(
    axis: plt.Axes,
    aggregate: CurveAggregate,
    key: str,
    *,
    label: str | None = None,
) -> None:
    style = _METHOD_STYLE_BY_ID[aggregate.variant_id]
    x_values = aggregate.axis_for(key)
    axis.plot(
        x_values,
        aggregate.means[key],
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=2,
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.7,
        linewidth=(
            SCI_LINE_WIDTH + 0.4
            if aggregate.variant_id == "ppo_pirs"
            else SCI_LINE_WIDTH
        ),
        label=aggregate.label if label is None else label,
    )
    axis.fill_between(
        x_values,
        aggregate.means[key] - aggregate.stds[key],
        aggregate.means[key] + aggregate.stds[key],
        color=sci_tint_color(style["color"]),
        linewidth=0,
        where=np.isfinite(aggregate.means[key]) & np.isfinite(aggregate.stds[key]),
    )


def _format_transition_axis(axis: plt.Axes) -> None:
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)


def _method(
    name: str,
    label: str,
    reward_preset: str,
    color: str,
) -> VariantSpec:
    return VariantSpec(
        id=name,
        label=label,
        color=color,
        manifest={
            "name": name,
            "label": label,
            "reward_preset": reward_preset,
            "color": color,
            "reward_config": reward_config_parameters(
                resolve_reward_preset(reward_preset).config
            ),
        },
        training={
            "reward_preset": reward_preset,
        },
    )


METHODS = (
    _method("ppo", "PPO", "basic", VIS_PPO_GRAY),
    _method("ppo_safety", "PPO+Safety", "basic_safety", VIS_SAFE_BLUE),
    _method(
        "ppo_punctuality", "PPO+Punctuality", "basic_punctuality", VIS_DSPL_MAGENTA
    ),
    _method("ppo_pirs", "PPO+PIRS", "basic_safety_punctuality", VIS_PROPOSED_ORANGE),
)


SPEC = AblationSpec(
    matrix_id="method",
    manifest_filename=METHOD_ABLATION_MANIFEST_FILENAME,
    default_output_root=DEFAULT_OUTPUT_ROOT,
    variants=METHODS,
    seeds=DEFAULT_SEEDS,
    cli=CLIConfig(
        description="Run method ablation experiments.",
        train_help="Train all methods and collect data.",
        show_help="Aggregate and plot method-ablation data.",
        train_arguments=(
            ArgumentSpec(("--output-root",), {"default": DEFAULT_OUTPUT_ROOT}),
            ArgumentSpec(
                ("--schedule-time-s",),
                {"type": float, "default": DEFAULT_SCHEDULE_TIME_S},
            ),
            ArgumentSpec(
                ("--step-distance",),
                {"type": float, "default": DEFAULT_STEP_DISTANCE},
            ),
            ArgumentSpec(
                ("--reward-discount",),
                {"type": float, "default": DEFAULT_REWARD_DISCOUNT},
            ),
            ArgumentSpec(("--num-envs",), {"type": int, "default": DEFAULT_NUM_ENVS}),
            ArgumentSpec(
                ("--evaluation-interval-rollouts",),
                {
                    "type": int,
                    "default": DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
                },
            ),
            ArgumentSpec(("--device",), {"default": DEFAULT_DEVICE}),
            ArgumentSpec(
                ("--resume",),
                {
                    "action": "store_true",
                    "help": (
                        "Resume a compatible manifest and skip runs with complete "
                        "final artifacts."
                    ),
                },
            ),
            ArgumentSpec(
                ("--force-new",),
                {
                    "action": "store_true",
                    "help": (
                        "Archive an existing manifest and start a fresh matrix; "
                        "cannot be combined with --resume."
                    ),
                },
            ),
            ArgumentSpec(
                ("--dry-run",),
                {"action": argparse.BooleanOptionalAction, "default": False},
            ),
        ),
        show_arguments=(
            ArgumentSpec(("--output-root",), {"default": DEFAULT_OUTPUT_ROOT}),
            ArgumentSpec(
                ("--figure-output-dir",),
                {
                    "type": Path,
                    "default": None,
                    "help": "Directory for the two fixed-name method figures.",
                },
            ),
            ArgumentSpec(
                ("--table-output-dir",),
                {
                    "type": Path,
                    "default": None,
                    "help": "Directory for the markdown tables.",
                },
            ),
            ArgumentSpec(
                ("--summary-output-file",),
                {
                    "type": Path,
                    "default": None,
                    "help": "Output path for the json summary.",
                },
            ),
            ArgumentSpec(("--no-show",), {"action": "store_true"}),
            ArgumentSpec(
                ("--dry-run",),
                {"action": argparse.BooleanOptionalAction, "default": False},
            ),
        ),
    ),
    run_id_template="method__{variant_id}__seed{seed:04d}__r{repeat_number:02d}",
    experiment_tag_template="{variant_id}__r{repeat_number:02d}",
    matrix_config={
        "protocol_version": PROTOCOL_VERSION,
        "variants": VariantPayloads(),
        "seeds": SeedValues(),
    },
    training_signature={
        "protocol_version": PROTOCOL_VERSION,
        "budget_mode": "environment_steps",
        "training_rollouts": METHOD_TRAINING_ROLLOUTS,
        "training_steps": METHOD_TRAINING_STEPS,
        "learning_rate_schedule": learning_rate_schedule_parameters(
            "environment_steps"
        ),
        "schedule_time_s": ArgRef("schedule_time_s", float),
        "step_distance": ArgRef("step_distance", float),
        "reward_discount": ArgRef("reward_discount", float),
        "num_envs": ArgRef("num_envs", int),
        "rollout_steps_per_update": DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        "evaluation_interval_rollouts": ArgRef("evaluation_interval_rollouts", int),
        "device": ArgRef("device", str),
        "enable_best_evaluation_artifacts": True,
    },
    training_overrides={
        "budget_mode": "environment_steps",
        "training_rollouts": METHOD_TRAINING_ROLLOUTS,
        "training_episodes": None,
        "num_envs": ArgRef("num_envs", int),
        "rollout_steps_per_update": DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        "enable_best_evaluation_artifacts": True,
        "evaluation_interval_rollouts": ArgRef("evaluation_interval_rollouts", int),
    },
    curve=CurveAggregationSpec(
        episode_reader="series",
        metrics=(
            CurveMetricSpec(
                "stop_error_m",
                "evaluation",
                "stop_error_m",
                "training_steps",
                transform="abs",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "abs_time_error_s",
                "evaluation",
                "time_error_s",
                "training_steps",
                transform="abs",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "total_energy_kwh",
                "evaluation",
                "total_energy_j",
                "training_steps",
                transform="j_to_kwh",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "comfort_tav",
                "evaluation",
                "comfort_tav",
                "training_steps",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "ep_reward",
                "evaluation",
                "total_reward",
                "training_steps",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "ep_len",
                "evaluation",
                "episode_steps",
                "training_steps",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "success_rate",
                "evaluation",
                "success",
                "training_steps",
                transform="bool",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "safe_rate",
                "evaluation",
                "safe",
                "training_steps",
                transform="bool",
                smooth=True,
                alignment="exact_union",
            ),
        ),
        primary_metric="stop_error_m",
        x_name="training_steps",
        default_smoothing_window=EVALUATION_SMOOTHING_WINDOW,
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m", transform="abs"),
            FinalMetricSpec("abs_time_error_s", "time_error_s", transform="abs"),
            FinalMetricSpec("total_energy_kwh", "total_energy_j", transform="j_to_kwh"),
            FinalMetricSpec("comfort_tav", "comfort_tav"),
        ),
        source="best",
    ),
    run_label_template="method={name} seed={seed} output={output_dir}",
    schema_version=MANIFEST_VERSION,
)

DRIVER = AblationDriver(SPEC)
MethodSpec = VariantSpec
MethodRun = AblationRun
build_arg_parser = DRIVER.build_arg_parser
resolve_run_matrix = DRIVER.resolve_runs
build_manifest = DRIVER.build_manifest
_manifest_store = DRIVER.manifest_store
load_manifest = DRIVER.load_manifest
_validate_manifest_compatibility = DRIVER.validate_manifest
build_curve_aggregates = DRIVER.build_curve_aggregates
build_final_aggregates = DRIVER.build_final_aggregates


def _plot_method_trajectory_metrics(
    aggregates: list[CurveAggregate],
) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_sci_curve_style()
    fig, axes = plt.subplots(2, 2)
    panels = (
        (axes[0, 0], "stop_error_m", "Absolute stop error (m)", "(a)"),
        (axes[0, 1], "abs_time_error_s", "Absolute time error (s)", "(b)"),
        (axes[1, 0], "total_energy_kwh", "Trajectory energy (kWh)", "(c)"),
        (
            axes[1, 1],
            "comfort_tav",
            "Cumulative acceleration variation (m/s²)",
            "(d)",
        ),
    )
    for axis, key, ylabel, panel in panels:
        for aggregate in aggregates:
            _plot_method_curve(axis, aggregate, key)
        axis.set_xlabel("Environment transitions")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0, right=METHOD_TRAINING_STEPS)
        _format_transition_axis(axis)
        axis.set_ylim(bottom=0)
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=panel)

    axes[0, 0].axhline(
        0.3,
        color="#666666",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )
    axes[0, 1].axhline(
        10.0,
        color="#666666",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )

    # Inset for stop error (axes[0, 0])
    inset_a = axes[0, 0].inset_axes((0.50, 0.48, 0.46, 0.46))
    for aggregate in aggregates:
        _plot_method_curve(inset_a, aggregate, "stop_error_m", label="_nolegend_")
    inset_a.set_xlim(
        300 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        396 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    )
    inset_a.set_ylim(0.0, 1.5)
    inset_a.axhline(0.3, color="#666666", linestyle="--", linewidth=0.8, zorder=0)
    apply_sci_grid(inset_a)
    inset_a.tick_params(labelsize=7)
    _format_transition_axis(inset_a)

    # Inset for time error (axes[0, 1])
    inset_b = axes[0, 1].inset_axes((0.50, 0.48, 0.46, 0.46))
    for aggregate in aggregates:
        _plot_method_curve(inset_b, aggregate, "abs_time_error_s", label="_nolegend_")
    inset_b.set_xlim(
        300 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        396 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    )
    inset_b.set_ylim(0.0, 30.0)
    inset_b.axhline(10.0, color="#666666", linestyle="--", linewidth=0.8, zorder=0)
    apply_sci_grid(inset_b)
    inset_b.tick_params(labelsize=7)
    _format_transition_axis(inset_b)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
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
    return fig


_plot_learning_curves = _plot_method_trajectory_metrics


def _collect_training_diagnostics(
    manifest: AblationManifest,
) -> dict[str, Any]:
    evaluation_interval = int(
        manifest.training_signature["evaluation_interval_rollouts"]
    )
    expected_rollouts = np.arange(
        evaluation_interval,
        METHOD_TRAINING_ROLLOUTS,
        evaluation_interval,
        dtype=np.int64,
    )
    num_points = len(expected_rollouts)
    periods = ((1, 100), (101, 200), (201, 300), (301, 400))
    window_transitions = float(evaluation_interval * DEFAULT_ROLLOUT_STEPS_PER_UPDATE)

    result: dict[str, Any] = {}
    for method in METHODS:
        runs = [r for r in manifest.runs if r.variant_id == method.id]
        runs.sort(key=lambda r: r.seed)
        seed_rates = []
        seed_arrivals = []
        period_stats: dict[str, dict[str, float | int]] = {}

        period_low_rates: dict[str, list[float]] = {
            f"{p0}-{p1}": [] for p0, p1 in periods
        }
        period_high_rates: dict[str, list[float]] = {
            f"{p0}-{p1}": [] for p0, p1 in periods
        }
        period_total_rates: dict[str, list[float]] = {
            f"{p0}-{p1}": [] for p0, p1 in periods
        }
        period_arrival_ratios: dict[str, list[float]] = {
            f"{p0}-{p1}": [] for p0, p1 in periods
        }
        period_low_counts: dict[str, int] = {f"{p0}-{p1}": 0 for p0, p1 in periods}
        period_high_counts: dict[str, int] = {f"{p0}-{p1}": 0 for p0, p1 in periods}
        period_total_counts: dict[str, int] = {f"{p0}-{p1}": 0 for p0, p1 in periods}
        period_arrived_counts: dict[str, int] = {f"{p0}-{p1}": 0 for p0, p1 in periods}
        period_completed_counts: dict[str, int] = {
            f"{p0}-{p1}": 0 for p0, p1 in periods
        }

        for run in runs:
            ep_file = Path(run.artifacts.path_for("episodes"))
            ep_data = np.load(ep_file)
            complete = np.asarray(ep_data["episode_complete"], dtype=np.bool_)
            truncated = np.asarray(ep_data["episode_truncated"], dtype=np.bool_)
            terminated = np.asarray(ep_data["episode_terminated"], dtype=np.bool_)
            code = np.asarray(ep_data["episode_violation_code"], dtype=np.int8)
            end_steps = np.asarray(ep_data["episode_end_step"], dtype=np.int64)

            run_rates = np.zeros(num_points, dtype=np.float64)
            run_arrivals = np.zeros(num_points, dtype=np.float64)
            for idx, r_end in enumerate(expected_rollouts):
                s_end = r_end * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
                s_start = (
                    r_end - evaluation_interval
                ) * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
                mask = complete & (end_steps > s_start) & (end_steps <= s_end)
                viol = np.sum(truncated & ((code == 2) | (code == 3)) & mask)
                run_rates[idx] = (float(viol) / window_transitions) * 10000.0

                comp_cnt = np.sum(mask)
                arr_cnt = np.sum(terminated & (~truncated) & mask)
                run_arrivals[idx] = (
                    float(arr_cnt) / float(comp_cnt) if comp_cnt > 0 else 0.0
                )

            seed_rates.append(run_rates)
            seed_arrivals.append(run_arrivals)

            for p0, p1 in periods:
                pkey = f"{p0}-{p1}"
                s_p_start = (p0 - 1) * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
                s_p_end = p1 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
                p_trans = float((p1 - p0 + 1) * DEFAULT_ROLLOUT_STEPS_PER_UPDATE)
                p_mask = complete & (end_steps > s_p_start) & (end_steps <= s_p_end)

                low_cnt = int(np.sum(truncated & (code == 2) & p_mask))
                high_cnt = int(np.sum(truncated & (code == 3) & p_mask))
                tot_cnt = low_cnt + high_cnt
                p_comp = int(np.sum(p_mask))
                p_arr = int(np.sum(terminated & (~truncated) & p_mask))

                period_low_rates[pkey].append((low_cnt / p_trans) * 10000.0)
                period_high_rates[pkey].append((high_cnt / p_trans) * 10000.0)
                period_total_rates[pkey].append((tot_cnt / p_trans) * 10000.0)
                period_arrival_ratios[pkey].append(
                    float(p_arr) / float(p_comp) if p_comp > 0 else 0.0
                )

                period_low_counts[pkey] += low_cnt
                period_high_counts[pkey] += high_cnt
                period_total_counts[pkey] += tot_cnt
                period_arrived_counts[pkey] += p_arr
                period_completed_counts[pkey] += p_comp

        rates_matrix = np.asarray(seed_rates, dtype=np.float64)
        arrivals_matrix = np.asarray(seed_arrivals, dtype=np.float64)

        for p0, p1 in periods:
            pkey = f"{p0}-{p1}"
            n_runs = len(runs)
            tot_p_trans = int((p1 - p0 + 1) * DEFAULT_ROLLOUT_STEPS_PER_UPDATE * n_runs)
            low_r = np.asarray(period_low_rates[pkey], dtype=np.float64)
            high_r = np.asarray(period_high_rates[pkey], dtype=np.float64)
            tot_r = np.asarray(period_total_rates[pkey], dtype=np.float64)
            arr_r = np.asarray(period_arrival_ratios[pkey], dtype=np.float64)
            comp_c = period_completed_counts[pkey]
            arr_c = period_arrived_counts[pkey]

            period_stats[pkey] = {
                "rate_low_mean": float(np.mean(low_r)),
                "rate_low_std": (
                    float(np.std(low_r, ddof=1)) if len(low_r) > 1 else 0.0
                ),
                "rate_high_mean": float(np.mean(high_r)),
                "rate_high_std": (
                    float(np.std(high_r, ddof=1)) if len(high_r) > 1 else 0.0
                ),
                "rate_total_mean": float(np.mean(tot_r)),
                "rate_total_std": (
                    float(np.std(tot_r, ddof=1)) if len(tot_r) > 1 else 0.0
                ),
                "arrival_ratio_mean": float(np.mean(arr_r)),
                "arrival_ratio_std": (
                    float(np.std(arr_r, ddof=1)) if len(arr_r) > 1 else 0.0
                ),
                "count_low": period_low_counts[pkey],
                "count_high": period_high_counts[pkey],
                "count_total": period_total_counts[pkey],
                "transitions": tot_p_trans,
                "count_completed": comp_c,
                "count_arrived": arr_c,
                "overall_arrival_pct": (
                    (float(arr_c) / float(comp_c) * 100.0) if comp_c > 0 else 0.0
                ),
            }

        result[method.id] = {
            "speed_violation_rate_means": np.mean(rates_matrix, axis=0),
            "speed_violation_rate_stds": (
                np.std(rates_matrix, axis=0, ddof=1)
                if len(runs) > 1
                else np.zeros(num_points, dtype=np.float64)
            ),
            "arrival_ratio_means": np.mean(arrivals_matrix, axis=0),
            "arrival_ratio_stds": (
                np.std(arrivals_matrix, axis=0, ddof=1)
                if len(runs) > 1
                else np.zeros(num_points, dtype=np.float64)
            ),
            "periods": period_stats,
        }
    return result


def _plot_method_training_curves(
    training_data: dict[str, Any],
    steps: NDArray[np.float64],
) -> plt.Figure | None:
    if not training_data:
        return None
    apply_sci_curve_style()
    fig, axes = plt.subplots(1, 2)
    for method in METHODS:
        data = training_data[method.id]
        style = _METHOD_STYLE_BY_ID[method.id]
        lw = SCI_LINE_WIDTH + 0.4 if method.id == "ppo_pirs" else SCI_LINE_WIDTH
        rate_m = data["speed_violation_rate_means"]
        rate_s = data["speed_violation_rate_stds"]
        axes[0].plot(
            steps,
            rate_m,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=lw,
            label=method.label,
        )
        axes[0].fill_between(
            steps,
            np.maximum(0.0, rate_m - rate_s),
            rate_m + rate_s,
            color=sci_tint_color(style["color"]),
            linewidth=0,
            where=np.isfinite(rate_m) & np.isfinite(rate_s),
        )

        arr_m = data["arrival_ratio_means"]
        arr_s = data["arrival_ratio_stds"]
        axes[1].plot(
            steps,
            arr_m,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=lw,
            label=method.label,
        )
        axes[1].fill_between(
            steps,
            np.clip(arr_m - arr_s, 0.0, 1.0),
            np.clip(arr_m + arr_s, 0.0, 1.0),
            color=sci_tint_color(style["color"]),
            linewidth=0,
            where=np.isfinite(arr_m) & np.isfinite(arr_s),
        )

    # Inset for late-stage speed violations in subplot (a) (rollouts 200–396, y: 0–5)
    inset_a = axes[0].inset_axes((0.48, 0.46, 0.48, 0.48))
    for method in METHODS:
        data = training_data[method.id]
        style = _METHOD_STYLE_BY_ID[method.id]
        lw = SCI_LINE_WIDTH + 0.4 if method.id == "ppo_pirs" else SCI_LINE_WIDTH
        rate_m = data["speed_violation_rate_means"]
        rate_s = data["speed_violation_rate_stds"]
        inset_a.plot(
            steps,
            rate_m,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=lw,
            label="_nolegend_",
        )
        inset_a.fill_between(
            steps,
            np.maximum(0.0, rate_m - rate_s),
            rate_m + rate_s,
            color=sci_tint_color(style["color"]),
            linewidth=0,
            where=np.isfinite(rate_m) & np.isfinite(rate_s),
        )
    inset_a.set_xlim(
        200 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        396 * DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    )
    inset_a.set_ylim(0.0, 5.0)
    apply_sci_grid(inset_a)
    inset_a.tick_params(labelsize=7)
    _format_transition_axis(inset_a)

    axes[0].set_xlabel("Environment transitions")
    axes[0].set_ylabel("Speed violations / 10⁴ transitions")
    axes[0].set_xlim(left=0, right=METHOD_TRAINING_STEPS)
    axes[0].set_ylim(bottom=0.0)
    _format_transition_axis(axes[0])
    apply_sci_grid(axes[0])
    add_panel_label(ax=axes[0], label="(a)")

    axes[1].set_xlabel("Environment transitions")
    axes[1].set_ylabel("Training arrival rate")
    axes[1].set_xlim(left=0, right=METHOD_TRAINING_STEPS)
    axes[1].set_ylim(-0.03, 1.03)
    _format_transition_axis(axes[1])
    apply_sci_grid(axes[1])
    add_panel_label(ax=axes[1], label="(b)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=2.85,
        left=0.11,
        bottom=0.18,
        top=0.90,
        wspace=0.24,
    )
    return fig


def _render_method_training_table(training_data: dict[str, Any]) -> str:
    lines = [
        "# Method Ablation Training Process Table",
        "",
        (
            "| Method | Rollouts | Low Viol. Rate (/10⁴) | High Viol. Rate (/10⁴) | "
            "Total Viol. Rate (/10⁴) | Arrival Ratio | Low Violations | "
            "High Violations | Total Violations | Transitions | Arrivals / Completed |"
        ),
        ("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"),
    ]
    periods = (
        (1, 100),
        (101, 200),
        (201, 300),
        (301, 400),
    )
    for method in METHODS:
        for p_start, p_end in periods:
            pkey = f"{p_start}-{p_end}"
            p_data = training_data[method.id]["periods"][pkey]
            rate_low_str = (
                f"{p_data['rate_low_mean']:.2f} ± {p_data['rate_low_std']:.2f}"
            )
            rate_high_str = (
                f"{p_data['rate_high_mean']:.2f} ± {p_data['rate_high_std']:.2f}"
            )
            rate_total_str = (
                f"{p_data['rate_total_mean']:.2f} ± {p_data['rate_total_std']:.2f}"
            )
            arr_ratio_str = (
                f"{p_data['arrival_ratio_mean'] * 100.0:.2f}% ± "
                f"{p_data['arrival_ratio_std'] * 100.0:.2f}%"
            )
            arr_counts_str = (
                f"{p_data['count_arrived']} / {p_data['count_completed']} "
                f"({p_data['overall_arrival_pct']:.2f}%)"
            )
            cells = [
                method.label,
                pkey,
                rate_low_str,
                rate_high_str,
                rate_total_str,
                arr_ratio_str,
                str(p_data["count_low"]),
                str(p_data["count_high"]),
                str(p_data["count_total"]),
                f"{p_data['transitions']:,}",
                arr_counts_str,
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "*Note: Statistics are aggregated across 4 stages "
                "(100 rollouts = 819,200 transitions each, totaling "
                "4,096,000 transitions across 5 seeds). "
                "Violation rate is the number of occurrences per 10k "
                "environment transitions; arrival rate is the proportion "
                "of completed episodes that reach the terminal. "
                "Mean and sample standard deviation (ddof=1) are calculated "
                "across 5 seeds. Cumulative violation counts and arrival "
                "episode numerators/denominators are also shown.*"
            ),
        ]
    )
    return "\n".join(lines)


def _compute_method_feasible_stats(
    manifest: AblationManifest,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        runs = [r for r in manifest.runs if r.variant_id == method.id]
        feasibles: list[bool] = []
        for run in runs:
            metrics_path = Path(run.artifacts.path_for("metrics_best"))
            metrics = load_evaluation_metrics(metrics_path)
            feasibles.append(bool(metrics.feasible))
        count = sum(feasibles)
        total = len(feasibles)
        rate = count / total if total > 0 else 0.0
        stats[method.id] = {
            "feasible_count": count,
            "total": total,
            "feasible_rate": rate,
            "rate_str": f"{rate * 100:.1f}% ({count}/{total})",
        }
    return stats


def _render_method_performance_table(
    finals: list[FinalMetricAggregate],
    manifest: AblationManifest,
) -> str:
    feasible_stats = _compute_method_feasible_stats(manifest)
    lines = [
        "# Method Ablation Performance Table",
        "",
        (
            "| Method | Strict feasibility rate | Stop error (m) | Time error (s) | "
            "Total energy (kWh) | Cumulative acceleration variation (m/s²) |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for aggregate in finals:
        method_id = aggregate.variant_id
        feas_str = feasible_stats.get(method_id, {}).get("rate_str", "0.0% (0/5)")
        stop_err_str = (
            f"{aggregate.means['stop_error_m']:.6f} ± "
            f"{aggregate.stds['stop_error_m']:.6f}"
        )
        time_err_str = (
            f"{aggregate.means['abs_time_error_s']:.6f} ± "
            f"{aggregate.stds['abs_time_error_s']:.6f}"
        )
        energy_str = (
            f"{aggregate.means['total_energy_kwh']:.6f} ± "
            f"{aggregate.stds['total_energy_kwh']:.6f}"
        )
        comfort_str = (
            f"{aggregate.means['comfort_tav']:.6f} ± "
            f"{aggregate.stds['comfort_tav']:.6f}"
        )
        cells = [
            aggregate.label or aggregate.variant_id,
            feas_str,
            stop_err_str,
            time_err_str,
            energy_str,
            comfort_str,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "*Note: Metrics are based on best/independent evaluation "
                "trajectories across 5 random seeds per method "
                "(mean ± sample standard deviation, ddof=1), including "
                "non-conforming runs. Strict feasibility rate is the proportion "
                "of strictly feasible trajectories across 5 seeds. "
                "Energy units are converted from J to kWh. "
                "The cumulative acceleration variation formula is "
                r"$\sum_t |a_t - a_{t-1}|$, with unit $\mathrm{m/s^2}$.*"
            ),
        ]
    )
    return "\n".join(lines)


def _find_representative_policy(manifest: AblationManifest) -> dict[str, Any]:
    pirs_runs = [r for r in manifest.runs if r.variant_id == "ppo_pirs"]
    if not pirs_runs:
        raise ValueError("Manifest contains no ppo_pirs runs")

    candidates = []
    for run in pirs_runs:
        metrics_path = Path(run.artifacts.path_for("metrics_best"))
        metrics = load_evaluation_metrics(metrics_path)
        candidates.append((run, metrics))

    candidates.sort(
        key=lambda pair: (pair[1].selection_comparison_key, -pair[0].seed),
        reverse=True,
    )
    best_run, best_metrics = candidates[0]
    return {
        "variant_id": best_run.variant_id,
        "seed": best_run.seed,
        "run_id": best_run.run_id,
        "model_path": str(best_run.artifacts.path_for("policy_best")),
        "stop_error_m": abs(float(best_metrics.stop_error_m)),
        "time_error_s": abs(float(best_metrics.time_error_s)),
        "total_energy_kwh": float(best_metrics.total_energy_j) / 3_600_000.0,
        "comfort_tav": float(best_metrics.comfort_tav),
        "safe": bool(best_metrics.safe),
        "feasible": bool(best_metrics.feasible),
        "selection_comparison_key": list(best_metrics.selection_comparison_key),
    }


def _print_representative_policy(rep_info: dict[str, Any]) -> None:
    print("=" * 60)
    print("Recommended Representative PPO+PIRS Policy:")
    print(f"  Model path: {rep_info['model_path']}")
    print(f"  Seed: {rep_info['seed']}")
    print(f"  Stop error: {rep_info['stop_error_m']:.6f} m")
    print(f"  Time error: {rep_info['time_error_s']:.6f} s")
    print(f"  Total energy: {rep_info['total_energy_kwh']:.6f} kWh")
    print(f"  Comfort (TAV): {rep_info['comfort_tav']:.6f} m/s²")
    print(f"  Safe: {rep_info['safe']}")
    print(f"  Feasible: {rep_info['feasible']}")
    print("=" * 60)


def _build_method_summary(
    manifest: AblationManifest,
    finals: list[FinalMetricAggregate],
    rep_info: dict[str, Any],
) -> dict[str, Any]:
    raw_seed_metrics: dict[str, list[dict[str, Any]]] = {m.id: [] for m in METHODS}
    for method in METHODS:
        runs = [r for r in manifest.runs if r.variant_id == method.id]
        runs.sort(key=lambda r: r.seed)
        for run in runs:
            metrics_path = Path(run.artifacts.path_for("metrics_best"))
            metrics = load_evaluation_metrics(metrics_path)
            raw_seed_metrics[method.id].append(
                {
                    "run_id": run.run_id,
                    "seed": run.seed,
                    "stop_error_m": abs(float(metrics.stop_error_m)),
                    "time_error_s": abs(float(metrics.time_error_s)),
                    "total_energy_j": float(metrics.total_energy_j),
                    "total_energy_kwh": float(metrics.total_energy_j) / 3_600_000.0,
                    "comfort_tav": float(metrics.comfort_tav),
                    "success": bool(metrics.success),
                    "safe": bool(metrics.safe),
                    "feasible": bool(metrics.feasible),
                    "selection_comparison_key": list(metrics.selection_comparison_key),
                }
            )

    seed_lookup = {
        (m_id, item["seed"]): item
        for m_id, items in raw_seed_metrics.items()
        for item in items
    }

    paired_differences: dict[str, Any] = {}
    for method in ("ppo_safety", "ppo_punctuality", "ppo_pirs"):
        diff_key = f"{method}_minus_ppo"
        by_seed: dict[str, dict[str, float]] = {}
        stop_diffs = []
        time_diffs = []
        energy_diffs = []
        comfort_diffs = []
        for seed in DEFAULT_SEEDS:
            curr = seed_lookup[(method, seed)]
            base = seed_lookup[("ppo", seed)]
            d_stop = curr["stop_error_m"] - base["stop_error_m"]
            d_time = curr["time_error_s"] - base["time_error_s"]
            d_energy = curr["total_energy_kwh"] - base["total_energy_kwh"]
            d_comfort = curr["comfort_tav"] - base["comfort_tav"]
            by_seed[str(seed)] = {
                "stop_error_m": d_stop,
                "abs_time_error_s": d_time,
                "total_energy_kwh": d_energy,
                "comfort_tav": d_comfort,
            }
            stop_diffs.append(d_stop)
            time_diffs.append(d_time)
            energy_diffs.append(d_energy)
            comfort_diffs.append(d_comfort)

        n = len(DEFAULT_SEEDS)
        paired_differences[diff_key] = {
            "by_seed": by_seed,
            "mean": {
                "stop_error_m": float(np.mean(stop_diffs)),
                "abs_time_error_s": float(np.mean(time_diffs)),
                "total_energy_kwh": float(np.mean(energy_diffs)),
                "comfort_tav": float(np.mean(comfort_diffs)),
            },
            "std": {
                "stop_error_m": float(np.std(stop_diffs, ddof=1)) if n > 1 else 0.0,
                "abs_time_error_s": (
                    float(np.std(time_diffs, ddof=1)) if n > 1 else 0.0
                ),
                "total_energy_kwh": (
                    float(np.std(energy_diffs, ddof=1)) if n > 1 else 0.0
                ),
                "comfort_tav": float(np.std(comfort_diffs, ddof=1)) if n > 1 else 0.0,
            },
        }

    feasible_stats = _compute_method_feasible_stats(manifest)
    feasible_summary = {
        m_id: {
            "feasible_count": data["feasible_count"],
            "total_runs": data["total"],
            "feasible_rate": data["feasible_rate"],
        }
        for m_id, data in feasible_stats.items()
    }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "budget_mode": "environment_steps",
        "training_rollouts": METHOD_TRAINING_ROLLOUTS,
        "training_steps": METHOD_TRAINING_STEPS,
        "seeds": list(DEFAULT_SEEDS),
        "methods": [m.id for m in METHODS],
        "feasible_summary": feasible_summary,
        "raw_seed_metrics": raw_seed_metrics,
        "paired_differences": paired_differences,
        "representative_policy": {
            "variant_id": rep_info["variant_id"],
            "seed": rep_info["seed"],
            "run_id": rep_info["run_id"],
            "model_path": rep_info["model_path"],
            "stop_error_m": rep_info["stop_error_m"],
            "time_error_s": rep_info["time_error_s"],
            "total_energy_kwh": rep_info["total_energy_kwh"],
            "comfort_tav": rep_info["comfort_tav"],
            "safe": rep_info["safe"],
            "feasible": rep_info["feasible"],
        },
    }


def _print_final_table(
    aggregates: list[FinalMetricAggregate],
    manifest: AblationManifest | None = None,
) -> None:
    feasible_stats = _compute_method_feasible_stats(manifest) if manifest else {}
    columns = (
        "method",
        "feasible_rate",
        "stop_error_m",
        "abs_time_error_s",
        "total_energy_kwh",
        "comfort_tav",
    )
    print("Best-evaluation summary (mean±std):")
    print(" | ".join(columns))
    for aggregate in aggregates:
        feas_str = feasible_stats.get(aggregate.variant_id, {}).get("rate_str", "N/A")
        cells = [
            aggregate.label or aggregate.variant_id,
            feas_str,
            *(
                f"{aggregate.means[key]:.6f}±{aggregate.stds[key]:.6f}"
                for key in columns[2:]
            ),
        ]
        print(" | ".join(cells))
        print(
            f"  feasible_rate={feas_str}, "
            f"success_rate={aggregate.success_rate:.3f}, "
            f"runs={aggregate.valid_run_count}"
        )


def _print_constraint_table(manifest: AblationManifest) -> None:
    print("Best-evaluation constraint rates:")
    print("method | success | precise | punctual | safe | feasible | n")
    for method in METHODS:
        assessments = []
        for run in manifest.runs:
            if run.variant_id == method.id:
                metrics = load_evaluation_metrics(
                    Path(run.artifacts.path_for("metrics_best"))
                )
                assessments.append(assess_constraints(metrics))
        n = len(assessments)
        fields = (
            "success",
            "precise_arrival",
            "punctual_arrival",
            "safe",
            "feasible",
        )
        rates = [
            sum(bool(getattr(item, field)) for item in assessments) / n
            for field in fields
        ]
        print(
            f"{method.label} | "
            + " | ".join(f"{value:.3f}" for value in rates)
            + f" | {n}"
        )


def validate_method_ablation_manifest(manifest: AblationManifest) -> None:
    if manifest.matrix_config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "method-ablation manifest uses an obsolete protocol; rerun it in a "
            "new YYYYMMDD_NN experiment directory"
        )
    expected_variants = [dict(method.manifest) for method in METHODS]
    if manifest.matrix_config.get("variants") != expected_variants:
        raise ValueError(
            "method-ablation manifest variant/reward matrix is incompatible"
        )
    if manifest.matrix_config.get("seeds") != list(DEFAULT_SEEDS):
        raise ValueError("method-ablation manifest seed matrix is incompatible")
    expected_training = {
        "budget_mode": "environment_steps",
        "training_rollouts": METHOD_TRAINING_ROLLOUTS,
        "training_steps": METHOD_TRAINING_STEPS,
        "rollout_steps_per_update": DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        "learning_rate_schedule": learning_rate_schedule_parameters(
            "environment_steps"
        ),
    }
    for key, value in expected_training.items():
        if manifest.training_signature.get(key) != value:
            raise ValueError(
                f"method-ablation training signature {key} is incompatible"
            )
    num_envs = manifest.training_signature.get("num_envs")
    if (
        not isinstance(num_envs, int)
        or num_envs <= 0
        or DEFAULT_ROLLOUT_STEPS_PER_UPDATE % num_envs != 0
    ):
        raise ValueError("method-ablation num_envs is incompatible")
    evaluation_interval = manifest.training_signature.get(
        "evaluation_interval_rollouts"
    )
    if (
        not isinstance(evaluation_interval, int)
        or not 1 <= evaluation_interval < METHOD_TRAINING_ROLLOUTS
    ):
        raise ValueError("method-ablation evaluation_interval_rollouts is incompatible")
    expected = {
        f"method__{method.id}__seed{seed:04d}__r{index + 1:02d}"
        for method in METHODS
        for index, seed in enumerate(DEFAULT_SEEDS)
    }
    actual = {run.run_id for run in manifest.runs}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"method-ablation run matrix mismatch: missing={missing}, extra={extra}"
        )
    incomplete = [run.run_id for run in manifest.runs if not manifest_run_complete(run)]
    if incomplete:
        raise ValueError(
            "method-ablation analysis requires completed budgets and canonical "
            f"artifacts for every run; invalid={incomplete}"
        )
    expected_rollouts = np.arange(
        evaluation_interval,
        METHOD_TRAINING_ROLLOUTS,
        evaluation_interval,
        dtype=np.int64,
    )
    expected_steps = (expected_rollouts * DEFAULT_ROLLOUT_STEPS_PER_UPDATE).astype(
        np.float64
    )
    for run in manifest.runs:
        history = load_evaluation_history(run.artifacts.path_for("evaluations"))
        if not np.array_equal(history.rollout_indices, expected_rollouts):
            raise ValueError(
                f"Run {run.run_id} periodic evaluation rollout indices do not match "
                f"the {len(expected_rollouts)} scheduled evaluation points"
            )
        if not np.array_equal(history.training_steps, expected_steps):
            raise ValueError(
                f"Run {run.run_id} periodic evaluation training steps do not match "
                f"the {len(expected_rollouts)} scheduled evaluation points"
            )


def run_train(args: argparse.Namespace) -> int:
    if args.num_envs <= 0 or DEFAULT_ROLLOUT_STEPS_PER_UPDATE % args.num_envs != 0:
        raise SystemExit(
            "--num-envs must be a positive divisor of "
            f"{DEFAULT_ROLLOUT_STEPS_PER_UPDATE}"
        )
    if not 1 <= args.evaluation_interval_rollouts < METHOD_TRAINING_ROLLOUTS:
        raise SystemExit("--evaluation-interval-rollouts must be in [1, 399]")
    DRIVER.train_experiment = train_single_experiment
    DRIVER.evaluate_experiment = evaluate_final_training_run
    return DRIVER.run_train(args)


def run_show(args: argparse.Namespace) -> int:
    manifest = DRIVER.load_manifest(args.output_root)
    try:
        validate_method_ablation_manifest(manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    curves, curve_warnings = DRIVER.build_curve_aggregates(manifest)
    finals, final_warnings = DRIVER.build_final_aggregates(manifest)
    warnings = [*curve_warnings, *final_warnings]
    if warnings:
        raise SystemExit(
            "Method-ablation analysis inputs are invalid:\n" + "\n".join(warnings)
        )
    expected_count = len(METHODS)
    if (
        len(curves) != expected_count
        or len(finals) != expected_count
        or any(item.valid_run_count != len(DEFAULT_SEEDS) for item in curves)
        or any(item.valid_run_count != len(DEFAULT_SEEDS) for item in finals)
    ):
        raise SystemExit("Method-ablation analysis refused a partial aggregation")
    expected_rollouts = np.arange(
        manifest.training_signature["evaluation_interval_rollouts"],
        METHOD_TRAINING_ROLLOUTS,
        manifest.training_signature["evaluation_interval_rollouts"],
        dtype=np.int64,
    )
    expected_steps = (expected_rollouts * DEFAULT_ROLLOUT_STEPS_PER_UPDATE).astype(
        np.float64
    )
    for aggregate in curves:
        for metric in (
            "stop_error_m",
            "abs_time_error_s",
            "total_energy_kwh",
            "comfort_tav",
        ):
            stats = aggregate.metrics[metric]
            if (
                not np.array_equal(aggregate.axis_for(metric), expected_steps)
                or not np.all(stats.count == len(DEFAULT_SEEDS))
                or not np.all(np.isfinite(stats.mean))
                or not np.all(np.isfinite(stats.std))
            ):
                raise SystemExit(
                    f"Method-ablation {metric} has missing or invalid evaluations"
                )

    training_data = _collect_training_diagnostics(manifest)
    training_table_md = _render_method_training_table(training_data)
    performance_table_md = _render_method_performance_table(finals, manifest)

    print(training_table_md)
    print()
    print(performance_table_md)
    print()
    _print_final_table(finals, manifest)
    _print_constraint_table(manifest)

    rep_info = _find_representative_policy(manifest)
    _print_representative_policy(rep_info)

    if args.dry_run:
        print(
            f"Dry run completed: verified 20 runs, {len(expected_rollouts)} "
            "evaluation points, "
            "and previewed outputs without plotting or file writes."
        )
        return 0

    if args.table_output_dir is not None:
        table_dir = Path(args.table_output_dir)
        table_dir.mkdir(parents=True, exist_ok=True)
        (table_dir / "method_training_table.md").write_text(
            training_table_md, encoding="utf-8"
        )
        (table_dir / "method_performance_table.md").write_text(
            performance_table_md, encoding="utf-8"
        )
        print(f"Saved tables to: {table_dir}")

    if args.summary_output_file is not None:
        summary_payload = _build_method_summary(manifest, finals, rep_info)
        summary_file = Path(args.summary_output_file)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"Saved summary to: {summary_file}")

    print("Learning curves use all periodic independent evaluations.")

    training_figure = _plot_method_training_curves(training_data, expected_steps)
    trajectory_figure = _plot_method_trajectory_metrics(curves)

    if args.figure_output_dir is not None:
        fig_dir = Path(args.figure_output_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for figure, filename in zip(
            (training_figure, trajectory_figure),
            METHOD_FIGURE_FILENAMES,
            strict=True,
        ):
            saved_path = save_ablation_figure(figure, fig_dir / filename)
            print(f"Saved figure to: {saved_path}")
    if not args.no_show:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_train(args) if args.command == "train" else run_show(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    raise SystemExit(main())
