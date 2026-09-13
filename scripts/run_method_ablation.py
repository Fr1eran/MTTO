"""Train and display the PPO/PPRS/DSPL ablation matrix."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    DSPL_ALGORITHM_ID,
    dspl_protocol_parameters,
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
from utils.io_utils import load_evaluation_metrics
from utils.plot_utils import (
    SCI_BAND_ALPHA,
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
)

METHOD_ABLATION_MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
PROTOCOL_VERSION = 9
DEFAULT_OUTPUT_ROOT = "output/paper_experiment/02_method_ablation"
DEFAULT_SELECTION_FILENAME = "selected_policy.json"
METHOD_FIGURE_FILENAMES = (
    "method_learning_curves.pdf",
    "safety_learning_process.pdf",
    "evaluation_success_rate.pdf",
)
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
METHOD_TRAINING_ROLLOUTS = 400
METHOD_TRAINING_STEPS = METHOD_TRAINING_ROLLOUTS * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
EVALUATION_SMOOTHING_WINDOW = 5
_METHOD_COLORS = {
    "ppo": VIS_PPO_GRAY,
    "ppo_pprs": VIS_SAFE_BLUE,
    "ppo_dspl": VIS_DSPL_MAGENTA,
    "ppo_pprs_dspl": VIS_PROPOSED_ORANGE,
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
            if aggregate.variant_id == "ppo_pprs_dspl"
            else SCI_LINE_WIDTH
        ),
        label=aggregate.label if label is None else label,
    )
    axis.fill_between(
        x_values,
        aggregate.means[key] - aggregate.stds[key],
        aggregate.means[key] + aggregate.stds[key],
        color=style["color"],
        alpha=SCI_BAND_ALPHA,
        linewidth=0,
        where=np.isfinite(aggregate.means[key]) & np.isfinite(aggregate.stds[key]),
    )


def _format_transition_axis(axis: plt.Axes) -> None:
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)


def _method(
    name: str,
    label: str,
    reward_preset: str,
    curriculum_profile: str,
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
            "curriculum_profile": curriculum_profile,
            "color": color,
            "pprs_enabled": reward_preset == "basic_safety_punctuality",
            "curriculum_enabled": curriculum_profile == "dspl",
            "reward_config": reward_config_parameters(
                resolve_reward_preset(reward_preset).config
            ),
        },
        training={
            "reward_preset": reward_preset,
            "curriculum_profile": curriculum_profile,
            "reference_curve_dir": (
                ArgRef("reference_curve_dir") if curriculum_profile != "none" else None
            ),
        },
    )


METHODS = (
    _method("ppo", "PPO", "basic", "none", "#0072B2"),
    _method(
        "ppo_pprs",
        "PPO+PPRS",
        "basic_safety_punctuality",
        "none",
        "#E69F00",
    ),
    _method(
        "ppo_dspl",
        "PPO+DSPL",
        "basic",
        "dspl",
        "#CC79A7",
    ),
    _method(
        "ppo_pprs_dspl",
        "PPO+PPRS+DSPL",
        "basic_safety_punctuality",
        "dspl",
        "#009E73",
    ),
)


SPEC = AblationSpec(
    matrix_id="method",
    manifest_filename=METHOD_ABLATION_MANIFEST_FILENAME,
    default_output_root=DEFAULT_OUTPUT_ROOT,
    variants=METHODS,
    seeds=DEFAULT_SEEDS,
    cli=CLIConfig(
        description="Run PPO/PPRS/DSPL ablation experiments.",
        train_help="Train all methods and collect data.",
        show_help="Aggregate and plot method-ablation data.",
        train_arguments=(
            ArgumentSpec(("--output-root",), {"default": DEFAULT_OUTPUT_ROOT}),
            ArgumentSpec(("--reference-curve-dir",), {"required": True}),
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
                    "help": "Directory for the three fixed-name method figures.",
                },
            ),
            ArgumentSpec(
                ("--selection-output-file",),
                {
                    "type": Path,
                    "default": None,
                    "help": (
                        "Best-policy JSON path; defaults to selected_policy.json "
                        "under the output root."
                    ),
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
        "reference_curve_dir": ArgRef("reference_curve_dir"),
    },
    training_signature={
        "protocol_version": PROTOCOL_VERSION,
        "curriculum_algorithm_id": DSPL_ALGORITHM_ID,
        "dspl_protocol": dspl_protocol_parameters(),
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
        primary_metric="ep_reward",
        x_name="training_steps",
        default_smoothing_window=EVALUATION_SMOOTHING_WINDOW,
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m"),
            FinalMetricSpec("abs_time_error_s", "time_error_s", transform="abs"),
            FinalMetricSpec("total_energy_kj", "total_energy_kj", feasible_only=True),
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


def _plot_learning_curves(
    aggregates: list[CurveAggregate],
) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_sci_curve_style()
    fig, axes = plt.subplots(2, 2)
    for axis, key, ylabel, panel in (
        (axes[0, 0], "ep_reward", "Mean evaluation reward", "(a)"),
        (axes[0, 1], "ep_len", "Mean evaluation episode length", "(b)"),
        (axes[1, 0], "stop_error_m", "Mean absolute stop error (m)", "(c)"),
        (
            axes[1, 1],
            "abs_time_error_s",
            "Mean absolute time error (s)",
            "(d)",
        ),
    ):
        for aggregate in aggregates:
            _plot_method_curve(axis, aggregate, key)
        axis.set_xlabel("Environment transitions")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0, right=METHOD_TRAINING_STEPS)
        _format_transition_axis(axis)
        if key in {"ep_len", "stop_error_m", "abs_time_error_s"}:
            axis.set_ylim(bottom=0)
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=panel)
    axes[0, 1].axhline(
        972,
        color="#666666",
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        zorder=0,
    )
    inset = axes[1, 0].inset_axes((0.53, 0.49, 0.43, 0.43))
    for aggregate in aggregates:
        _plot_method_curve(inset, aggregate, "stop_error_m", label="_nolegend_")
    inset.set_xlim(0.75 * METHOD_TRAINING_STEPS, METHOD_TRAINING_STEPS)
    inset.set_ylim(0, 150)
    apply_sci_grid(inset)
    inset.tick_params(labelsize=6)
    _format_transition_axis(inset)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=4.8,
        left=0.11,
        bottom=0.12,
        top=0.89,
        wspace=0.42,
        hspace=0.42,
    )
    return fig


def _plot_evaluation_success_rate(
    aggregates: list[CurveAggregate],
) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_sci_curve_style()
    fig, axis = plt.subplots()
    for aggregate in aggregates:
        style = _METHOD_STYLE_BY_ID[aggregate.variant_id]
        axis.plot(
            aggregate.axis_for("success_rate"),
            aggregate.means["success_rate"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=(
                SCI_LINE_WIDTH + 0.4
                if aggregate.variant_id == "ppo_pprs_dspl"
                else SCI_LINE_WIDTH
            ),
            label=aggregate.label,
        )
        axis.fill_between(
            aggregate.axis_for("success_rate"),
            np.clip(
                aggregate.means["success_rate"] - aggregate.stds["success_rate"],
                0,
                1,
            ),
            np.clip(
                aggregate.means["success_rate"] + aggregate.stds["success_rate"],
                0,
                1,
            ),
            color=style["color"],
            alpha=SCI_BAND_ALPHA,
            linewidth=0,
        )
    axis.set(
        xlabel="Environment transitions",
        ylabel="Evaluation success rate",
        xlim=(0, METHOD_TRAINING_STEPS),
        ylim=(-0.03, 1.03),
    )
    _format_transition_axis(axis)
    apply_sci_grid(axis)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1),
    )
    apply_sci_figure_layout(
        fig, columns=2, height_in=3.1, left=0.11, bottom=0.18, top=0.84
    )
    return fig


def _plot_safety_learning_process(
    aggregates: list[CurveAggregate],
) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_sci_curve_style()
    fig, axis = plt.subplots()
    for aggregate in aggregates:
        style = _METHOD_STYLE_BY_ID[aggregate.variant_id]
        violation_mean = 1.0 - aggregate.means["safe_rate"]
        violation_std = aggregate.stds["safe_rate"]
        x_values = aggregate.axis_for("safe_rate")
        axis.plot(
            x_values,
            violation_mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=(
                SCI_LINE_WIDTH + 0.4
                if aggregate.variant_id == "ppo_pprs_dspl"
                else SCI_LINE_WIDTH
            ),
            label=aggregate.label,
        )
        axis.fill_between(
            x_values,
            np.clip(violation_mean - violation_std, 0, 1),
            np.clip(violation_mean + violation_std, 0, 1),
            color=style["color"],
            alpha=SCI_BAND_ALPHA,
            linewidth=0,
        )
    axis.set(
        xlabel="Environment transitions",
        ylabel="Evaluation safety violation rate",
        xlim=(0, METHOD_TRAINING_STEPS),
        ylim=(-0.03, 1.03),
    )
    _format_transition_axis(axis)
    apply_sci_grid(axis)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(aggregates)),
        frameon=False,
        bbox_to_anchor=(0.5, 1),
        borderaxespad=0,
    )
    apply_sci_figure_layout(
        fig, columns=2, height_in=3.1, left=0.11, bottom=0.18, top=0.84
    )
    return fig


def _print_final_table(aggregates: list[FinalMetricAggregate]) -> None:
    columns = (
        "method",
        "stop_error_m",
        "abs_time_error_s",
        "total_energy_kj",
        "comfort_tav",
    )
    print("Best-evaluation summary (mean±std):")
    print(" | ".join(columns))
    for aggregate in aggregates:
        cells = [aggregate.label or aggregate.variant_id]
        cells.extend(
            f"{aggregate.means[key]:.6f}±{aggregate.stds[key]:.6f}"
            for key in columns[1:]
        )
        print(" | ".join(cells))
        print(
            f"  success_rate={aggregate.success_rate:.3f}, "
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
    if (
        manifest.training_signature.get("curriculum_algorithm_id") != DSPL_ALGORITHM_ID
        or manifest.training_signature.get("dspl_protocol")
        != dspl_protocol_parameters()
    ):
        raise ValueError("method-ablation DSPL protocol is incompatible")
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
    if not isinstance(evaluation_interval, int) or evaluation_interval <= 0:
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


def build_policy_selection(manifest: AblationManifest) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for run in manifest.runs:
        if run.variant_id != "ppo_pprs_dspl":
            continue
        metrics = load_evaluation_metrics(Path(run.artifacts.path_for("metrics_best")))
        assessment = assess_constraints(metrics)
        candidates.append(
            {
                "run_id": run.run_id,
                "variant_id": run.variant_id,
                "seed": run.seed,
                "repeat_index": run.repeat_index,
                "rank_key": list(metrics.selection_comparison_key),
                "assessment": assessment.to_dict(),
                "metrics": {
                    "stop_error_m": abs(float(metrics.stop_error_m)),
                    "abs_time_error_s": abs(float(metrics.time_error_s)),
                    "total_energy_kj": float(metrics.total_energy_kj),
                    "comfort_tav": float(metrics.comfort_tav),
                },
                "model_dir": str(Path(run.artifacts.path_for("metrics_best")).parent),
                "artifacts": {
                    "policy_best": run.artifacts.path_for("policy_best"),
                    "trajectory_best": run.artifacts.path_for("trajectory_best"),
                    "metrics_best": run.artifacts.path_for("metrics_best"),
                    "metadata": run.artifacts.path_for("metadata_best"),
                },
            }
        )
    if len(candidates) != len(DEFAULT_SEEDS):
        raise ValueError(
            "policy selection requires all five PPO+PPRS+DSPL best policies"
        )
    best_key = max(tuple(item["rank_key"]) for item in candidates)  # type: ignore[arg-type]
    selected = min(
        (item for item in candidates if tuple(item["rank_key"]) == best_key),  # type: ignore[arg-type]
        key=lambda item: str(item["run_id"]),
    )
    return {
        "artifact_type": "paper_policy_selection",
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "source_manifest": (
            str(Path(manifest.output_root) / METHOD_ABLATION_MANIFEST_FILENAME)
            if manifest.output_root is not None
            else None
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            "strict feasibility first; minimum energy among feasible policies; "
            "otherwise safe success, precision, stop error, punctuality, absolute "
            "time error, and energy; run_id breaks exact ties"
        ),
        "candidate_variant_id": "ppo_pprs_dspl",
        "selected": selected,
        "candidates": sorted(candidates, key=lambda item: str(item["run_id"])),
    }


def save_policy_selection(payload: dict[str, object], output_path: Path) -> Path:
    payload = json.loads(json.dumps(payload))
    source_manifest = payload.get("source_manifest")
    if isinstance(source_manifest, str):
        payload["source_manifest"] = os.path.relpath(
            source_manifest, output_path.parent
        )
    for key in ("selected", "candidates"):
        items = payload[key] if key == "candidates" else [payload[key]]
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            model_dir = Path(str(item["model_dir"]))
            item["model_dir"] = os.path.relpath(model_dir, output_path.parent)
            artifacts = item.get("artifacts")
            if isinstance(artifacts, dict):
                for name, value in artifacts.items():
                    artifacts[name] = os.path.relpath(
                        Path(str(value)), output_path.parent
                    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def run_train(args: argparse.Namespace) -> int:
    if args.num_envs <= 0 or DEFAULT_ROLLOUT_STEPS_PER_UPDATE % args.num_envs != 0:
        raise SystemExit(
            "--num-envs must be a positive divisor of "
            f"{DEFAULT_ROLLOUT_STEPS_PER_UPDATE}"
        )
    if args.evaluation_interval_rollouts <= 0:
        raise SystemExit("--evaluation-interval-rollouts must be positive")
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
    interval = int(manifest.training_signature["evaluation_interval_rollouts"])
    expected_steps = (
        np.arange(interval, METHOD_TRAINING_ROLLOUTS, interval, dtype=np.int64)
        * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    ).astype(np.float64)
    for aggregate in curves:
        for metric in (
            "ep_reward",
            "ep_len",
            "stop_error_m",
            "abs_time_error_s",
            "success_rate",
            "safe_rate",
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
    _print_final_table(finals)
    _print_constraint_table(manifest)
    selection = build_policy_selection(manifest)
    selected = selection["selected"]
    assert isinstance(selected, dict)
    print(
        "Selected best-evaluation policy: "
        f"run_id={selected['run_id']} seed={selected['seed']} "
        f"directory={selected['model_dir']}"
    )
    print("Learning curves use all periodic independent evaluations.")
    if args.dry_run:
        return 0
    selection_path = args.selection_output_file or (
        Path(args.output_root) / DEFAULT_SELECTION_FILENAME
    )
    save_policy_selection(selection, selection_path)
    print(f"Saved policy selection to: {selection_path}")
    curve_figure = _plot_learning_curves(curves)
    safety_figure = _plot_safety_learning_process(curves)
    success_figure = _plot_evaluation_success_rate(curves)
    if args.figure_output_dir is not None:
        for figure, filename in zip(
            (curve_figure, safety_figure, success_figure),
            METHOD_FIGURE_FILENAMES,
            strict=True,
        ):
            saved_path = save_ablation_figure(figure, args.figure_output_dir / filename)
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
