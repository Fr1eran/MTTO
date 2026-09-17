"""Run and display fixed spatial control-step ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from contracts.ablation import AblationManifest
from rl.evaluation import calculate_route_completion_ratio
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
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
    VariantSpec,
    VariantValues,
    manifest_run_complete,
)
from utils.ablation.plotting import save_ablation_figure
from utils.io_utils import (
    format_float_token,
    load_evaluation_history,
    load_evaluation_metrics,
)
from utils.plot_utils import (
    SCI_LINE_WIDTH,
    SCI_SERIES_LINE_STYLES,
    VIS_ACTUAL_PURPLE,
    VIS_PPO_GRAY,
    VIS_PROPOSED_ORANGE,
    VIS_SAFE_BLUE,
    add_panel_label,
    apply_sci_curve_style,
    apply_sci_figure_layout,
    apply_sci_grid,
    sci_tint_color,
)

DEFAULT_STEP_DISTANCES = (10.0, 30.0, 50.0, 100.0)
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
DEFAULT_OUTPUT_ROOT = "output/paper_experiment/01_step_distance"
DEFAULT_EVALUATION_SMOOTHING_WINDOW = 5
STEP_DISTANCE_TRAINING_ROLLOUTS = 400
STEP_DISTANCE_TRAINING_STEPS = (
    STEP_DISTANCE_TRAINING_ROLLOUTS * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
)
STEP_DISTANCE_MANIFEST_FILENAME = "manifest.json"
STEP_DISTANCE_FIGURE_FILENAME = "step_distance_learning_curves.pdf"
STEP_DISTANCE_TABLE_FILENAME = "step_distance_table.md"
STEP_DISTANCE_SUMMARY_FILENAME = "step_distance_summary.json"
MANIFEST_VERSION = 2
PROTOCOL_VERSION = 12
FIXED_REWARD_PRESET = "basic_safety_punctuality"
TRAJECTORY_METRIC_KEYS = (
    "stop_error_m",
    "abs_time_error_s",
    "total_energy_kwh",
    "comfort_tav",
)
_STEP_DISTANCE_COLORS = {
    "10p0": VIS_PPO_GRAY,
    "30p0": VIS_PROPOSED_ORANGE,
    "50p0": VIS_SAFE_BLUE,
    "100p0": VIS_ACTUAL_PURPLE,
}
_STEP_DISTANCE_STYLES = {
    variant_id: {
        "color": _STEP_DISTANCE_COLORS[variant_id],
        **SCI_SERIES_LINE_STYLES[index],
    }
    for index, variant_id in enumerate(_STEP_DISTANCE_COLORS)
}


def _step_variant(distance: float) -> VariantSpec:
    token = format_float_token(distance)
    return VariantSpec(
        id=token,
        label=f"{distance:g} m",
        color=None,
        manifest={"step_distance": float(distance)},
        training={"step_distance": float(distance)},
    )


def _step_variants() -> tuple[VariantSpec, ...]:
    return tuple(_step_variant(distance) for distance in DEFAULT_STEP_DISTANCES)


SPEC = AblationSpec(
    matrix_id="step_distance",
    manifest_filename=STEP_DISTANCE_MANIFEST_FILENAME,
    default_output_root=DEFAULT_OUTPUT_ROOT,
    variants=_step_variants(),
    seeds=DEFAULT_SEEDS,
    cli=CLIConfig(
        description=("Run fixed spatial control-step ablation with full PIRS."),
        train_help="Run the ablation matrix.",
        show_help="Plot periodic independent-evaluation learning curves.",
        train_arguments=(
            ArgumentSpec(
                ("--output-root", "--ablation-output-root"),
                {
                    "dest": "output_root",
                    "default": DEFAULT_OUTPUT_ROOT,
                    "help": "Root directory for step-distance ablation outputs.",
                },
            ),
            ArgumentSpec(
                ("--schedule-time-s",),
                {"type": float, "default": DEFAULT_SCHEDULE_TIME_S},
            ),
            ArgumentSpec(
                ("--reward-discount",),
                {"type": float, "default": DEFAULT_REWARD_DISCOUNT},
            ),
            ArgumentSpec(("--num-envs",), {"type": int, "default": DEFAULT_NUM_ENVS}),
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
                {
                    "action": argparse.BooleanOptionalAction,
                    "default": False,
                    "help": "Resolve the run matrix without starting training.",
                },
            ),
        ),
        show_arguments=(
            ArgumentSpec(
                ("--output-root", "--ablation-root"),
                {
                    "dest": "output_root",
                    "default": DEFAULT_OUTPUT_ROOT,
                    "help": "Root directory containing the step-distance manifest.",
                },
            ),
            ArgumentSpec(
                ("--figure-output-dir",),
                {
                    "type": Path,
                    "default": None,
                    "help": (
                        "Directory for the fixed-name paper-ready PDF, markdown table, "
                        "and JSON summary. If omitted, only display the figure."
                    ),
                },
            ),
            ArgumentSpec(
                ("--episode-smoothing-window", "--evaluation-smoothing-window"),
                {
                    "dest": "episode_smoothing_window",
                    "type": int,
                    "default": DEFAULT_EVALUATION_SMOOTHING_WINDOW,
                    "help": (
                        "Evaluation-point trailing moving-average window with an "
                        "expanding warm-up (default: 5)."
                    ),
                },
            ),
            ArgumentSpec(
                ("--no-show",),
                {
                    "action": "store_true",
                    "help": "Save without opening the interactive display window.",
                },
            ),
            ArgumentSpec(
                ("--dry-run",),
                {
                    "action": argparse.BooleanOptionalAction,
                    "default": False,
                    "help": "Resolve monitor inputs without plotting.",
                },
            ),
        ),
    ),
    run_id_template=(
        "step_distance__ds{variant_id}__seed{seed:04d}__r{repeat_number:02d}"
    ),
    experiment_tag_template="ds{variant_id}__r{repeat_number:02d}",
    matrix_config={
        "protocol_version": PROTOCOL_VERSION,
        "step_distances": VariantValues("step_distance"),
        "seeds": SeedValues(),
        "reward_preset": FIXED_REWARD_PRESET,
        "reward_config": reward_config_parameters(
            resolve_reward_preset(FIXED_REWARD_PRESET).config
        ),
    },
    training_signature={
        "protocol_version": PROTOCOL_VERSION,
        "budget_mode": "environment_steps",
        "training_rollouts": STEP_DISTANCE_TRAINING_ROLLOUTS,
        "training_steps": STEP_DISTANCE_TRAINING_STEPS,
        "learning_rate_schedule": learning_rate_schedule_parameters(
            "environment_steps"
        ),
        "schedule_time_s": ArgRef("schedule_time_s", float),
        "reward_discount": ArgRef("reward_discount", float),
        "num_envs": ArgRef("num_envs", int),
        "rollout_steps_per_update": DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        "n_steps_per_env": None,
        "evaluation_interval_rollouts": DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        "device": ArgRef("device", str),
        "enable_monitor": True,
        "enable_auto_analysis": False,
        "enable_best_evaluation_artifacts": True,
        "evaluation_deterministic": True,
    },
    training_overrides={
        "budget_mode": "environment_steps",
        "training_rollouts": STEP_DISTANCE_TRAINING_ROLLOUTS,
        "training_episodes": None,
        "reward_preset": FIXED_REWARD_PRESET,
        "enable_best_evaluation_artifacts": True,
        "rollout_steps_per_update": DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        "evaluation_interval_rollouts": DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        "evaluation_interval_episodes": None,
        "tensorboard_log_dir": None,
        "tb_log_name": None,
    },
    curve=CurveAggregationSpec(
        episode_reader="series",
        metrics=(
            CurveMetricSpec(
                "route_completion_ratio",
                "evaluation",
                "route_completion_ratio",
                "training_steps",
                transform="identity",
                smooth=True,
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "feasible_rate",
                "evaluation",
                "feasible",
                "training_steps",
                transform="bool",
                smooth=True,
                alignment="exact_union",
            ),
        ),
        primary_metric="route_completion_ratio",
        x_name="training_steps",
        default_smoothing_window=DEFAULT_EVALUATION_SMOOTHING_WINDOW,
        warn_non_completed=True,
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m"),
            FinalMetricSpec("abs_time_error_s", "time_error_s", transform="abs"),
            FinalMetricSpec("total_energy_kwh", "total_energy_j", transform="j_to_kwh"),
            FinalMetricSpec("comfort_tav", "comfort_tav"),
        ),
        source="best",
        warn_non_completed=True,
    ),
    run_label_template=(
        "step_distance={step_distance:g} seed={seed} repeat={repeat_number} "
        "output={output_dir}"
    ),
    schema_version=MANIFEST_VERSION,
)

DRIVER = AblationDriver(SPEC)
StepDistanceRunEntry = AblationRun
build_arg_parser = DRIVER.build_arg_parser
_manifest_store = DRIVER.manifest_store
load_step_distance_manifest = DRIVER.load_manifest
_validate_manifest_compatibility = DRIVER.validate_manifest
resolve_metric_source = DRIVER.resolve_metric_source


def _sync_matrix() -> None:
    variants = _step_variants()
    seeds = tuple(DEFAULT_SEEDS)
    if DRIVER.spec.variants != variants or DRIVER.spec.seeds != seeds:
        DRIVER.spec = replace(DRIVER.spec, variants=variants, seeds=seeds)


def resolve_step_distance_run_matrix(args: argparse.Namespace) -> list[AblationRun]:
    _sync_matrix()
    return DRIVER.resolve_runs(args)


def build_step_distance_manifest(
    args: argparse.Namespace,
    run_entries: list[AblationRun],
    *,
    statuses: dict[str, object] | None = None,
) -> AblationManifest:
    return DRIVER.build_manifest(args, run_entries, statuses)


def build_curve_aggregates(
    manifest: AblationManifest | dict[str, object],
    step_distances: list[float] | None = None,
    *,
    episode_smoothing_window: int = DEFAULT_EVALUATION_SMOOTHING_WINDOW,
) -> tuple[list[CurveAggregate], list[str]]:
    return DRIVER.build_curve_aggregates(
        manifest,
        step_distances,
        episode_smoothing_window=episode_smoothing_window,
    )


def build_metric_aggregates(
    manifest: AblationManifest | dict[str, object],
    *,
    step_distances: list[float] | None = None,
    metric_source: str = "best",
) -> tuple[list[FinalMetricAggregate], list[str]]:
    return DRIVER.build_final_aggregates(
        manifest, step_distances, metric_source=metric_source
    )


def _format_transition_axis(axis: plt.Axes) -> None:
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)


def plot_curve_aggregates(
    aggregates: list[CurveAggregate], *, show: bool = True
) -> Figure | None:
    if not aggregates:
        print("No curve aggregates available; skipped plotting.")
        return None
    apply_sci_curve_style()
    figure, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    completion_axis, feasible_axis = axes[0]
    for axis in (completion_axis, feasible_axis):
        axis.set_box_aspect(3 / 4)
    for aggregate in aggregates:
        style = _STEP_DISTANCE_STYLES[aggregate.variant_id]
        color = style["color"]
        label = aggregate.label or f"{aggregate.variant_id} m"
        for axis, key in (
            (completion_axis, "route_completion_ratio"),
            (feasible_axis, "feasible_rate"),
        ):
            mean, std = aggregate.means[key], aggregate.stds[key]
            x = aggregate.axis_for(key)
            if key == "feasible_rate":
                plot_mean = np.clip(mean, 0.0, 1.0)
                band_lower = np.clip(mean - std, 0.0, 1.0)
                band_upper = np.clip(mean + std, 0.0, 1.0)
            else:
                plot_mean = mean
                band_lower = mean - std
                band_upper = mean + std
            axis.plot(
                x,
                plot_mean,
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=3,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.7,
                linewidth=(
                    SCI_LINE_WIDTH + 0.4
                    if aggregate.variant_id == "30p0"
                    else SCI_LINE_WIDTH
                ),
                label=label,
            )
            axis.fill_between(
                x,
                band_lower,
                band_upper,
                color=sci_tint_color(color),
                linewidth=0,
            )
    completion_axis.set(
        xlabel="Environment transitions",
        ylabel="Route completion ratio",
        xlim=(0, STEP_DISTANCE_TRAINING_STEPS),
        ylim=(0.0, 1.0),
    )
    feasible_axis.set(
        xlabel="Environment transitions",
        ylabel="Strict feasibility rate",
        xlim=(0, STEP_DISTANCE_TRAINING_STEPS),
        ylim=(-0.03, 1.03),
    )
    for axis, panel in ((completion_axis, "(a)"), (feasible_axis, "(b)")):
        _format_transition_axis(axis)
        apply_sci_grid(axis)
        add_panel_label(ax=axis, label=panel)
    handles, labels = completion_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=min(4, len(aggregates)),
        borderaxespad=0,
        handlelength=1.8,
        columnspacing=1.2,
        frameon=False,
    )
    apply_sci_figure_layout(
        figure,
        columns=2,
        height_in=3.25,
        left=0.10,
        bottom=0.19,
        top=0.84,
        wspace=0.34,
    )
    if show:
        plt.show()
    return figure


def save_compact_figure(figure: Figure, output_dir: Path) -> Path:
    saved = save_ablation_figure(figure, output_dir / STEP_DISTANCE_FIGURE_FILENAME)
    assert saved is not None
    return saved


def _print_run_matrix(runs: list[AblationRun]) -> None:
    print("Resolved step-distance ablation run matrix:")
    for index, run in enumerate(runs, start=1):
        print(
            f"[{index}] step_distance={run.step_distance:g} "
            f"repeat={run.repeat_index + 1} seed={run.seed} "
            f"output_dir={run.training_spec.output_dir} "
            f"training_rollouts={run.training_spec.training_rollouts} "
            f"derived_total_timesteps={run.training_spec.total_timesteps}"
        )


def _print_curve_summary(aggregates: list[CurveAggregate]) -> None:
    print("Curve summary:")
    if not aggregates:
        print("  no valid step-distance curves available.")
        return
    for aggregate in aggregates:
        transition_end = float(aggregate.x[-1]) if aggregate.x.size else 0.0
        print(
            f"  - step_distance={aggregate.label or aggregate.variant_id} "
            f"valid_runs={aggregate.valid_run_count} "
            f"points={aggregate.x.size} transition_end={transition_end:g}"
        )


def build_step_distance_summary_and_table(
    manifest: AblationManifest,
) -> tuple[str, dict[str, object], float | None, str]:
    variant_results: list[dict[str, object]] = []

    for variant in _step_variants():
        variant_runs = [
            run
            for run in manifest.runs
            if run.variant_id == variant.id
            or DRIVER._entry_matches_variant(run, variant)
        ]
        if not variant_runs:
            continue
        seeds_data: list[dict[str, object]] = []
        stop_errors: list[float] = []
        time_errors: list[float] = []
        energy_kwhs: list[float] = []
        comforts: list[float] = []
        route_ratios: list[float] = []
        feasibles: list[bool] = []

        for run in variant_runs:
            metrics = load_evaluation_metrics(
                Path(run.artifacts.path_for("metrics_best"))
            )
            meta_path = (
                Path(run.artifacts.path_for("metadata"))
                if run.artifacts.metadata
                else None
            )
            metadata = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path is not None and meta_path.is_file()
                else {}
            )
            training_budget = metadata.get("training_budget", {})
            completed_episodes = int(
                training_budget.get("actual_completed_episodes")
                or training_budget.get("completed_episodes")
                or 0
            )

            route_ratio = calculate_route_completion_ratio(
                start_position_m=metrics.start_position_m,
                target_position_m=metrics.target_position_m,
                final_position_m=metrics.final_position_m,
            )
            stop_err = abs(float(metrics.stop_error_m))
            time_err = abs(float(metrics.time_error_s))
            e_kwh = float(metrics.total_energy_j) / 3_600_000.0
            comf = float(metrics.comfort_tav)
            feas = bool(metrics.feasible)

            stop_errors.append(stop_err)
            time_errors.append(time_err)
            energy_kwhs.append(e_kwh)
            comforts.append(comf)
            route_ratios.append(route_ratio)
            feasibles.append(feas)

            seeds_data.append(
                {
                    "seed": run.seed,
                    "feasible": feas,
                    "safe": bool(metrics.safe),
                    "success": bool(metrics.success),
                    "completed_training_episodes": completed_episodes,
                    "route_completion_ratio": route_ratio,
                    "stop_error_m": stop_err,
                    "abs_time_error_s": time_err,
                    "energy_kwh": e_kwh,
                    "comfort_tav": comf,
                }
            )

        feasible_count = sum(feasibles)
        feasible_rate = feasible_count / len(variant_runs) if variant_runs else 0.0
        safe_count = sum(d["safe"] for d in seeds_data)
        success_count = sum(d["success"] for d in seeds_data)
        mean_route_ratio = float(np.mean(route_ratios)) if route_ratios else 0.0

        # Sample standard deviation (ddof=1) over all trajectories
        stop_mean = float(np.mean(stop_errors)) if stop_errors else 0.0
        stop_std = float(np.std(stop_errors, ddof=1)) if len(stop_errors) > 1 else 0.0
        time_mean = float(np.mean(time_errors)) if time_errors else 0.0
        time_std = float(np.std(time_errors, ddof=1)) if len(time_errors) > 1 else 0.0
        energy_mean = float(np.mean(energy_kwhs)) if energy_kwhs else 0.0
        energy_std = float(np.std(energy_kwhs, ddof=1)) if len(energy_kwhs) > 1 else 0.0
        comfort_mean = float(np.mean(comforts)) if comforts else 0.0
        comfort_std = float(np.std(comforts, ddof=1)) if len(comforts) > 1 else 0.0

        feasible_energies = [
            e for f, e in zip(feasibles, energy_kwhs, strict=True) if f
        ]
        mean_feas_energy = (
            float(np.mean(feasible_energies)) if feasible_energies else None
        )
        feasible_comforts = [c for f, c in zip(feasibles, comforts, strict=True) if f]
        mean_feas_comfort = (
            float(np.mean(feasible_comforts)) if feasible_comforts else None
        )

        distance = float(variant.manifest["step_distance"])
        variant_results.append(
            {
                "variant_id": variant.id,
                "label": variant.label,
                "step_distance": distance,
                "feasible_count": feasible_count,
                "feasible_rate": feasible_rate,
                "safe_count": safe_count,
                "success_count": success_count,
                "mean_route_completion_ratio": mean_route_ratio,
                "mean_feasible_energy_kwh": mean_feas_energy,
                "mean_feasible_comfort": mean_feas_comfort,
                "metrics": {
                    "stop_error_m": {"mean": stop_mean, "std": stop_std},
                    "abs_time_error_s": {"mean": time_mean, "std": time_std},
                    "energy_kwh": {"mean": energy_mean, "std": energy_std},
                    "comfort_tav": {"mean": comfort_mean, "std": comfort_std},
                },
                "per_seed": seeds_data,
            }
        )

    # Step selection logic:
    # 严格可行数最多 → 五条 best/ 轨迹的平均里程完成率最高 → 可行轨迹平均能耗最低 →
    # 可行轨迹平均舒适度最低 → 较小步长。
    # 若四组均无可行轨迹，则报告“本轮不能确定合格步长”。
    if all(item["feasible_count"] == 0 for item in variant_results):
        recommended_step_distance = None
        selection_status = "本轮不能确定合格步长"
    else:
        sorted_variants = sorted(
            variant_results,
            key=lambda item: (
                -int(item["feasible_count"]),
                -float(item["mean_route_completion_ratio"]),
                (
                    float(item["mean_feasible_energy_kwh"])
                    if item["mean_feasible_energy_kwh"] is not None
                    else float("inf")
                ),
                (
                    float(item["mean_feasible_comfort"])
                    if item["mean_feasible_comfort"] is not None
                    else float("inf")
                ),
                float(item["step_distance"]),
            ),
        )
        recommended_step_distance = float(sorted_variants[0]["step_distance"])
        selection_status = f"推荐步长: {recommended_step_distance:g} m"

    # Format Markdown table
    header = (
        "| 步长 | 严格可行率 | 绝对停站误差 (m) | 绝对到站时间误差 (s) "
        "| 能耗 (kWh) | TAV (m/s²) |"
    )
    separator = "| --- | --- | --- | --- | --- | --- |"
    rows = [header, separator]
    for res in variant_results:
        m = res["metrics"]
        n_seeds = len(res["per_seed"])
        feas_str = (
            f"{res['feasible_rate'] * 100:.1f}% ({res['feasible_count']}/{n_seeds})"
        )
        rows.append(
            f"| {res['label']} | {feas_str} | "
            f"{m['stop_error_m']['mean']:.4f}±{m['stop_error_m']['std']:.4f} | "
            f"{m['abs_time_error_s']['mean']:.4f}±{m['abs_time_error_s']['std']:.4f} | "
            f"{m['energy_kwh']['mean']:.4f}±{m['energy_kwh']['std']:.4f} | "
            f"{m['comfort_tav']['mean']:.4f}±{m['comfort_tav']['std']:.4f} |"
        )
    table_note = (
        "\n*注：提前失败会影响能耗和误差的解释，须结合严格可行率综合评估。"
        "TAV（累计加速度变化量）公式为 "
        r"$\sum_t |a_t - a_{t-1}|$，单位为 $\mathrm{m/s^2}$。*"
    )
    markdown_table = "\n".join(rows) + table_note

    summary_payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "matrix_id": "step_distance",
        "recommended_step_distance": recommended_step_distance,
        "selection_status": selection_status,
        "variants": {item["variant_id"]: item for item in variant_results},
    }

    return markdown_table, summary_payload, recommended_step_distance, selection_status


def _validate_analysis_manifest(manifest: AblationManifest) -> None:
    if manifest.matrix_config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "step-distance manifest uses an obsolete protocol; rerun in the "
            f"current protocol-v{PROTOCOL_VERSION} output directory"
        )
    expected_config = {
        "step_distances": list(DEFAULT_STEP_DISTANCES),
        "seeds": list(DEFAULT_SEEDS),
        "reward_preset": FIXED_REWARD_PRESET,
        "reward_config": reward_config_parameters(
            resolve_reward_preset(FIXED_REWARD_PRESET).config
        ),
    }
    for key, value in expected_config.items():
        if manifest.matrix_config.get(key) != value:
            raise ValueError(f"step-distance manifest {key} is incompatible")
    if manifest.training_signature.get("budget_mode") != "environment_steps":
        raise ValueError("step-distance training budget_mode must be environment_steps")
    if (
        manifest.training_signature.get("training_rollouts")
        != STEP_DISTANCE_TRAINING_ROLLOUTS
    ):
        raise ValueError("step-distance training rollout budget is incompatible")
    if (
        manifest.training_signature.get("training_steps")
        != STEP_DISTANCE_TRAINING_STEPS
        or manifest.training_signature.get("rollout_steps_per_update")
        != DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    ):
        raise ValueError("step-distance rollout size or training steps is incompatible")
    if (
        manifest.training_signature.get("evaluation_interval_rollouts")
        != DEFAULT_EVALUATION_INTERVAL_ROLLOUTS
        or "evaluation_interval_episodes" in manifest.training_signature
    ):
        raise ValueError("step-distance rollout evaluation schedule is incompatible")
    expected = {
        "step_distance__"
        f"ds{format_float_token(distance)}__seed{seed:04d}__r{index + 1:02d}"
        for distance in DEFAULT_STEP_DISTANCES
        for index, seed in enumerate(DEFAULT_SEEDS)
    }
    actual = {run.run_id for run in manifest.runs}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"step-distance run matrix mismatch: missing={missing}, extra={extra}"
        )
    incomplete = [run.run_id for run in manifest.runs if not manifest_run_complete(run)]
    if incomplete:
        raise ValueError(
            "step-distance analysis requires completed budgets and canonical "
            f"artifacts for every run; invalid={incomplete}"
        )
    # Strictly verify that every run has exactly 33 periodic evaluation points
    expected_rollouts = np.arange(
        DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        STEP_DISTANCE_TRAINING_ROLLOUTS,
        DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
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
                "the 33 canonical evaluation points"
            )
        if not np.array_equal(history.training_steps, expected_steps):
            raise ValueError(
                f"Run {run.run_id} periodic evaluation training steps do not match "
                "the 33 canonical evaluation points"
            )


def _run_train_command(args: argparse.Namespace) -> int:
    _sync_matrix()
    runs = DRIVER.resolve_runs(args)
    _print_run_matrix(runs)
    DRIVER.train_experiment = train_single_experiment
    DRIVER.evaluate_experiment = evaluate_final_training_run
    return DRIVER.run_train(args)


def _run_show_command(args: argparse.Namespace) -> int:
    try:
        manifest = DRIVER.load_manifest(args.output_root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        _validate_analysis_manifest(manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.episode_smoothing_window < 1:
        raise SystemExit("--episode-smoothing-window must be >= 1")
    curves, curve_warnings = DRIVER.build_curve_aggregates(
        manifest, episode_smoothing_window=args.episode_smoothing_window
    )
    metric_source = DRIVER.resolve_metric_source(manifest)
    metrics, metric_warnings = DRIVER.build_final_aggregates(
        manifest, metric_source=metric_source
    )
    warnings = curve_warnings + metric_warnings
    if warnings:
        raise SystemExit(
            "Step-distance analysis inputs are invalid:\n" + "\n".join(warnings)
        )
    expected_count = len(DEFAULT_STEP_DISTANCES)
    if (
        len(curves) != expected_count
        or len(metrics) != expected_count
        or any(item.valid_run_count != len(DEFAULT_SEEDS) for item in curves)
        or any(item.valid_run_count != len(DEFAULT_SEEDS) for item in metrics)
    ):
        raise SystemExit("Step-distance analysis refused a partial aggregation")

    # Strictly verify 33 points with 5 valid seeds and finite values
    expected_steps = (
        np.arange(
            DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
            STEP_DISTANCE_TRAINING_ROLLOUTS,
            DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
            dtype=np.int64,
        )
        * DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    ).astype(np.float64)
    for aggregate in curves:
        for metric in ("route_completion_ratio", "feasible_rate"):
            stats = aggregate.metrics[metric]
            if (
                not np.array_equal(aggregate.axis_for(metric), expected_steps)
                or not np.all(stats.count == len(DEFAULT_SEEDS))
                or not np.all(np.isfinite(stats.mean))
                or not np.all(np.isfinite(stats.std))
            ):
                raise SystemExit(
                    f"Step-distance {metric} has missing or invalid evaluations"
                )

    _print_curve_summary(curves)
    print(
        "Evaluation smoothing: expanding warm-up then trailing "
        f"window={args.episode_smoothing_window} evaluation points."
    )

    markdown_table, summary_payload, recommended_distance, selection_status = (
        build_step_distance_summary_and_table(manifest)
    )
    print("\nPaper table (Markdown):")
    print(markdown_table)
    print(f"\nStep selection result: {selection_status}")

    if args.dry_run:
        print(
            "Dry run completed: verified 20 runs, 33 evaluation points, "
            "and previewed outputs without plotting or file writes."
        )
        return 0

    figure = plot_curve_aggregates(curves, show=False)
    if figure is None:
        raise SystemExit("No valid periodic-evaluation curves available for plotting.")

    if args.figure_output_dir is not None:
        out_dir = Path(args.figure_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = save_compact_figure(figure, out_dir)
        print(f"Saved figure to: {pdf_path}")

        table_path = out_dir / STEP_DISTANCE_TABLE_FILENAME
        table_path.write_text(markdown_table, encoding="utf-8")
        print(f"Saved paper table to: {table_path}")

        summary_path = out_dir / STEP_DISTANCE_SUMMARY_FILENAME
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Saved summary JSON to: {summary_path}")

    if not args.no_show:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return (
        _run_train_command(args) if args.command == "train" else _run_show_command(args)
    )


if __name__ == "__main__":
    raise SystemExit(main())
