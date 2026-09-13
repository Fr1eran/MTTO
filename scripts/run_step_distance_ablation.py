"""Run and display fixed spatial control-step ablations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from contracts.ablation import AblationManifest
from rl.experiment_statistics import assess_constraints
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
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
    VariantSpec,
    VariantValues,
    manifest_run_complete,
)
from utils.ablation.plotting import save_ablation_figure
from utils.io_utils import format_float_token, load_evaluation_metrics
from utils.plot_utils import (
    SCI_BAND_ALPHA,
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
)

DEFAULT_STEP_DISTANCES = (10.0, 30.0, 50.0, 100.0)
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
DEFAULT_OUTPUT_ROOT = "output/paper_experiment/01_step_distance"
DEFAULT_REFERENCE_CURVE_DIR = "output/optimal/dp/465p0_0p1_uni10p0"
DEFAULT_EVALUATION_INTERVAL_EPISODES = 100
DEFAULT_EPISODE_SMOOTHING_WINDOW = 5
STEP_DISTANCE_TRAINING_EPISODES = 4_000
STEP_DISTANCE_MANIFEST_FILENAME = "manifest.json"
STEP_DISTANCE_FIGURE_FILENAME = "step_distance_learning_curves.pdf"
MANIFEST_VERSION = 1
PROTOCOL_VERSION = 10
FIXED_REWARD_PRESET = "basic_safety_punctuality"
FIXED_CURRICULUM_PROFILE = "dspl"
TRAJECTORY_METRIC_KEYS = (
    "stop_error_m",
    "abs_time_error_s",
    "total_energy_kj",
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
        description=("Run fixed spatial control-step ablation with PPRS + DSPL."),
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
                ("--reference-curve-dir",),
                {
                    "default": DEFAULT_REFERENCE_CURVE_DIR,
                    "help": (
                        "Directory containing the matching DP reference trajectory "
                        "required by DSPL."
                    ),
                },
            ),
            ArgumentSpec(
                ("--training-episodes",),
                {
                    "type": int,
                    "default": STEP_DISTANCE_TRAINING_EPISODES,
                    "help": (
                        "Global completed training episodes for every ablation run."
                    ),
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
            ArgumentSpec(
                ("--rollout-steps-per-update",),
                {"type": int, "default": DEFAULT_ROLLOUT_STEPS_PER_UPDATE},
            ),
            ArgumentSpec(
                ("--evaluation-interval-episodes",),
                {
                    "type": int,
                    "default": DEFAULT_EVALUATION_INTERVAL_EPISODES,
                    "help": (
                        "Completed-training-episode interval for periodic evaluation."
                    ),
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
                        "Directory for the fixed-name paper-ready PDF. If omitted, "
                        "only display the figure."
                    ),
                },
            ),
            ArgumentSpec(
                ("--episode-smoothing-window",),
                {
                    "type": int,
                    "default": DEFAULT_EPISODE_SMOOTHING_WINDOW,
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
        "curriculum_profile": FIXED_CURRICULUM_PROFILE,
        "reference_curve_dir": ArgRef("reference_curve_dir"),
        "reward_config": reward_config_parameters(
            resolve_reward_preset(FIXED_REWARD_PRESET).config
        ),
        "dspl_protocol": dspl_protocol_parameters(),
    },
    training_signature={
        "protocol_version": PROTOCOL_VERSION,
        "budget_mode": "completed_episodes",
        "curriculum_algorithm_id": DSPL_ALGORITHM_ID,
        "schedule_time_s": ArgRef("schedule_time_s", float),
        "reward_discount": ArgRef("reward_discount", float),
        "num_envs": ArgRef("num_envs", int),
        "rollout_steps_per_update": ArgRef("rollout_steps_per_update", int),
        "n_steps_per_env": None,
        "training_episodes": ArgRef("training_episodes", int),
        "learning_rate_schedule": learning_rate_schedule_parameters(),
        "device": ArgRef("device", str),
        "enable_monitor": True,
        "enable_auto_analysis": False,
        "enable_best_evaluation_artifacts": True,
        "evaluation_interval_episodes": ArgRef("evaluation_interval_episodes", int),
        "evaluation_deterministic": True,
    },
    training_overrides={
        "budget_mode": "completed_episodes",
        "training_rollouts": None,
        "reward_preset": FIXED_REWARD_PRESET,
        "curriculum_profile": FIXED_CURRICULUM_PROFILE,
        "reference_curve_dir": ArgRef("reference_curve_dir"),
        "enable_best_evaluation_artifacts": True,
        "evaluation_interval_rollouts": None,
        "evaluation_interval_episodes": ArgRef("evaluation_interval_episodes", int),
        "tensorboard_log_dir": None,
        "tb_log_name": None,
    },
    curve=CurveAggregationSpec(
        episode_reader="sequence",
        metrics=(
            CurveMetricSpec(
                "trip_completion_pct",
                "evaluation",
                "route_completion_ratio",
                "scheduled_completed_training_episodes",
                transform="ratio_to_pct",
                smooth=True,
            ),
            CurveMetricSpec(
                "evaluation_episode_return",
                "evaluation",
                "total_reward",
                "scheduled_completed_training_episodes",
                smooth=True,
            ),
        ),
        primary_metric="trip_completion_pct",
        x_name="scheduled_completed_training_episodes",
        default_smoothing_window=DEFAULT_EPISODE_SMOOTHING_WINDOW,
        warn_non_completed=True,
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m"),
            FinalMetricSpec("abs_time_error_s", "time_error_s", transform="abs"),
            FinalMetricSpec("total_energy_kj", "total_energy_kj", feasible_only=True),
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
    episode_smoothing_window: int = DEFAULT_EPISODE_SMOOTHING_WINDOW,
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


def plot_curve_aggregates(
    aggregates: list[CurveAggregate], *, show: bool = True
) -> Figure | None:
    if not aggregates:
        print("No curve aggregates available; skipped plotting.")
        return None
    apply_sci_curve_style()
    figure, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    completion_axis, return_axis = axes[0]
    for axis in (completion_axis, return_axis):
        axis.set_box_aspect(3 / 4)
    for aggregate in aggregates:
        style = _STEP_DISTANCE_STYLES[aggregate.variant_id]
        color = style["color"]
        label = aggregate.label or f"{aggregate.variant_id} m"
        for axis, key in (
            (completion_axis, "trip_completion_pct"),
            (return_axis, "evaluation_episode_return"),
        ):
            mean, std = aggregate.means[key], aggregate.stds[key]
            x = aggregate.axis_for(key)
            axis.plot(
                x,
                mean,
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
                mean - std,
                mean + std,
                color=color,
                alpha=SCI_BAND_ALPHA,
                linewidth=0,
            )
    completion_axis.set(
        xlabel="Completed training episodes",
        ylabel="Policy trip completion (%)",
        xlim=(0, STEP_DISTANCE_TRAINING_EPISODES),
        ylim=(0, 100),
    )
    return_axis.set(
        xlabel="Completed training episodes",
        ylabel="Evaluation episode return",
        xlim=(0, STEP_DISTANCE_TRAINING_EPISODES),
    )
    for axis, panel in ((completion_axis, "(a)"), (return_axis, "(b)")):
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
            f"training_episodes={run.training_spec.training_episodes} "
            f"derived_total_timesteps={run.training_spec.total_timesteps}"
        )


def _print_curve_summary(aggregates: list[CurveAggregate]) -> None:
    print("Curve summary:")
    if not aggregates:
        print("  no valid step-distance curves available.")
        return
    for aggregate in aggregates:
        episode_end = float(aggregate.x[-1]) if aggregate.x.size else 0.0
        print(
            f"  - step_distance={aggregate.label or aggregate.variant_id} "
            f"valid_runs={aggregate.valid_run_count} "
            f"episode_points={aggregate.x.size} episode_end={episode_end:g}"
        )


def _print_metric_table(
    aggregates: list[FinalMetricAggregate], *, metric_source: str
) -> None:
    if not aggregates:
        print(
            f"{metric_source.title()} trajectory evaluation summary: no valid metrics."
        )
        return
    columns = ["step_distance", *TRAJECTORY_METRIC_KEYS]
    rows = [
        [
            aggregate.label or aggregate.variant_id,
            *[
                f"{aggregate.means[key]:.6f}±{aggregate.stds[key]:.6f}"
                for key in TRAJECTORY_METRIC_KEYS
            ],
        ]
        for aggregate in aggregates
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]

    def formatted(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(f"{metric_source.title()} trajectory evaluation summary (mean±std):")
    print(formatted(columns))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(formatted(row))


def _print_constraint_table(manifest: AblationManifest) -> None:
    print("Best-evaluation constraint rates:")
    print("step_distance | success | precise | punctual | safe | feasible | n")
    for variant in _step_variants():
        assessments = []
        for run in manifest.runs:
            if run.variant_id == variant.id:
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
            f"{variant.label} | "
            + " | ".join(f"{value:.3f}" for value in rates)
            + f" | {n}"
        )


def _print_warnings(warnings: list[str]) -> None:
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _validate_analysis_manifest(manifest: AblationManifest) -> None:
    if manifest.matrix_config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "step-distance manifest uses an obsolete protocol; rerun in the "
            "current protocol-v10 output directory"
        )
    expected_config = {
        "step_distances": list(DEFAULT_STEP_DISTANCES),
        "seeds": list(DEFAULT_SEEDS),
        "reward_preset": FIXED_REWARD_PRESET,
        "curriculum_profile": FIXED_CURRICULUM_PROFILE,
        "reward_config": reward_config_parameters(
            resolve_reward_preset(FIXED_REWARD_PRESET).config
        ),
        "dspl_protocol": dspl_protocol_parameters(),
    }
    for key, value in expected_config.items():
        if manifest.matrix_config.get(key) != value:
            raise ValueError(f"step-distance manifest {key} is incompatible")
    if manifest.training_signature.get("curriculum_algorithm_id") != DSPL_ALGORITHM_ID:
        raise ValueError("step-distance DSPL protocol is incompatible")
    if (
        manifest.training_signature.get("training_episodes")
        != STEP_DISTANCE_TRAINING_EPISODES
    ):
        raise ValueError("step-distance training episode budget is incompatible")
    if (
        manifest.training_signature.get("evaluation_interval_episodes")
        != DEFAULT_EVALUATION_INTERVAL_EPISODES
        or "evaluation_interval_rollouts" in manifest.training_signature
    ):
        raise ValueError("step-distance episode evaluation schedule is incompatible")
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
    _print_curve_summary(curves)
    print(
        "Evaluation smoothing: expanding warm-up then trailing "
        f"window={args.episode_smoothing_window} evaluation points."
    )
    _print_metric_table(metrics, metric_source=metric_source)
    _print_constraint_table(manifest)
    if args.dry_run:
        print(
            "Dry run completed: episode-metrics and "
            f"{metric_source}-trajectory inputs resolved."
        )
        return 0
    if not curves:
        raise SystemExit("No valid periodic-evaluation curves available for plotting.")
    figure = plot_curve_aggregates(curves, show=False)
    if figure is None:
        raise SystemExit("No valid periodic-evaluation curves available for plotting.")
    if args.figure_output_dir is not None:
        output_path = save_compact_figure(figure, args.figure_output_dir)
        print(f"Saved compact figure to {output_path}")
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
