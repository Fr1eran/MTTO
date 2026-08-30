"""Run and display fixed spatial control-step ablations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from contracts.ablation import AblationManifest
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_TRAINING_EPISODES,
    add_panel_label,
    apply_rl_curve_plot_style,
    evaluate_final_training_run,
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
)
from utils.ablation.plotting import save_ablation_figure
from utils.io_utils import format_float_token
from utils.plot_utils import SCI_EXPORT_PAD_INCHES, apply_sci_figure_layout

DEFAULT_STEP_DISTANCES = (10.0, 30.0, 50.0, 100.0)
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/step_distance_ablation"
DEFAULT_REFERENCE_CURVE_DIR = "output/optimal/dp/465p0_0p1_uni10p0"
DEFAULT_EPISODE_SMOOTHING_WINDOW = 100
STEP_DISTANCE_MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
FIXED_REWARD_PRESET = "basic_safety"
FIXED_CURRICULUM_PROFILE = "dspdl_completion"
TRAJECTORY_METRIC_KEYS = (
    "stop_error_m",
    "time_error_s",
    "total_energy_kj",
    "comfort_tav",
)


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
        description="Run fixed spatial control-step ablation with PBRS + DSPDL.",
        train_help="Run the ablation matrix.",
        show_help="Plot episode-metrics learning curves.",
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
                        "required by DSPDL."
                    ),
                },
            ),
            ArgumentSpec(
                ("--enable-best-evaluation-artifacts",),
                {
                    "action": argparse.BooleanOptionalAction,
                    "default": False,
                    "help": (
                        "Optionally retain periodic best-trajectory evaluation; "
                        "final-policy evaluation is always enabled."
                    ),
                },
            ),
            ArgumentSpec(
                ("--training-episodes",),
                {
                    "type": int,
                    "default": DEFAULT_TRAINING_EPISODES,
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
                ("--evaluation-interval-rollouts",),
                {
                    "type": int,
                    "default": DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
                    "help": (
                        "Completed-rollout interval for best trajectory evaluation."
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
                ("--output-file",),
                {
                    "type": Path,
                    "default": None,
                    "help": (
                        "Path for saving a compact paper-ready figure. If omitted, "
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
                        "Trailing moving-average window in completed training "
                        "episodes (default: 100)."
                    ),
                },
            ),
            ArgumentSpec(
                ("--dpi",),
                {
                    "type": float,
                    "default": 300.0,
                    "help": "DPI used when saving the figure.",
                },
            ),
            ArgumentSpec(
                ("--pad-inches",),
                {
                    "type": float,
                    "default": SCI_EXPORT_PAD_INCHES,
                    "help": "Padding around the tight saved figure.",
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
        "step_distances": VariantValues("step_distance"),
        "seeds": SeedValues(),
        "reward_preset": FIXED_REWARD_PRESET,
        "curriculum_profile": FIXED_CURRICULUM_PROFILE,
        "reference_curve_dir": ArgRef("reference_curve_dir"),
    },
    training_signature={
        "schedule_time_s": ArgRef("schedule_time_s", float),
        "reward_discount": ArgRef("reward_discount", float),
        "num_envs": ArgRef("num_envs", int),
        "rollout_steps_per_update": ArgRef("rollout_steps_per_update", int),
        "n_steps_per_env": None,
        "training_episodes": ArgRef("training_episodes", int),
        "device": ArgRef("device", str),
        "enable_monitor": True,
        "enable_auto_analysis": False,
        "enable_best_evaluation_artifacts": ArgRef(
            "enable_best_evaluation_artifacts", bool
        ),
        "evaluation_interval_rollouts": ArgRef(
            "evaluation_interval_rollouts", lambda value: max(1, int(value))
        ),
        "evaluation_deterministic": True,
    },
    training_overrides={
        "reward_preset": FIXED_REWARD_PRESET,
        "curriculum_profile": FIXED_CURRICULUM_PROFILE,
        "reference_curve_dir": ArgRef("reference_curve_dir"),
        "enable_best_evaluation_artifacts": ArgRef(
            "enable_best_evaluation_artifacts", bool
        ),
        "evaluation_interval_rollouts": ArgRef(
            "evaluation_interval_rollouts", lambda value: max(1, int(value))
        ),
        "tensorboard_log_dir": None,
        "tb_log_name": None,
    },
    curve=CurveAggregationSpec(
        episode_reader="sequence",
        metrics=(
            CurveMetricSpec(
                "ep_reward", "episode", "total_reward", "episode_number", smooth=True
            ),
            CurveMetricSpec(
                "ep_len", "episode", "length", "episode_number", smooth=True
            ),
        ),
        primary_metric="ep_reward",
        x_name="episode_number",
        default_smoothing_window=DEFAULT_EPISODE_SMOOTHING_WINDOW,
        warn_non_completed=True,
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m"),
            FinalMetricSpec("time_error_s", "time_error_s"),
            FinalMetricSpec("total_energy_kj", "total_energy_kj"),
            FinalMetricSpec("comfort_tav", "comfort_tav"),
        ),
        source="auto",
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
    metric_source: str = "final",
) -> tuple[list[FinalMetricAggregate], list[str]]:
    return DRIVER.build_final_aggregates(
        manifest, step_distances, metric_source=metric_source
    )


def _color_for_index(index: int) -> Any:
    return plt.get_cmap("tab10")(index % 10)


def plot_curve_aggregates(
    aggregates: list[CurveAggregate], *, show: bool = True
) -> Figure | None:
    if not aggregates:
        print("No curve aggregates available; skipped plotting.")
        return None
    apply_rl_curve_plot_style()
    figure, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    reward_axis, length_axis = axes[0]
    for axis in (reward_axis, length_axis):
        axis.set_box_aspect(3 / 4)
    for index, aggregate in enumerate(aggregates):
        color = _color_for_index(index)
        label = aggregate.label or f"{aggregate.variant_id} m"
        for axis, key in (
            (reward_axis, "ep_reward"),
            (length_axis, "ep_len"),
        ):
            mean, std = aggregate.means[key], aggregate.stds[key]
            axis.plot(aggregate.x, mean, color=color, label=label)
            axis.fill_between(
                aggregate.x, mean - std, mean + std, color=color, alpha=0.18
            )
    reward_axis.set(xlabel="Training episodes", ylabel="Mean episode reward")
    length_axis.set(xlabel="Training episodes", ylabel="Mean episode length")
    for axis, panel in ((reward_axis, "(a)"), (length_axis, "(b)")):
        axis.grid(True, alpha=0.3)
        add_panel_label(ax=axis, label=panel)
    handles, labels = reward_axis.get_legend_handles_labels()
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


def save_compact_figure(
    figure: Figure, output_file: Path, dpi: float, pad_inches: float
) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    saved = save_ablation_figure(figure, output_file, dpi=dpi, pad_inches=pad_inches)
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


def _print_warnings(warnings: list[str]) -> None:
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


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
    if args.episode_smoothing_window < 1:
        raise SystemExit("--episode-smoothing-window must be >= 1")
    curves, curve_warnings = DRIVER.build_curve_aggregates(
        manifest, episode_smoothing_window=args.episode_smoothing_window
    )
    metric_source = DRIVER.resolve_metric_source(manifest)
    metrics, metric_warnings = DRIVER.build_final_aggregates(
        manifest, metric_source=metric_source
    )
    _print_warnings(curve_warnings + metric_warnings)
    _print_curve_summary(curves)
    print(
        f"Episode smoothing: trailing window={args.episode_smoothing_window} "
        "completed episodes."
    )
    _print_metric_table(metrics, metric_source=metric_source)
    if args.dry_run:
        print(
            "Dry run completed: episode-metrics and "
            f"{metric_source}-trajectory inputs resolved."
        )
        return 0
    if not curves:
        raise SystemExit("No valid monitor curves available for plotting.")
    figure = plot_curve_aggregates(curves, show=False)
    if figure is None:
        raise SystemExit("No valid monitor curves available for plotting.")
    if args.output_file is not None:
        output_path = save_compact_figure(
            figure, args.output_file, args.dpi, args.pad_inches
        )
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
