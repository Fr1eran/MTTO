"""Train and display the PPO/PBRS/DSPDL method-ablation matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from contracts.ablation import AblationManifest
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_STEP_DISTANCE,
    DEFAULT_TRAINING_EPISODES,
    add_panel_label,
    apply_rl_curve_plot_style,
    evaluate_final_training_run,
    train_single_experiment,
)
from rl.operational_state import ViolationCode
from rl.training_analysis.collect import (
    extract_complete_episode_sequence,
    load_reward_diagnostics_artifact,
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
    aggregate_matrix,
    manifest_runs,
)
from utils.ablation.plotting import save_ablation_figure
from utils.plot_utils import apply_sci_figure_layout

METHOD_ABLATION_MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/method_ablation"
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
DEFAULT_EPISODE_SMOOTHING_WINDOW = 100
SAFETY_EPISODE_BIN_WIDTH = 500


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
    _method("ppo_pbrs", "PPO+PBRS", "basic_safety", "none", "#E69F00"),
    _method("ppo_dspdl", "PPO+DSPDL", "basic", "dspdl_completion", "#CC79A7"),
    _method(
        "ppo_pbrs_dspdl",
        "PPO+PBRS+DSPDL",
        "basic_safety",
        "dspdl_completion",
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
        description="Run PPO/PBRS/DSPDL method-ablation experiments.",
        train_help="Train all methods and collect data.",
        show_help="Aggregate and plot method-ablation data.",
        train_arguments=(
            ArgumentSpec(("--output-root",), {"default": DEFAULT_OUTPUT_ROOT}),
            ArgumentSpec(("--reference-curve-dir",), {"required": True}),
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
                ("--step-distance",),
                {"type": float, "default": DEFAULT_STEP_DISTANCE},
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
                {"type": int, "default": DEFAULT_EVALUATION_INTERVAL_ROLLOUTS},
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
            ArgumentSpec(("--output-file",), {"type": Path, "default": None}),
            ArgumentSpec(("--safety-output-file",), {"type": Path, "default": None}),
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
            ArgumentSpec(("--dpi",), {"type": float, "default": 300.0}),
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
        "variants": VariantPayloads(),
        "seeds": SeedValues(),
        "reference_curve_dir": ArgRef("reference_curve_dir"),
    },
    training_signature={
        "training_episodes": ArgRef("training_episodes", int),
        "schedule_time_s": ArgRef("schedule_time_s", float),
        "step_distance": ArgRef("step_distance", float),
        "reward_discount": ArgRef("reward_discount", float),
        "num_envs": ArgRef("num_envs", int),
        "rollout_steps_per_update": ArgRef("rollout_steps_per_update", int),
        "evaluation_interval_rollouts": ArgRef(
            "evaluation_interval_rollouts", lambda value: max(1, int(value))
        ),
        "device": ArgRef("device", str),
    },
    training_overrides={
        "evaluation_interval_rollouts": ArgRef(
            "evaluation_interval_rollouts", lambda value: max(1, int(value))
        )
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
            CurveMetricSpec(
                "stop_error_m",
                "evaluation",
                "stop_error_m",
                "training_steps",
                success_only=True,
            ),
            CurveMetricSpec(
                "abs_time_error_s",
                "evaluation",
                "abs_time_error_s",
                "training_steps",
                success_only=True,
            ),
        ),
        primary_metric="ep_reward",
        x_name="episode_number",
        default_smoothing_window=DEFAULT_EPISODE_SMOOTHING_WINDOW,
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m"),
            FinalMetricSpec("time_error_s", "time_error_s"),
            FinalMetricSpec("total_energy_kj", "total_energy_kj"),
            FinalMetricSpec("comfort_tav", "comfort_tav"),
        )
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


@dataclass(frozen=True)
class SafetyLearningAggregate:
    method: VariantSpec
    episode_bin_edges: np.ndarray
    episode_bin_centers: np.ndarray
    mean_violation_rate: np.ndarray
    std_violation_rate: np.ndarray
    valid_seed_counts: np.ndarray


def _plot_learning_curves(aggregates: list[CurveAggregate]) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_rl_curve_plot_style()
    fig, axes = plt.subplots(2, 2)
    panels = (
        ("ep_reward", "Mean episode reward", "(a)"),
        ("ep_len", "Mean episode length", "(b)"),
        ("stop_error_m", "Mean absolute stop error (m)", "(c)"),
        ("abs_time_error_s", "Mean absolute time error (s)", "(d)"),
    )
    for axis, (key, ylabel, panel) in zip(axes.flat, panels, strict=True):
        for aggregate in aggregates:
            x_values = aggregate.axis_for(key)
            axis.plot(
                x_values,
                aggregate.means[key],
                color=aggregate.color,
                label=aggregate.label,
            )
            axis.fill_between(
                x_values,
                aggregate.means[key] - aggregate.stds[key],
                aggregate.means[key] + aggregate.stds[key],
                color=aggregate.color,
                alpha=0.16,
            )
        axis.set_xlabel(
            "Training episodes" if key in ("ep_reward", "ep_len") else "Training steps"
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        add_panel_label(ax=axis, label=panel)
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


def build_safety_learning_aggregates(
    manifest: AblationManifest | dict[str, object],
) -> tuple[list[SafetyLearningAggregate], list[str]]:
    aggregates: list[SafetyLearningAggregate] = []
    warnings: list[str] = []
    for method in METHODS:
        runs: list[np.ndarray] = []
        legacy_run_count = 0
        for run in manifest_runs(manifest):
            if run.variant_id != method.id or run.status != "completed":
                continue
            try:
                artifact = load_reward_diagnostics_artifact(
                    Path(run.artifacts.path_for("episodes"))
                )
                episodes = extract_complete_episode_sequence(artifact)
                if episodes.episode_number.size == 0:
                    raise ValueError("reward diagnostics contains no complete episodes")
                if np.any(episodes.violation_code < 0):
                    legacy_run_count += 1
                    continue
                runs.append(
                    np.isin(
                        episodes.violation_code,
                        [int(ViolationCode.SPEED_LOW), int(ViolationCode.SPEED_HIGH)],
                    ).astype(np.float64)
                )
            except (OSError, KeyError, ValueError) as exc:
                warnings.append(
                    f"Skipped {method.label} training safety history: {exc}"
                )
        if legacy_run_count:
            warnings.append(
                f"Skipped {legacy_run_count} {method.label} training safety run(s): "
                "reward diagnostics lack per-episode violation codes; rerun training "
                "to create schema-v3 artifacts"
            )
        if not runs:
            continue
        max_episodes = max(run.size for run in runs)
        bin_count = (max_episodes - 1) // SAFETY_EPISODE_BIN_WIDTH + 1
        edges = np.arange(bin_count + 1, dtype=float) * SAFETY_EPISODE_BIN_WIDTH
        rates = np.full((len(runs), bin_count), np.nan)
        for row, violations in enumerate(runs):
            bins = np.arange(violations.size) // SAFETY_EPISODE_BIN_WIDTH
            for column in np.unique(bins):
                rates[row, column] = np.mean(violations[bins == column])
        mean, std, counts = aggregate_matrix(rates)
        aggregates.append(
            SafetyLearningAggregate(
                method, edges, (edges[:-1] + edges[1:]) / 2, mean, std, counts
            )
        )
    return aggregates, warnings


def _plot_safety_learning_process(
    aggregates: list[SafetyLearningAggregate],
) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_rl_curve_plot_style()
    fig, axis = plt.subplots()
    for marker, aggregate in zip(("o", "s", "^", "D"), aggregates, strict=False):
        axis.plot(
            aggregate.episode_bin_centers,
            aggregate.mean_violation_rate,
            color=aggregate.method.color,
            marker=marker,
            markersize=4.0,
            label=aggregate.method.label,
        )
        axis.fill_between(
            aggregate.episode_bin_centers,
            np.clip(aggregate.mean_violation_rate - aggregate.std_violation_rate, 0, 1),
            np.clip(aggregate.mean_violation_rate + aggregate.std_violation_rate, 0, 1),
            where=np.isfinite(aggregate.std_violation_rate),
            color=aggregate.method.color,
            alpha=0.18,
            linewidth=0,
        )
    axis.set(
        xlabel="Completed training episodes",
        ylabel="Training safety violation rate",
        xlim=(0, max(item.episode_bin_edges[-1] for item in aggregates)),
        ylim=(-0.03, 1.03),
    )
    axis.grid(True, alpha=0.3)
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
        "time_error_s",
        "total_energy_kj",
        "comfort_tav",
    )
    print("Final-policy evaluation summary (mean±std):")
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


def run_train(args: argparse.Namespace) -> int:
    DRIVER.train_experiment = train_single_experiment
    DRIVER.evaluate_experiment = evaluate_final_training_run
    return DRIVER.run_train(args)


def run_show(args: argparse.Namespace) -> int:
    manifest = DRIVER.load_manifest(args.output_root)
    if args.episode_smoothing_window < 1:
        raise SystemExit("--episode-smoothing-window must be >= 1")
    curves, curve_warnings = DRIVER.build_curve_aggregates(
        manifest, episode_smoothing_window=args.episode_smoothing_window
    )
    safety, safety_warnings = build_safety_learning_aggregates(manifest)
    finals, final_warnings = DRIVER.build_final_aggregates(manifest)
    print("\n".join([*curve_warnings, *safety_warnings, *final_warnings]))
    _print_final_table(finals)
    print(
        f"Episode smoothing: trailing window={args.episode_smoothing_window} "
        "completed episodes."
    )
    if args.dry_run:
        return 0
    curve_figure = _plot_learning_curves(curves)
    safety_figure = _plot_safety_learning_process(safety)
    save_ablation_figure(curve_figure, args.output_file, dpi=args.dpi)
    save_ablation_figure(safety_figure, args.safety_output_file, dpi=args.dpi)
    if safety_figure is None:
        print(
            "No training safety violation figure was produced; "
            "rerun method ablation with schema-v3 reward diagnostics."
        )
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
