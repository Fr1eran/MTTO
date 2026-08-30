"""Train and display the safety reward-ablation matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    apply_rl_curve_plot_style,
    evaluate_final_training_run,
    reward_preset_names,
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
    load_npz_arrays,
    manifest_runs,
)
from utils.ablation.plotting import save_ablation_figure
from utils.plot_utils import apply_sci_figure_layout

REWARD_ABLATION_MANIFEST_FILENAME = "manifest.json"
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/reward_ablation_safety"
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
SAFETY_BIN_SIZE_M = 5_000.0
MANIFEST_VERSION = 1


def _reward(preset: str, label: str, color: str) -> VariantSpec:
    return VariantSpec(
        id=preset,
        label=label,
        color=color,
        manifest={"preset": preset, "label": label, "color": color},
        training={
            "reward_preset": preset,
            "curriculum_profile": "none",
            "reference_curve_dir": None,
        },
    )


REWARD_ABLATIONS = (
    _reward("basic", "PPO", "#0072B2"),
    _reward("basic_safety", "PPO+Safety PBRS", "#E69F00"),
)

SPEC = AblationSpec(
    matrix_id="reward",
    manifest_filename=REWARD_ABLATION_MANIFEST_FILENAME,
    default_output_root=DEFAULT_OUTPUT_ROOT,
    variants=REWARD_ABLATIONS,
    seeds=DEFAULT_SEEDS,
    cli=CLIConfig(
        description="Compare PPO with and without safety PBRS shaping.",
        train_help="Train or resume selected presets.",
        show_help="Aggregate and plot completed runs.",
        train_arguments=(
            ArgumentSpec(("--output-root",), {"default": DEFAULT_OUTPUT_ROOT}),
            ArgumentSpec(
                ("--reward-presets",),
                {"nargs": "+", "choices": reward_preset_names(), "default": None},
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
                    "help": "Resume a compatible manifest and skip complete runs.",
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
                ("--reward-presets",),
                {"nargs": "+", "choices": reward_preset_names(), "default": None},
            ),
            ArgumentSpec(("--output-file",), {"type": Path, "default": None}),
            ArgumentSpec(("--safety-output-file",), {"type": Path, "default": None}),
            ArgumentSpec(("--dpi",), {"type": float, "default": 300.0}),
            ArgumentSpec(("--no-show",), {"action": "store_true"}),
            ArgumentSpec(
                ("--dry-run",),
                {"action": argparse.BooleanOptionalAction, "default": False},
            ),
        ),
    ),
    run_id_template="reward__{variant_id}__seed{seed:04d}__r{repeat_number:02d}",
    experiment_tag_template="r{repeat_number:02d}",
    matrix_config={"variants": VariantPayloads(), "seeds": SeedValues()},
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
        "curriculum_profile": "none",
    },
    training_overrides={
        "evaluation_interval_rollouts": ArgRef(
            "evaluation_interval_rollouts", lambda value: max(1, int(value))
        )
    },
    selection_arg="reward_presets",
    all_variants_in_manifest=True,
    curve=CurveAggregationSpec(
        episode_reader="series",
        metrics=(
            CurveMetricSpec(
                "ep_reward",
                "episode",
                "total_reward",
                "end_step",
                alignment="exact_union",
            ),
            CurveMetricSpec(
                "success",
                "evaluation",
                "success",
                "training_steps",
                transform="bool",
                alignment="indexed",
            ),
            CurveMetricSpec(
                "stop_error_m",
                "evaluation",
                "stop_error_m",
                "training_steps",
                alignment="indexed",
            ),
            CurveMetricSpec(
                "abs_time_error_s",
                "evaluation",
                "abs_time_error_s",
                "training_steps",
                alignment="indexed",
            ),
            CurveMetricSpec(
                "total_energy_kj",
                "evaluation",
                "total_energy_j",
                "training_steps",
                transform="j_to_kj",
                alignment="indexed",
            ),
            CurveMetricSpec(
                "comfort_tav",
                "evaluation",
                "comfort_tav",
                "training_steps",
                alignment="indexed",
            ),
        ),
        primary_metric="ep_reward",
        x_name="episode_end_step",
    ),
    final=FinalAggregationSpec(
        metrics=(
            FinalMetricSpec("stop_error_m", "stop_error_m"),
            FinalMetricSpec("abs_time_error_s", "time_error_s", "abs"),
            FinalMetricSpec("total_energy_kj", "total_energy_kj"),
            FinalMetricSpec("comfort_tav", "comfort_tav"),
        )
    ),
    run_label_template="preset={preset} seed={seed} output={output_dir}",
    schema_version=MANIFEST_VERSION,
)

DRIVER = AblationDriver(SPEC)
RewardAblationSpec = VariantSpec
RewardAblationRun = AblationRun
build_arg_parser = DRIVER.build_arg_parser
_manifest_store = DRIVER.manifest_store
load_manifest = DRIVER.load_manifest
_validate_manifest_compatibility = DRIVER.validate_manifest
build_manifest = DRIVER.build_manifest
build_curve_aggregates = DRIVER.build_curve_aggregates
build_final_aggregates = DRIVER.build_final_aggregates


def _sync_seeds() -> None:
    if DRIVER.spec.seeds != tuple(DEFAULT_SEEDS):
        DRIVER.spec = replace(DRIVER.spec, seeds=tuple(DEFAULT_SEEDS))


def resolve_reward_ablation_specs(
    requested_presets: list[str] | None,
) -> tuple[VariantSpec, ...]:
    return DRIVER.resolve_variants(requested_presets)


def resolve_run_matrix(
    args: argparse.Namespace, *, requested_presets: list[str] | None = None
) -> list[AblationRun]:
    _sync_seeds()
    return DRIVER.resolve_runs(args, requested_presets)


def _plot_learning_curves(aggregates: list[CurveAggregate]) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_rl_curve_plot_style()
    figure, axes = plt.subplots(2, 3)
    panels = (
        ("ep_reward", "Mean episode reward", "(a)"),
        ("success", "Fixed-start success rate", "(b)"),
        ("stop_error_m", "Stop error (m)", "(c)"),
        ("abs_time_error_s", "Absolute time error (s)", "(d)"),
        ("total_energy_kj", "Total energy (kJ)", "(e)"),
        ("comfort_tav", "Comfort TAV", "(f)"),
    )
    for axis, (key, ylabel, panel) in zip(axes.flat, panels, strict=True):
        for aggregate in aggregates:
            steps = aggregate.axis_for(key)
            mean, std = aggregate.means[key], aggregate.stds[key]
            axis.plot(steps, mean, color=aggregate.color, label=aggregate.label)
            axis.fill_between(
                steps, mean - std, mean + std, color=aggregate.color, alpha=0.16
            )
        if key == "success":
            axis.set_ylim(-0.02, 1.02)
        axis.set(xlabel="Training steps", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        axis.text(
            0.02,
            0.98,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(aggregates), frameon=False
    )
    apply_sci_figure_layout(
        figure,
        columns=2,
        height_in=5.4,
        left=0.09,
        bottom=0.11,
        top=0.91,
        wspace=0.42,
        hspace=0.45,
    )
    return figure


def _load_violation_positions(path: str) -> tuple[np.ndarray, int]:
    data = load_npz_arrays(
        path, ("safety_violation_positions_m", "safety_violation_position_offsets")
    )
    positions = data["safety_violation_positions_m"].astype(float).reshape(-1)
    return positions, max(1, data["safety_violation_position_offsets"].size - 1)


def _plot_safety_boxplot(
    manifest: AblationManifest | dict[str, object],
    selected: tuple[VariantSpec, ...] | None = None,
) -> plt.Figure | None:
    selected = REWARD_ABLATIONS if selected is None else selected
    samples_by_preset: dict[str, list[tuple[np.ndarray, int]]] = {}
    for ablation in selected:
        samples = []
        for entry in manifest_runs(manifest):
            if entry.variant_id != ablation.id or entry.status != "completed":
                continue
            try:
                samples.append(
                    _load_violation_positions(entry.artifacts.path_for("evaluations"))
                )
            except OSError, KeyError, TypeError, ValueError:
                continue
        if samples:
            samples_by_preset[ablation.id] = samples
    if not samples_by_preset:
        return None
    maximum = max(
        (
            float(np.max(positions))
            for samples in samples_by_preset.values()
            for positions, _ in samples
            if positions.size
        ),
        default=SAFETY_BIN_SIZE_M,
    )
    edges = np.arange(0, maximum + 2 * SAFETY_BIN_SIZE_M, SAFETY_BIN_SIZE_M)
    figure, axis = plt.subplots()
    legend_specs: list[VariantSpec] = []
    for offset, ablation in zip(
        np.linspace(-0.3, 0.3, len(selected)), selected, strict=True
    ):
        rates = [
            np.histogram(positions, bins=edges)[0] / count
            for positions, count in samples_by_preset.get(ablation.id, [])
        ]
        if not rates:
            continue
        legend_specs.append(ablation)
        axis.boxplot(
            list(np.asarray(rates, dtype=float).T),
            positions=np.arange(edges.size - 1) + offset,
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": ablation.color, "alpha": 0.25},
            medianprops={"color": ablation.color},
        )
    axis.set_xticks(
        np.arange(edges.size - 1), [f"{value / 1000:g}" for value in edges[:-1]]
    )
    axis.set(
        xlabel="Violation position bin (km)",
        ylabel="Mean violations per fixed-start evaluation",
        title="Safety violations by position",
    )
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(
        [Line2D([0], [0], color=item.color, lw=5) for item in legend_specs],
        [item.label for item in legend_specs],
        loc="upper right",
    )
    apply_sci_figure_layout(figure, columns=2, height_in=3.1)
    return figure


def _print_final_table(aggregates: list[FinalMetricAggregate]) -> None:
    print("Final-policy evaluation summary (mean±std):")
    print(
        "preset | runs | success_rate | stop_error_m | abs_time_error_s | "
        "total_energy_kj | comfort_tav"
    )
    for aggregate in aggregates:
        cells = [
            aggregate.label or aggregate.variant_id,
            str(aggregate.valid_run_count),
            f"{aggregate.success_rate:.3f}",
        ]
        cells.extend(
            f"{aggregate.means[key]:.6f}±{aggregate.stds[key]:.6f}"
            for key in (
                "stop_error_m",
                "abs_time_error_s",
                "total_energy_kj",
                "comfort_tav",
            )
        )
        print(" | ".join(cells))


def run_train(args: argparse.Namespace) -> int:
    _sync_seeds()
    DRIVER.train_experiment = train_single_experiment
    DRIVER.evaluate_experiment = evaluate_final_training_run
    return DRIVER.run_train(args)


def run_show(args: argparse.Namespace) -> int:
    try:
        manifest = DRIVER.load_manifest(args.output_root)
        selected = DRIVER.resolve_variants(args.reward_presets)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cannot load reward ablation: {exc}")
        return 2
    curves, curve_warnings = DRIVER.build_curve_aggregates(manifest, selected)
    finals, final_warnings = DRIVER.build_final_aggregates(manifest, selected)
    for warning in (*curve_warnings, *final_warnings):
        print(f"WARNING: {warning}")
    _print_final_table(finals)
    if args.dry_run:
        return 0
    learning_figure = _plot_learning_curves(curves)
    safety_figure = _plot_safety_boxplot(manifest, selected)
    save_ablation_figure(learning_figure, args.output_file, dpi=args.dpi)
    save_ablation_figure(safety_figure, args.safety_output_file, dpi=args.dpi)
    if not args.no_show:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_train(args) if args.command == "train" else run_show(args)


if __name__ == "__main__":
    raise SystemExit(main())
