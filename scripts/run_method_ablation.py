"""Train and display the PPO/PBRS/DSPDL method-ablation matrix."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_STEP_DISTANCE,
    DEFAULT_VEC_ENV_TYPE,
    VEC_ENV_TYPE_CHOICES,
    TrainingRunSpec,
    apply_rl_curve_plot_style,
    build_default_training_args,
    evaluate_final_training_run,
    resolve_training_run_spec,
    train_single_experiment,
)
from rl.training_analysis.collect import (
    extract_complete_episode_series,
    load_reward_diagnostics_artifact,
)

METHOD_ABLATION_MANIFEST_FILENAME = "method_ablation_manifest.json"
EVALUATION_HISTORY_FILENAME = "evaluation_history.npz"
FINAL_TRAJECTORY_METRICS_FILENAME = "final_trajectory_metrics.json"
MANIFEST_VERSION = 1
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/method_ablation"
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
SAFETY_BIN_SIZE_M = 5_000.0


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    reward_preset: str
    curriculum_profile: str
    color: str


METHODS = (
    MethodSpec("ppo", "PPO", "basic", "none", "#0072B2"),
    MethodSpec("ppo_pbrs", "PPO+PBRS", "basic_safety", "none", "#E69F00"),
    MethodSpec("ppo_dspdl", "PPO+DSPDL", "basic", "dspdl", "#CC79A7"),
    MethodSpec(
        "ppo_pbrs_dspdl",
        "PPO+PBRS+DSPDL",
        "basic_safety",
        "dspdl",
        "#009E73",
    ),
)


@dataclass(frozen=True)
class MethodRun:
    method: MethodSpec
    repeat_index: int
    seed: int
    train_args: argparse.Namespace
    spec: TrainingRunSpec
    reward_diagnostics_path: str
    evaluation_history_path: str
    final_metrics_path: str


@dataclass(frozen=True)
class CurveAggregate:
    method: MethodSpec
    steps: np.ndarray
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    valid_run_count: int


@dataclass(frozen=True)
class FinalMetricAggregate:
    method: MethodSpec
    valid_run_count: int
    success_rate: float
    means: dict[str, float]
    stds: dict[str, float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PPO/PBRS/DSPDL method-ablation experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train all methods and collect data.")
    train.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--reference-curve-dir", required=True)
    train.add_argument("--total-timesteps", type=int, default=1_000_000)
    train.add_argument("--schedule-time-s", type=float, default=DEFAULT_SCHEDULE_TIME_S)
    train.add_argument("--step-distance", type=float, default=DEFAULT_STEP_DISTANCE)
    train.add_argument("--reward-discount", type=float, default=DEFAULT_REWARD_DISCOUNT)
    train.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    train.add_argument(
        "--vec-env-type",
        choices=VEC_ENV_TYPE_CHOICES,
        default=DEFAULT_VEC_ENV_TYPE,
    )
    train.add_argument(
        "--rollout-steps-per-update",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    )
    train.add_argument(
        "--evaluation-interval-rollouts",
        type=int,
        default=DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    )
    train.add_argument("--device", default=DEFAULT_DEVICE)
    train.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=False
    )

    show = subparsers.add_parser(
        "show", help="Aggregate and plot method-ablation data."
    )
    show.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    show.add_argument("--output-file", type=Path, default=None)
    show.add_argument("--safety-output-file", type=Path, default=None)
    show.add_argument("--dpi", type=float, default=300.0)
    show.add_argument("--no-show", action="store_true")
    show.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _tag(method: MethodSpec, repeat_index: int) -> str:
    return f"{method.name}__r{repeat_index + 1:02d}"


def _build_train_args(
    args: argparse.Namespace, method: MethodSpec, repeat_index: int, seed: int
) -> argparse.Namespace:
    result = argparse.Namespace(**vars(build_default_training_args()))
    result.output_root = args.output_root
    result.schedule_time_s = args.schedule_time_s
    result.step_distance = args.step_distance
    result.reward_discount = args.reward_discount
    result.reward_preset = method.reward_preset
    result.curriculum_profile = method.curriculum_profile
    result.reference_curve_dir = (
        args.reference_curve_dir if method.curriculum_profile == "dspdl" else None
    )
    result.experiment_tag = _tag(method, repeat_index)
    result.run_mode = "reproduce"
    result.enable_tb = False
    result.enable_monitor = True
    result.enable_auto_analysis = False
    result.enable_best_evaluation_artifacts = False
    result.enable_safety_truncation_histogram = False
    result.num_envs = args.num_envs
    result.vec_env_type = args.vec_env_type
    result.rollout_steps_per_update = args.rollout_steps_per_update
    result.total_timesteps = args.total_timesteps
    result.evaluation_interval_rollouts = args.evaluation_interval_rollouts
    result.evaluation_deterministic = True
    result.seed = seed
    result.device = args.device
    result.dry_run = False
    return result


def resolve_run_matrix(args: argparse.Namespace) -> list[MethodRun]:
    runs: list[MethodRun] = []
    for method in METHODS:
        for repeat_index, seed in enumerate(DEFAULT_SEEDS):
            train_args = _build_train_args(args, method, repeat_index, seed)
            spec = resolve_training_run_spec(train_args)
            periodic_path = os.path.join(
                spec.final_output_dir, EVALUATION_HISTORY_FILENAME
            )
            train_args.evaluation_history_path = periodic_path
            spec = resolve_training_run_spec(train_args)
            runs.append(
                MethodRun(
                    method,
                    repeat_index,
                    seed,
                    train_args,
                    spec,
                    spec.reward_diagnostics_path,
                    periodic_path,
                    os.path.join(
                        spec.final_output_dir, FINAL_TRAJECTORY_METRICS_FILENAME
                    ),
                )
            )
    return runs


def _training_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "total_timesteps": int(args.total_timesteps),
        "schedule_time_s": float(args.schedule_time_s),
        "step_distance": float(args.step_distance),
        "reward_discount": float(args.reward_discount),
        "num_envs": int(args.num_envs),
        "vec_env_type": str(args.vec_env_type),
        "rollout_steps_per_update": int(args.rollout_steps_per_update),
        "evaluation_interval_rollouts": int(args.evaluation_interval_rollouts),
        "device": str(args.device),
    }


def build_manifest(
    args: argparse.Namespace,
    runs: list[MethodRun],
    statuses: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    statuses = statuses or {}
    return {
        "manifest_version": MANIFEST_VERSION,
        "output_root": args.output_root,
        "reference_curve_dir": args.reference_curve_dir,
        "methods": [method.__dict__ for method in METHODS],
        "seed_list": list(DEFAULT_SEEDS),
        "training": _training_signature(args),
        "runs": [
            {
                "method": run.method.name,
                "repeat_index": run.repeat_index,
                "seed": run.seed,
                "output_dir": run.spec.output_dir,
                "final_output_dir": run.spec.final_output_dir,
                "reward_diagnostics_path": run.reward_diagnostics_path,
                "evaluation_history_path": run.evaluation_history_path,
                "final_metrics_path": run.final_metrics_path,
                **statuses.get(
                    (run.method.name, run.repeat_index), {"status": "pending"}
                ),
            }
            for run in runs
        ],
    }


def _write_manifest(root: str, payload: dict[str, Any]) -> None:
    os.makedirs(root, exist_ok=True)
    with (Path(root) / METHOD_ABLATION_MANIFEST_FILENAME).open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def load_manifest(root: str) -> dict[str, Any]:
    path = Path(root) / METHOD_ABLATION_MANIFEST_FILENAME
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _validate_manifest_compatibility(
    manifest: dict[str, Any], args: argparse.Namespace
) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            "Existing method-ablation manifest has an incompatible version"
        )
    if manifest.get("reference_curve_dir") != args.reference_curve_dir:
        raise ValueError(
            "Existing method-ablation manifest uses a different reference curve"
        )
    if manifest.get("methods") != [method.__dict__ for method in METHODS]:
        raise ValueError(
            "Existing method-ablation manifest uses a different method matrix"
        )
    if manifest.get("seed_list") != list(DEFAULT_SEEDS):
        raise ValueError("Existing method-ablation manifest uses different seeds")
    expected_training = _training_signature(args)
    actual_training = manifest.get("training")
    if actual_training != expected_training:
        raise ValueError(
            "Existing method-ablation manifest uses different training settings: "
            f"expected={expected_training}, actual={actual_training}"
        )


def _validate_existing_manifest(args: argparse.Namespace) -> None:
    path = Path(args.output_root) / METHOD_ABLATION_MANIFEST_FILENAME
    if path.is_file():
        _validate_manifest_compatibility(load_manifest(args.output_root), args)


def _method_by_name(name: str) -> MethodSpec:
    for method in METHODS:
        if method.name == name:
            return method
    raise ValueError(f"Unknown method: {name}")


def _load_npz(path: str, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing {missing} in {path}")
        return {key: np.asarray(data[key]) for key in data.files}


def _completed_method_runs(
    manifest: dict[str, Any], method: MethodSpec
) -> list[dict[str, Any]]:
    return [
        run
        for run in manifest.get("runs", [])
        if isinstance(run, dict)
        and run.get("method") == method.name
        and run.get("status") == "completed"
    ]


def _align(reference: np.ndarray, steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(reference, steps, values, left=np.nan, right=np.nan)


def build_curve_aggregates(
    manifest: dict[str, Any],
) -> tuple[list[CurveAggregate], list[str]]:
    aggregates: list[CurveAggregate] = []
    warnings: list[str] = []
    for method in METHODS:
        curves: list[dict[str, np.ndarray]] = []
        for run in _completed_method_runs(manifest, method):
            try:
                reward_artifact = load_reward_diagnostics_artifact(
                    str(run["reward_diagnostics_path"])
                )
                episodes = extract_complete_episode_series(reward_artifact)
                periodic = _load_npz(
                    str(run["evaluation_history_path"]),
                    ("training_steps", "stop_error_m", "abs_time_error_s"),
                )
                curves.append(
                    {
                        "episode_steps": episodes.end_step.astype(float),
                        "ep_reward": episodes.total_reward,
                        "ep_len": episodes.length,
                        "eval_steps": periodic["training_steps"].astype(float),
                        "stop_error_m": periodic["stop_error_m"].astype(float),
                        "abs_time_error_s": periodic["abs_time_error_s"].astype(float),
                    }
                )
            except (OSError, KeyError, ValueError) as exc:
                warnings.append(f"Skipped {method.label} curve: {exc}")
        if not curves:
            continue
        reference = np.unique(
            np.concatenate(
                [curve["episode_steps"] for curve in curves]
                + [curve["eval_steps"] for curve in curves]
            )
        )
        means: dict[str, np.ndarray] = {}
        stds: dict[str, np.ndarray] = {}
        for key, step_key in (
            ("ep_reward", "episode_steps"),
            ("ep_len", "episode_steps"),
            ("stop_error_m", "eval_steps"),
            ("abs_time_error_s", "eval_steps"),
        ):
            aligned = np.vstack(
                [_align(reference, curve[step_key], curve[key]) for curve in curves]
            )
            means[key] = np.nanmean(aligned, axis=0)
            stds[key] = np.nanstd(aligned, axis=0)
        aggregates.append(CurveAggregate(method, reference, means, stds, len(curves)))
    return aggregates, warnings


def build_final_aggregates(
    manifest: dict[str, Any],
) -> tuple[list[FinalMetricAggregate], list[str]]:
    keys = ("stop_error_m", "time_error_s", "total_energy_kj", "comfort_tav")
    results: list[FinalMetricAggregate] = []
    warnings: list[str] = []
    for method in METHODS:
        values = {key: [] for key in keys}
        successes: list[float] = []
        for run in _completed_method_runs(manifest, method):
            try:
                with Path(str(run["final_metrics_path"])).open(
                    encoding="utf-8"
                ) as file_obj:
                    metrics = json.load(file_obj)
                for key in keys:
                    values[key].append(float(metrics[key]))
                successes.append(float(bool(metrics.get("success", False))))
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                warnings.append(f"Skipped {method.label} final metrics: {exc}")
        if successes:
            results.append(
                FinalMetricAggregate(
                    method,
                    len(successes),
                    float(np.mean(successes)),
                    {key: float(np.mean(value)) for key, value in values.items()},
                    {key: float(np.std(value)) for key, value in values.items()},
                )
            )
    return results, warnings


def _plot_learning_curves(aggregates: list[CurveAggregate]) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_rl_curve_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    panels = (
        ("ep_reward", "Mean episode reward", "(a)"),
        ("ep_len", "Mean episode length", "(b)"),
        ("stop_error_m", "Mean absolute stop error (m)", "(c)"),
        ("abs_time_error_s", "Mean absolute time error (s)", "(d)"),
    )
    for axis, (key, ylabel, panel) in zip(axes.flat, panels, strict=True):
        for aggregate in aggregates:
            axis.plot(
                aggregate.steps,
                aggregate.means[key],
                color=aggregate.method.color,
                label=aggregate.method.label,
            )
            axis.fill_between(
                aggregate.steps,
                aggregate.means[key] - aggregate.stds[key],
                aggregate.means[key] + aggregate.stds[key],
                color=aggregate.method.color,
                alpha=0.16,
            )
        axis.set_xlabel("Training steps")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.text(0.5, -0.22, panel, transform=axis.transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _load_violation_positions(path: str) -> tuple[np.ndarray, int]:
    data = _load_npz(
        path,
        ("safety_violation_positions_m", "safety_violation_position_offsets"),
    )
    positions = np.asarray(data["safety_violation_positions_m"], dtype=float).reshape(
        -1
    )
    evaluation_count = max(
        1, np.asarray(data["safety_violation_position_offsets"]).size - 1
    )
    return positions, evaluation_count


def _plot_safety_boxplot(manifest: dict[str, Any]) -> plt.Figure | None:
    per_method: dict[str, list[tuple[np.ndarray, int]]] = {}
    for method in METHODS:
        samples: list[np.ndarray] = []
        for run in _completed_method_runs(manifest, method):
            try:
                samples.append(
                    _load_violation_positions(str(run["evaluation_history_path"]))
                )
            except OSError, ValueError:
                continue
        if samples:
            per_method[method.name] = samples
    if not per_method:
        return None
    max_position = max(
        (
            float(np.max(values))
            for sample_list in per_method.values()
            for values, _ in sample_list
            if values.size
        ),
        default=SAFETY_BIN_SIZE_M,
    )
    edges = np.arange(0.0, max_position + 2 * SAFETY_BIN_SIZE_M, SAFETY_BIN_SIZE_M)
    fig, axis = plt.subplots(figsize=(12.5, 5.2))
    offsets = np.linspace(-0.3, 0.3, len(METHODS))
    for offset, method in zip(offsets, METHODS, strict=True):
        runs = per_method.get(method.name, [])
        values = [
            np.histogram(positions, bins=edges)[0] / evaluation_count
            for positions, evaluation_count in runs
        ]
        if not values:
            continue
        bin_samples = [column for column in np.asarray(values, dtype=float).T]
        axis.boxplot(
            bin_samples,
            positions=np.arange(edges.size - 1) + offset,
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": method.color, "alpha": 0.25},
            medianprops={"color": method.color},
        )
    axis.set_xticks(
        np.arange(edges.size - 1), [f"{value / 1000:g}" for value in edges[:-1]]
    )
    axis.set_xlabel("Violation position bin (km)")
    axis.set_ylabel("Mean violations per periodic evaluation")
    axis.set_title("Fixed-start safety violations by position")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(
        [Line2D([0], [0], color=method.color, lw=5) for method in METHODS],
        [method.label for method in METHODS],
        loc="upper right",
    )
    fig.tight_layout()
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
        cells = [aggregate.method.label]
        for key in columns[1:]:
            cells.append(f"{aggregate.means[key]:.6f}±{aggregate.stds[key]:.6f}")
        print(" | ".join(cells))
        print(
            f"  success_rate={aggregate.success_rate:.3f}, "
            + f"runs={aggregate.valid_run_count}"
        )


def _save(fig: plt.Figure | None, path: Path | None, dpi: float) -> None:
    if fig is not None and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def run_train(args: argparse.Namespace) -> int:
    runs = resolve_run_matrix(args)
    _validate_existing_manifest(args)
    if args.dry_run:
        for run in runs:
            print(f"{run.method.label} seed={run.seed} output={run.spec.output_dir}")
        return 0
    statuses: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        _write_manifest(args.output_root, build_manifest(args, runs, statuses))
        try:
            train_single_experiment(run.train_args, spec=run.spec)
            _, metrics_path = evaluate_final_training_run(run.spec)
            statuses[(run.method.name, run.repeat_index)] = {
                "status": "completed",
                "final_metrics_path": metrics_path,
            }
        except Exception as exc:
            statuses[(run.method.name, run.repeat_index)] = {
                "status": "failed",
                "error_message": str(exc),
            }
            _write_manifest(args.output_root, build_manifest(args, runs, statuses))
            raise
    _write_manifest(args.output_root, build_manifest(args, runs, statuses))
    return 0


def run_show(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.output_root)
    curves, curve_warnings = build_curve_aggregates(manifest)
    finals, final_warnings = build_final_aggregates(manifest)
    print("\n".join([*curve_warnings, *final_warnings]))
    _print_final_table(finals)
    if args.dry_run:
        return 0
    curve_figure = _plot_learning_curves(curves)
    safety_figure = _plot_safety_boxplot(manifest)
    _save(curve_figure, args.output_file, args.dpi)
    _save(safety_figure, args.safety_output_file, args.dpi)
    if not args.no_show:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_train(args) if args.command == "train" else run_show(args)


if __name__ == "__main__":
    raise SystemExit(main())
