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

from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_STEP_DISTANCE,
    DEFAULT_TRAINING_EPISODES,
    TrainingRunSpec,
    add_panel_label,
    apply_rl_curve_plot_style,
    build_default_training_args,
    evaluate_final_training_run,
    resolve_training_run_spec,
    train_single_experiment,
)
from rl.operational_state import ViolationCode
from rl.training_analysis.collect import (
    extract_complete_episode_sequence,
    load_reward_diagnostics_artifact,
)
from rl.training_analysis.process import trailing_moving_average
from utils.plot_utils import apply_sci_figure_layout, save_sci_figure

METHOD_ABLATION_MANIFEST_FILENAME = "method_ablation_manifest.json"
EVALUATION_HISTORY_FILENAME = "evaluation_history.npz"
FINAL_TRAJECTORY_METRICS_FILENAME = "final_trajectory_metrics.json"
MANIFEST_VERSION = 5
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/method_ablation"
DEFAULT_SEEDS = (11, 131, 239, 359, 443)
SAFETY_BIN_SIZE_M = 5_000.0
DEFAULT_EPISODE_SMOOTHING_WINDOW = 100
SAFETY_EPISODE_BIN_WIDTH = 500


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
    MethodSpec(
        "ppo_dspdl",
        "PPO+DSPDL",
        "basic",
        "dspdl_completion",
        "#CC79A7",
    ),
    MethodSpec(
        "ppo_pbrs_dspdl",
        "PPO+PBRS+DSPDL",
        "basic_safety",
        "dspdl_completion",
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
    episode_numbers: np.ndarray
    evaluation_steps: np.ndarray
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    episode_valid_seed_counts: np.ndarray
    valid_run_count: int


@dataclass(frozen=True)
class FinalMetricAggregate:
    method: MethodSpec
    valid_run_count: int
    success_rate: float
    means: dict[str, float]
    stds: dict[str, float]


@dataclass(frozen=True)
class SafetyLearningAggregate:
    method: MethodSpec
    episode_bin_edges: np.ndarray
    episode_bin_centers: np.ndarray
    mean_violation_rate: np.ndarray
    std_violation_rate: np.ndarray
    valid_seed_counts: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PPO/PBRS/DSPDL method-ablation experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train all methods and collect data.")
    train.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--reference-curve-dir", required=True)
    train.add_argument(
        "--training-episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
        help="Global completed training episodes for every ablation run.",
    )
    train.add_argument("--schedule-time-s", type=float, default=DEFAULT_SCHEDULE_TIME_S)
    train.add_argument("--step-distance", type=float, default=DEFAULT_STEP_DISTANCE)
    train.add_argument("--reward-discount", type=float, default=DEFAULT_REWARD_DISCOUNT)
    train.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
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
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible manifest and skip runs with complete final artifacts."
        ),
    )
    train.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=False
    )

    show = subparsers.add_parser(
        "show", help="Aggregate and plot method-ablation data."
    )
    show.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    show.add_argument("--output-file", type=Path, default=None)
    show.add_argument("--safety-output-file", type=Path, default=None)
    show.add_argument(
        "--episode-smoothing-window",
        type=int,
        default=DEFAULT_EPISODE_SMOOTHING_WINDOW,
        help=(
            "Trailing moving-average window in completed training episodes "
            f"(default: {DEFAULT_EPISODE_SMOOTHING_WINDOW})."
        ),
    )
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
        args.reference_curve_dir if method.curriculum_profile != "none" else None
    )
    result.experiment_tag = _tag(method, repeat_index)
    result.run_mode = "reproduce"
    result.enable_tb = False
    result.enable_monitor = True
    result.enable_auto_analysis = False
    result.enable_best_evaluation_artifacts = False
    result.enable_safety_truncation_histogram = False
    result.num_envs = args.num_envs
    result.rollout_steps_per_update = args.rollout_steps_per_update
    result.training_episodes = args.training_episodes
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
        "training_episodes": int(args.training_episodes),
        "schedule_time_s": float(args.schedule_time_s),
        "step_distance": float(args.step_distance),
        "reward_discount": float(args.reward_discount),
        "num_envs": int(args.num_envs),
        "rollout_steps_per_update": int(args.rollout_steps_per_update),
        "evaluation_interval_rollouts": int(args.evaluation_interval_rollouts),
        "device": str(args.device),
    }


def _manifest_training_budget_summary(spec: TrainingRunSpec) -> dict[str, object]:
    """Return resumable budget fields without duplicating the stop reason."""
    budget = spec.run_metadata["training_budget"]
    return {
        key: budget[key]
        for key in (
            "mode",
            "training_episodes",
            "effective_training_episodes",
            "max_episode_steps",
            "derived_total_timesteps",
            "actual_completed_episodes",
            "actual_training_timesteps",
            "target_reached",
        )
        if key in budget
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


def _completed_statuses_for_resume(
    args: argparse.Namespace,
    runs: list[MethodRun],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Keep only completed runs whose canonical final artifacts still exist."""
    manifest_path = Path(args.output_root) / METHOD_ABLATION_MANIFEST_FILENAME
    if not args.resume or not manifest_path.is_file():
        return {}

    expected_runs = {(run.method.name, run.repeat_index): run for run in runs}
    statuses: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in load_manifest(args.output_root).get("runs", []):
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            continue
        key = (str(entry.get("method")), int(entry.get("repeat_index", -1)))
        run = expected_runs.get(key)
        if run is None:
            continue
        if (
            Path(run.spec.final_model_save_path).is_file()
            and Path(run.final_metrics_path).is_file()
        ):
            status_payload: dict[str, Any] = {
                "status": "completed",
                "final_metrics_path": run.final_metrics_path,
            }
            training_budget = entry.get("training_budget")
            if isinstance(training_budget, dict):
                status_payload["training_budget"] = training_budget
            statuses[key] = status_payload
    return statuses


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
    aligned = np.full(reference.shape, np.nan, dtype=np.float64)
    indices = np.searchsorted(reference, steps)
    valid = indices < reference.size
    matching_indices = np.flatnonzero(valid)
    valid[matching_indices] = (
        reference[indices[matching_indices]] == steps[matching_indices]
    )
    aligned[indices[valid]] = values[valid]
    return aligned


def _smooth_episode_curve(
    episode_numbers: np.ndarray,
    values: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("episode_smoothing_window must be >= 1")
    smoothed = trailing_moving_average(values, window)
    if smoothed.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return episode_numbers[window - 1 :], smoothed


def _align_episode_values(
    reference: np.ndarray,
    episode_numbers: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    aligned = np.full(reference.shape, np.nan, dtype=np.float64)
    if episode_numbers.size == 0:
        return aligned
    indices = episode_numbers.astype(np.int64) - int(reference[0])
    valid = (indices >= 0) & (indices < reference.size)
    aligned[indices[valid]] = values[valid]
    return aligned


def _mean_and_sample_std(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(np.isfinite(values), axis=0)
    means = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    if np.any(valid):
        means[valid] = np.nansum(values[:, valid], axis=0) / counts[valid]
    stds = np.full(means.shape, np.nan, dtype=np.float64)
    multiple = counts >= 2
    if np.any(multiple):
        stds[multiple] = np.nanstd(values[:, multiple], axis=0, ddof=1)
    return means, stds, counts


def build_curve_aggregates(
    manifest: dict[str, Any],
    *,
    episode_smoothing_window: int = DEFAULT_EPISODE_SMOOTHING_WINDOW,
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
                episodes = extract_complete_episode_sequence(reward_artifact)
                periodic = _load_npz(
                    str(run["evaluation_history_path"]),
                    (
                        "training_steps",
                        "success",
                        "stop_error_m",
                        "abs_time_error_s",
                    ),
                )
                evaluation_steps = periodic["training_steps"].astype(float)
                evaluation_success = periodic["success"].astype(bool)
                if evaluation_success.shape != evaluation_steps.shape:
                    raise ValueError("success and training_steps have different shapes")
                curves.append(
                    {
                        "episode_numbers": episodes.episode_number.astype(float),
                        "ep_reward": episodes.total_reward,
                        "ep_len": episodes.length,
                        "eval_steps": evaluation_steps,
                        "stop_error_m": np.where(
                            evaluation_success,
                            periodic["stop_error_m"].astype(float),
                            np.nan,
                        ),
                        "abs_time_error_s": np.where(
                            evaluation_success,
                            periodic["abs_time_error_s"].astype(float),
                            np.nan,
                        ),
                    }
                )
            except (OSError, KeyError, ValueError) as exc:
                warnings.append(f"Skipped {method.label} curve: {exc}")
        if not curves:
            continue
        smoothed_episode_curves: list[dict[str, np.ndarray]] = []
        for curve in curves:
            reward_x, reward_values = _smooth_episode_curve(
                curve["episode_numbers"],
                curve["ep_reward"],
                window=episode_smoothing_window,
            )
            length_x, length_values = _smooth_episode_curve(
                curve["episode_numbers"],
                curve["ep_len"],
                window=episode_smoothing_window,
            )
            if reward_x.size == 0 or length_x.size == 0:
                warnings.append(
                    f"Skipped {method.label} episode curve because it has fewer than "
                    + f"{episode_smoothing_window} complete episodes."
                )
                continue
            smoothed_episode_curves.append(
                {
                    "episode_numbers": reward_x,
                    "ep_reward": reward_values,
                    "ep_len": length_values,
                }
            )
        if not smoothed_episode_curves:
            warnings.append(f"No smoothed episode curves available for {method.label}.")
            continue

        first_episode = int(
            min(curve["episode_numbers"][0] for curve in smoothed_episode_curves)
        )
        last_episode = int(
            max(curve["episode_numbers"][-1] for curve in smoothed_episode_curves)
        )
        episode_reference = np.arange(
            first_episode, last_episode + 1, dtype=np.float64
        )
        evaluation_reference = np.unique(
            np.concatenate([curve["eval_steps"] for curve in curves])
        )
        means: dict[str, np.ndarray] = {}
        stds: dict[str, np.ndarray] = {}
        episode_counts: np.ndarray | None = None
        for key in ("ep_reward", "ep_len"):
            aligned = np.vstack(
                [
                    _align_episode_values(
                        episode_reference, curve["episode_numbers"], curve[key]
                    )
                    for curve in smoothed_episode_curves
                ]
            )
            means[key], stds[key], counts = _mean_and_sample_std(aligned)
            if episode_counts is None:
                episode_counts = counts
            else:
                episode_counts = np.minimum(episode_counts, counts)
        for key in ("stop_error_m", "abs_time_error_s"):
            aligned = np.vstack(
                [
                    _align(evaluation_reference, curve["eval_steps"], curve[key])
                    for curve in curves
                ]
            )
            means[key], stds[key], _ = _mean_and_sample_std(aligned)
        assert episode_counts is not None
        aggregates.append(
            CurveAggregate(
                method,
                episode_reference,
                evaluation_reference,
                means,
                stds,
                episode_counts,
                len(smoothed_episode_curves),
            )
        )
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
    fig, axes = plt.subplots(2, 2)
    panels = (
        ("ep_reward", "Mean episode reward", "(a)"),
        ("ep_len", "Mean episode length", "(b)"),
        ("stop_error_m", "Mean absolute stop error (m)", "(c)"),
        (
            "abs_time_error_s",
            "Mean absolute time error (s)",
            "(d)",
        ),
    )
    for axis, (key, ylabel, panel) in zip(axes.flat, panels, strict=True):
        for aggregate in aggregates:
            x_values = (
                aggregate.episode_numbers
                if key in ("ep_reward", "ep_len")
                else aggregate.evaluation_steps
            )
            axis.plot(
                x_values,
                aggregate.means[key],
                color=aggregate.method.color,
                label=aggregate.method.label,
            )
            axis.fill_between(
                x_values,
                aggregate.means[key] - aggregate.stds[key],
                aggregate.means[key] + aggregate.stds[key],
                color=aggregate.method.color,
                alpha=0.16,
            )
        axis.set_xlabel(
            "Training episodes" if key in ("ep_reward", "ep_len") else "Training steps"
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        _ = add_panel_label(ax=axis, label=panel)
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
    manifest: dict[str, Any],
) -> tuple[list[SafetyLearningAggregate], list[str]]:
    aggregates: list[SafetyLearningAggregate] = []
    warnings: list[str] = []
    for method in METHODS:
        runs: list[np.ndarray] = []
        legacy_run_count = 0
        for run in _completed_method_runs(manifest, method):
            try:
                artifact = load_reward_diagnostics_artifact(
                    str(run["reward_diagnostics_path"])
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
                        [
                            int(ViolationCode.SPEED_LOW),
                            int(ViolationCode.SPEED_HIGH),
                        ],
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

        max_completed_episodes = max(run.size for run in runs)
        bin_count = (
            (max_completed_episodes - 1) // SAFETY_EPISODE_BIN_WIDTH + 1
        )
        bin_edges = (
            np.arange(bin_count + 1, dtype=np.float64) * SAFETY_EPISODE_BIN_WIDTH
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        per_seed_violation_rates = np.full(
            (len(runs), bin_count), np.nan, dtype=np.float64
        )
        for run_index, violations in enumerate(runs):
            episode_numbers = np.arange(1, violations.size + 1, dtype=np.int64)
            bin_indices = (episode_numbers - 1) // SAFETY_EPISODE_BIN_WIDTH
            for bin_index in np.unique(bin_indices):
                in_bin = bin_indices == bin_index
                per_seed_violation_rates[run_index, bin_index] = float(
                    np.mean(violations[in_bin])
                )

        mean_rate, std_rate, valid_counts = _mean_and_sample_std(
            per_seed_violation_rates
        )
        aggregates.append(
            SafetyLearningAggregate(
                method=method,
                episode_bin_edges=bin_edges,
                episode_bin_centers=bin_centers,
                mean_violation_rate=mean_rate,
                std_violation_rate=std_rate,
                valid_seed_counts=valid_counts,
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
    markers = ("o", "s", "^", "D")
    max_episode = max(aggregate.episode_bin_edges[-1] for aggregate in aggregates)
    for marker, aggregate in zip(markers, aggregates, strict=False):
        axis.plot(
            aggregate.episode_bin_centers,
            aggregate.mean_violation_rate,
            color=aggregate.method.color,
            marker=marker,
            markersize=4.0,
            label=aggregate.method.label,
        )
        std_valid = np.isfinite(aggregate.std_violation_rate)
        axis.fill_between(
            aggregate.episode_bin_centers,
            np.clip(
                aggregate.mean_violation_rate - aggregate.std_violation_rate, 0.0, 1.0
            ),
            np.clip(
                aggregate.mean_violation_rate + aggregate.std_violation_rate, 0.0, 1.0
            ),
            where=std_valid,
            color=aggregate.method.color,
            alpha=0.18,
            linewidth=0.0,
        )
    axis.set_xlabel("Completed training episodes")
    axis.set_ylabel("Training safety violation rate")
    axis.set_xlim(0.0, max_episode)
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(aggregates)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
    )
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=3.1,
        left=0.11,
        bottom=0.18,
        top=0.84,
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
        _ = save_sci_figure(fig, path, dpi=dpi)


def run_train(args: argparse.Namespace) -> int:
    runs = resolve_run_matrix(args)
    _validate_existing_manifest(args)
    if args.dry_run:
        for run in runs:
            print(f"{run.method.label} seed={run.seed} output={run.spec.output_dir}")
        return 0
    statuses = _completed_statuses_for_resume(args, runs)
    for run in runs:
        key = (run.method.name, run.repeat_index)
        if key in statuses:
            print(
                f"Skipping completed run: {run.method.label} seed={run.seed} "
                + f"output={run.spec.final_output_dir}"
            )
            continue
        _write_manifest(args.output_root, build_manifest(args, runs, statuses))
        try:
            completed_spec = train_single_experiment(run.train_args, spec=run.spec)
            _, metrics_path = evaluate_final_training_run(completed_spec)
            statuses[key] = {
                "status": "completed",
                "final_metrics_path": metrics_path,
                "training_budget": _manifest_training_budget_summary(completed_spec),
            }
        except Exception as exc:
            statuses[key] = {
                "status": "failed",
                "error_message": str(exc),
            }
            _write_manifest(args.output_root, build_manifest(args, runs, statuses))
            raise
    _write_manifest(args.output_root, build_manifest(args, runs, statuses))
    return 0


def run_show(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.output_root)
    if args.episode_smoothing_window < 1:
        raise SystemExit("--episode-smoothing-window must be >= 1")
    curves, curve_warnings = build_curve_aggregates(
        manifest,
        episode_smoothing_window=args.episode_smoothing_window,
    )
    safety_aggregates, safety_warnings = build_safety_learning_aggregates(manifest)
    finals, final_warnings = build_final_aggregates(manifest)
    print("\n".join([*curve_warnings, *safety_warnings, *final_warnings]))
    _print_final_table(finals)
    print(
        "Episode smoothing: "
        + f"trailing window={args.episode_smoothing_window} completed episodes."
    )
    if args.dry_run:
        return 0
    curve_figure = _plot_learning_curves(curves)
    safety_figure = _plot_safety_learning_process(safety_aggregates)
    _save(curve_figure, args.output_file, args.dpi)
    _save(safety_figure, args.safety_output_file, args.dpi)
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
