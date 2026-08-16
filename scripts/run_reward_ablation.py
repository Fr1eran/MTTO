"""Train, resume, aggregate, and display the reward-ablation matrix."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
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
    reward_preset_names,
    train_single_experiment,
)
from rl.training_analysis.collect import (
    extract_complete_episode_series,
    load_reward_diagnostics_artifact,
)

REWARD_ABLATION_MANIFEST_FILENAME = "reward_ablation_manifest.json"
EVALUATION_HISTORY_FILENAME = "evaluation_history.npz"
FINAL_TRAJECTORY_METRICS_FILENAME = "final_trajectory_metrics.json"
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/reward_ablation_safety"
DEFAULT_SEEDS = (11, 131, 239, 359, 443)

SAFETY_BIN_SIZE_M = 5_000.0
MANIFEST_VERSION = 3


@dataclass(frozen=True)
class RewardAblationSpec:
    preset: str
    label: str
    color: str


REWARD_ABLATIONS = (
    RewardAblationSpec("basic", "PPO", "#0072B2"),
    RewardAblationSpec("basic_safety", "PPO+Safety PBRS", "#E69F00"),
)


@dataclass(frozen=True)
class RewardAblationRun:
    ablation: RewardAblationSpec
    repeat_index: int
    seed: int
    train_args: argparse.Namespace
    spec: TrainingRunSpec
    reward_diagnostics_path: str
    evaluation_history_path: str
    final_metrics_path: str

    @property
    def key(self) -> tuple[str, int]:
        return self.ablation.preset, self.seed


@dataclass(frozen=True)
class CurveAggregate:
    ablation: RewardAblationSpec
    episode_steps: np.ndarray
    eval_steps: np.ndarray
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    valid_run_count: int


@dataclass(frozen=True)
class FinalMetricAggregate:
    ablation: RewardAblationSpec
    valid_run_count: int
    success_rate: float
    means: dict[str, float]
    stds: dict[str, float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PPO with and without safety PBRS shaping."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train or resume selected presets.")
    train.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    train.add_argument(
        "--reward-presets",
        nargs="+",
        choices=reward_preset_names(),
        default=None,
    )
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

    show = subparsers.add_parser("show", help="Aggregate and plot completed runs.")
    show.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    show.add_argument(
        "--reward-presets",
        nargs="+",
        choices=reward_preset_names(),
        default=None,
    )
    show.add_argument("--output-file", type=Path, default=None)
    show.add_argument("--safety-output-file", type=Path, default=None)
    show.add_argument("--dpi", type=float, default=300.0)
    show.add_argument("--no-show", action="store_true")
    show.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def resolve_reward_ablation_specs(
    requested_presets: list[str] | None,
) -> tuple[RewardAblationSpec, ...]:
    by_preset = {item.preset: item for item in REWARD_ABLATIONS}
    requested = list(by_preset) if requested_presets is None else requested_presets
    resolved: list[RewardAblationSpec] = []
    seen: set[str] = set()
    for preset in requested:
        if preset in seen:
            continue
        if preset not in by_preset:
            allowed = ", ".join(by_preset)
            raise ValueError(
                f"Preset '{preset}' is not in the reward-ablation matrix: {allowed}"
            )
        seen.add(preset)
        resolved.append(by_preset[preset])
    return tuple(resolved)


def _experiment_tag(repeat_index: int) -> str:
    return f"r{repeat_index + 1:02d}"


def _build_train_args(
    args: argparse.Namespace,
    ablation: RewardAblationSpec,
    repeat_index: int,
    seed: int,
) -> argparse.Namespace:
    result = argparse.Namespace(**vars(build_default_training_args()))
    result.output_root = args.output_root
    result.schedule_time_s = args.schedule_time_s
    result.step_distance = args.step_distance
    result.reward_discount = args.reward_discount
    result.reward_preset = ablation.preset
    result.curriculum_profile = "none"
    result.reference_curve_dir = None
    result.experiment_tag = _experiment_tag(repeat_index)
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


def resolve_run_matrix(
    args: argparse.Namespace,
    *,
    requested_presets: list[str] | None = None,
) -> list[RewardAblationRun]:
    presets = resolve_reward_ablation_specs(
        args.reward_presets if requested_presets is None else requested_presets
    )
    runs: list[RewardAblationRun] = []
    for ablation in presets:
        for repeat_index, seed in enumerate(DEFAULT_SEEDS):
            train_args = _build_train_args(args, ablation, repeat_index, seed)
            initial_spec = resolve_training_run_spec(train_args)
            periodic_path = os.path.join(
                initial_spec.final_output_dir,
                EVALUATION_HISTORY_FILENAME,
            )
            train_args.evaluation_history_path = periodic_path
            spec = resolve_training_run_spec(train_args)
            runs.append(
                RewardAblationRun(
                    ablation=ablation,
                    repeat_index=repeat_index,
                    seed=seed,
                    train_args=train_args,
                    spec=spec,
                    reward_diagnostics_path=spec.reward_diagnostics_path,
                    evaluation_history_path=periodic_path,
                    final_metrics_path=os.path.join(
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
        "curriculum_profile": "none",
    }


def _manifest_run(run: RewardAblationRun, status: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_preset": run.ablation.preset,
        "repeat_index": run.repeat_index,
        "seed": run.seed,
        "experiment_tag": run.train_args.experiment_tag,
        "output_dir": run.spec.output_dir,
        "final_output_dir": run.spec.final_output_dir,
        "run_metadata_path": run.spec.run_metadata_path,
        "reward_diagnostics_path": run.reward_diagnostics_path,
        "evaluation_history_path": run.evaluation_history_path,
        "final_metrics_path": run.final_metrics_path,
        **status,
    }


def build_manifest(
    args: argparse.Namespace,
    all_runs: list[RewardAblationRun],
    statuses: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status_map = statuses or {}
    return {
        "manifest_version": MANIFEST_VERSION,
        "output_root": args.output_root,
        "reward_ablations": [asdict(item) for item in REWARD_ABLATIONS],
        "seed_list": list(DEFAULT_SEEDS),
        "training": _training_signature(args),
        "runs": [
            _manifest_run(
                run,
                status_map.get(run.key, {"status": "pending"}),
            )
            for run in all_runs
        ],
    }


def _manifest_path(root: str) -> Path:
    return Path(root) / REWARD_ABLATION_MANIFEST_FILENAME


def _write_manifest(root: str, payload: dict[str, Any]) -> None:
    path = _manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary_path, path)


def load_manifest(root: str) -> dict[str, Any]:
    path = _manifest_path(root)
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Reward-ablation manifest must be an object: {path}")
    return payload


def _validate_manifest_compatibility(
    manifest: dict[str, Any], args: argparse.Namespace
) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            "Existing reward-ablation manifest has an incompatible version"
        )
    if manifest.get("seed_list") != list(DEFAULT_SEEDS):
        raise ValueError("Existing reward-ablation manifest uses different seeds")
    expected_ablations = [asdict(item) for item in REWARD_ABLATIONS]
    if manifest.get("reward_ablations") != expected_ablations:
        raise ValueError(
            "Existing reward-ablation manifest uses a different reward matrix"
        )
    expected_training = _training_signature(args)
    actual_training = manifest.get("training")
    if actual_training != expected_training:
        raise ValueError(
            "Existing reward-ablation manifest uses different training settings: "
            f"expected={expected_training}, actual={actual_training}"
        )


def _statuses_from_manifest(
    manifest: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    statuses: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in manifest.get("runs", []):
        if not isinstance(entry, dict):
            continue
        preset = entry.get("reward_preset")
        seed = entry.get("seed")
        status = entry.get("status")
        if not isinstance(preset, str) or not isinstance(seed, int):
            continue
        payload: dict[str, Any] = {
            "status": status if isinstance(status, str) else "pending"
        }
        if isinstance(entry.get("error_message"), str):
            payload["error_message"] = entry["error_message"]
        statuses[(preset, seed)] = payload
    return statuses


def _required_artifacts_exist(run: RewardAblationRun) -> bool:
    return all(
        Path(path).is_file()
        for path in (
            run.reward_diagnostics_path,
            run.evaluation_history_path,
            run.final_metrics_path,
        )
    )


def _load_npz(path: str, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing {missing} in {path}")
        return {key: np.asarray(data[key]) for key in data.files}


def _completed_preset_runs(
    manifest: dict[str, Any], preset: str
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in manifest.get("runs", [])
        if isinstance(entry, dict)
        and entry.get("reward_preset") == preset
        and entry.get("status") == "completed"
    ]


def _align(reference: np.ndarray, steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    if steps.size == 0 or values.size == 0 or steps.size != values.size:
        raise ValueError("Curve steps and values must be non-empty and equally sized")
    return np.interp(reference, steps, values, left=np.nan, right=np.nan)


def _aggregate_by_evaluation_index(
    curves: list[dict[str, np.ndarray]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_count = max(curve[key].size for curve in curves)
    value_matrix = np.full((len(curves), max_count), np.nan, dtype=np.float64)
    step_matrix = np.full_like(value_matrix, np.nan)
    for row, curve in enumerate(curves):
        count = curve[key].size
        value_matrix[row, :count] = curve[key]
        step_matrix[row, :count] = curve["eval_steps"][:count]
    return (
        np.nanmean(step_matrix, axis=0),
        np.nanmean(value_matrix, axis=0),
        np.nanstd(value_matrix, axis=0),
    )


def build_curve_aggregates(
    manifest: dict[str, Any],
    selected: tuple[RewardAblationSpec, ...] | None = None,
) -> tuple[list[CurveAggregate], list[str]]:
    selected = REWARD_ABLATIONS if selected is None else selected
    aggregates: list[CurveAggregate] = []
    warnings: list[str] = []
    periodic_keys = (
        "success",
        "stop_error_m",
        "abs_time_error_s",
        "total_energy_kj",
        "comfort_tav",
    )
    for ablation in selected:
        curves: list[dict[str, np.ndarray]] = []
        for entry in _completed_preset_runs(manifest, ablation.preset):
            try:
                reward_artifact = load_reward_diagnostics_artifact(
                    str(entry["reward_diagnostics_path"])
                )
                episodes = extract_complete_episode_series(reward_artifact)
                periodic = _load_npz(
                    str(entry["evaluation_history_path"]),
                    ("training_steps", *periodic_keys),
                )
                eval_steps = periodic["training_steps"].astype(np.float64)
                if eval_steps.size == 0:
                    raise ValueError("Periodic evaluation metrics are empty")
                curve = {
                    "episode_steps": episodes.end_step.astype(np.float64),
                    "ep_reward": episodes.total_reward,
                    "eval_steps": eval_steps,
                }
                for key in periodic_keys:
                    values = periodic[key].astype(np.float64)
                    if values.size != eval_steps.size:
                        raise ValueError(f"Mismatched periodic array: {key}")
                    curve[key] = values
                curves.append(curve)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                warnings.append(f"Skipped {ablation.label} curve: {exc}")
        if not curves:
            continue

        episode_reference = np.unique(
            np.concatenate([curve["episode_steps"] for curve in curves])
        )
        episode_matrix = np.vstack(
            [
                _align(episode_reference, curve["episode_steps"], curve["ep_reward"])
                for curve in curves
            ]
        )
        means = {"ep_reward": np.nanmean(episode_matrix, axis=0)}
        stds = {"ep_reward": np.nanstd(episode_matrix, axis=0)}

        eval_reference: np.ndarray | None = None
        for key in periodic_keys:
            steps, mean, std = _aggregate_by_evaluation_index(curves, key)
            if eval_reference is None:
                eval_reference = steps
            means[key] = mean
            stds[key] = std
        assert eval_reference is not None
        aggregates.append(
            CurveAggregate(
                ablation=ablation,
                episode_steps=episode_reference,
                eval_steps=eval_reference,
                means=means,
                stds=stds,
                valid_run_count=len(curves),
            )
        )
    return aggregates, warnings


def build_final_aggregates(
    manifest: dict[str, Any],
    selected: tuple[RewardAblationSpec, ...] | None = None,
) -> tuple[list[FinalMetricAggregate], list[str]]:
    selected = REWARD_ABLATIONS if selected is None else selected
    metric_sources = {
        "stop_error_m": "stop_error_m",
        "abs_time_error_s": "time_error_s",
        "total_energy_kj": "total_energy_kj",
        "comfort_tav": "comfort_tav",
    }
    aggregates: list[FinalMetricAggregate] = []
    warnings: list[str] = []
    for ablation in selected:
        values = {key: [] for key in metric_sources}
        successes: list[float] = []
        for entry in _completed_preset_runs(manifest, ablation.preset):
            try:
                with Path(str(entry["final_metrics_path"])).open(
                    encoding="utf-8"
                ) as file_obj:
                    metrics = json.load(file_obj)
                if not isinstance(metrics, dict):
                    raise ValueError("Final metrics payload must be an object")
                for result_key, source_key in metric_sources.items():
                    value = float(metrics[source_key])
                    values[result_key].append(
                        abs(value) if result_key == "abs_time_error_s" else value
                    )
                successes.append(float(bool(metrics["success"])))
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                warnings.append(f"Skipped {ablation.label} final metrics: {exc}")
        if successes:
            aggregates.append(
                FinalMetricAggregate(
                    ablation=ablation,
                    valid_run_count=len(successes),
                    success_rate=float(np.mean(successes)),
                    means={key: float(np.mean(item)) for key, item in values.items()},
                    stds={key: float(np.std(item)) for key, item in values.items()},
                )
            )
    return aggregates, warnings


def _plot_learning_curves(aggregates: list[CurveAggregate]) -> plt.Figure | None:
    if not aggregates:
        return None
    apply_rl_curve_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.3))
    panels = (
        ("ep_reward", "Mean episode reward", "episode_steps", "(a)"),
        ("success", "Fixed-start success rate", "eval_steps", "(b)"),
        ("stop_error_m", "Stop error (m)", "eval_steps", "(c)"),
        ("abs_time_error_s", "Absolute time error (s)", "eval_steps", "(d)"),
        ("total_energy_kj", "Total energy (kJ)", "eval_steps", "(e)"),
        ("comfort_tav", "Comfort TAV", "eval_steps", "(f)"),
    )
    for axis, (key, ylabel, step_attribute, panel) in zip(
        axes.flat, panels, strict=True
    ):
        for aggregate in aggregates:
            steps = getattr(aggregate, step_attribute)
            mean = aggregate.means[key]
            std = aggregate.stds[key]
            axis.plot(
                steps,
                mean,
                color=aggregate.ablation.color,
                label=aggregate.ablation.label,
            )
            axis.fill_between(
                steps,
                mean - std,
                mean + std,
                color=aggregate.ablation.color,
                alpha=0.16,
            )
        if key == "success":
            axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Training steps")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.text(0.5, -0.23, panel, transform=axis.transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(aggregates))
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _load_violation_positions(path: str) -> tuple[np.ndarray, int]:
    data = _load_npz(
        path,
        ("safety_violation_positions_m", "safety_violation_position_offsets"),
    )
    positions = data["safety_violation_positions_m"].astype(np.float64).reshape(-1)
    evaluation_count = max(1, data["safety_violation_position_offsets"].size - 1)
    return positions, evaluation_count


def _plot_safety_boxplot(
    manifest: dict[str, Any],
    selected: tuple[RewardAblationSpec, ...] | None = None,
) -> plt.Figure | None:
    selected = REWARD_ABLATIONS if selected is None else selected
    samples_by_preset: dict[str, list[tuple[np.ndarray, int]]] = {}
    for ablation in selected:
        samples: list[tuple[np.ndarray, int]] = []
        for entry in _completed_preset_runs(manifest, ablation.preset):
            try:
                samples.append(
                    _load_violation_positions(str(entry["evaluation_history_path"]))
                )
            except OSError, KeyError, TypeError, ValueError:
                continue
        if samples:
            samples_by_preset[ablation.preset] = samples
    if not samples_by_preset:
        return None

    maximum_position = max(
        (
            float(np.max(positions))
            for samples in samples_by_preset.values()
            for positions, _ in samples
            if positions.size
        ),
        default=SAFETY_BIN_SIZE_M,
    )
    edges = np.arange(
        0.0,
        maximum_position + 2.0 * SAFETY_BIN_SIZE_M,
        SAFETY_BIN_SIZE_M,
    )
    figure, axis = plt.subplots(figsize=(12.5, 5.2))
    offsets = np.linspace(-0.3, 0.3, len(selected))
    legend_specs: list[RewardAblationSpec] = []
    for offset, ablation in zip(offsets, selected, strict=True):
        samples = samples_by_preset.get(ablation.preset, [])
        rates = [
            np.histogram(positions, bins=edges)[0] / evaluation_count
            for positions, evaluation_count in samples
        ]
        if not rates:
            continue
        legend_specs.append(ablation)
        axis.boxplot(
            [column for column in np.asarray(rates, dtype=np.float64).T],
            positions=np.arange(edges.size - 1) + offset,
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": ablation.color, "alpha": 0.25},
            medianprops={"color": ablation.color},
        )
    axis.set_xticks(
        np.arange(edges.size - 1),
        [f"{value / 1000:g}" for value in edges[:-1]],
    )
    axis.set_xlabel("Violation position bin (km)")
    axis.set_ylabel("Mean violations per fixed-start evaluation")
    axis.set_title("Safety violations by position")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(
        [Line2D([0], [0], color=item.color, lw=5) for item in legend_specs],
        [item.label for item in legend_specs],
        loc="upper right",
    )
    figure.tight_layout()
    return figure


def _print_final_table(aggregates: list[FinalMetricAggregate]) -> None:
    print("Final-policy evaluation summary (mean±std):")
    print(
        "preset | runs | success_rate | stop_error_m | abs_time_error_s | "
        "total_energy_kj | comfort_tav"
    )
    for aggregate in aggregates:
        cells = [
            aggregate.ablation.label,
            str(aggregate.valid_run_count),
            f"{aggregate.success_rate:.3f}",
        ]
        for key in (
            "stop_error_m",
            "abs_time_error_s",
            "total_energy_kj",
            "comfort_tav",
        ):
            cells.append(f"{aggregate.means[key]:.6f}±{aggregate.stds[key]:.6f}")
        print(" | ".join(cells))


def _save(figure: plt.Figure | None, path: Path | None, dpi: float) -> None:
    if figure is None or path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def _load_resume_statuses(
    args: argparse.Namespace,
) -> dict[tuple[str, int], dict[str, Any]]:
    path = _manifest_path(args.output_root)
    if not path.is_file():
        return {}
    manifest = load_manifest(args.output_root)
    _validate_manifest_compatibility(manifest, args)
    return _statuses_from_manifest(manifest)


def run_train(args: argparse.Namespace) -> int:
    selected_runs = resolve_run_matrix(args)
    all_runs = resolve_run_matrix(
        args, requested_presets=[item.preset for item in REWARD_ABLATIONS]
    )
    try:
        statuses = _load_resume_statuses(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cannot resume reward ablation: {exc}")
        return 2

    actions: list[tuple[RewardAblationRun, str]] = []
    for run in selected_runs:
        status = statuses.get(run.key, {}).get("status")
        action = (
            "skip"
            if status == "completed" and _required_artifacts_exist(run)
            else "run"
        )
        actions.append((run, action))
        print(
            f"{action.upper():4s} preset={run.ablation.preset} seed={run.seed} "
            f"output={run.spec.output_dir}"
        )
    if args.dry_run:
        return 0

    failed = False
    for run, action in actions:
        if action == "skip":
            continue
        statuses[run.key] = {"status": "running"}
        _write_manifest(args.output_root, build_manifest(args, all_runs, statuses))
        try:
            train_single_experiment(run.train_args, spec=run.spec)
            _, metrics_path = evaluate_final_training_run(run.spec)
            statuses[run.key] = {
                "status": "completed",
                "final_metrics_path": metrics_path,
            }
        except Exception as exc:
            failed = True
            statuses[run.key] = {
                "status": "failed",
                "error_message": str(exc),
            }
            print(f"FAILED preset={run.ablation.preset} seed={run.seed}: {exc}")
        _write_manifest(args.output_root, build_manifest(args, all_runs, statuses))

    _write_manifest(args.output_root, build_manifest(args, all_runs, statuses))
    return 1 if failed else 0


def run_show(args: argparse.Namespace) -> int:
    try:
        manifest = load_manifest(args.output_root)
        selected = resolve_reward_ablation_specs(args.reward_presets)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cannot load reward ablation: {exc}")
        return 2

    curves, curve_warnings = build_curve_aggregates(manifest, selected)
    finals, final_warnings = build_final_aggregates(manifest, selected)
    for warning in (*curve_warnings, *final_warnings):
        print(f"WARNING: {warning}")
    _print_final_table(finals)
    if args.dry_run:
        return 0

    learning_figure = _plot_learning_curves(curves)
    safety_figure = _plot_safety_boxplot(manifest, selected)
    _save(learning_figure, args.output_file, args.dpi)
    _save(safety_figure, args.safety_output_file, args.dpi)
    if not args.no_show:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_train(args) if args.command == "train" else run_show(args)


if __name__ == "__main__":
    raise SystemExit(main())
