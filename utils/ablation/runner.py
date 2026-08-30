"""Declarative RL-ablation driver and fail-fast matrix state machine."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from contracts.ablation import (
    ABLATION_MANIFEST_SCHEMA_VERSION,
    AblationManifest,
    AblationRunRecord,
    ArtifactRefs,
    ManifestStatusUpdate,
)
from rl.experiment_utils import (
    TrainingRunSpec,
    build_default_training_args,
    evaluate_final_training_run,
    resolve_training_run_spec,
    train_single_experiment,
)
from rl.training_analysis.collect import (
    extract_complete_episode_sequence,
    extract_complete_episode_series,
    load_reward_diagnostics_artifact,
)
from utils.io_utils import load_evaluation_history, load_evaluation_metrics

from .artifacts import artifact_paths, canonical_artifacts_complete
from .manifest import ManifestStore, build_manifest_payload, manifest_runs, status_map
from .models import ArtifactLayout, CurveAggregate, FinalMetricAggregate, MetricStats
from .statistics import aggregate_indexed_series, aggregate_matrix, align_exact

type ValueTransform = Literal["identity", "abs", "j_to_kj", "bool"]
type CurveSource = Literal["episode", "evaluation"]
type AlignmentMode = Literal["exact_range", "exact_union", "indexed"]


@dataclass(frozen=True)
class ArgumentSpec:
    """One declarative ``argparse`` option."""

    flags: tuple[str, ...]
    kwargs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CLIConfig:
    description: str
    train_help: str
    show_help: str
    train_arguments: tuple[ArgumentSpec, ...]
    show_arguments: tuple[ArgumentSpec, ...]


@dataclass(frozen=True)
class ArgRef:
    """Resolve a value from the parsed command-line namespace."""

    name: str
    cast: Callable[[object], object] | None = None


@dataclass(frozen=True)
class VariantValues:
    """Resolve one manifest field from every configured variant."""

    name: str


@dataclass(frozen=True)
class VariantPayloads:
    pass


@dataclass(frozen=True)
class SeedValues:
    cast: Callable[[object], object] = int


@dataclass(frozen=True)
class VariantSpec:
    id: str
    label: str
    color: str | None
    manifest: Mapping[str, object]
    training: Mapping[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> object:
        """Keep concise domain-specific access such as ``variant.preset``."""
        try:
            return self.manifest[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class CurveMetricSpec:
    name: str
    source: CurveSource
    value: str
    axis: str
    transform: ValueTransform = "identity"
    smooth: bool = False
    success_only: bool = False
    alignment: AlignmentMode = "exact_union"


@dataclass(frozen=True)
class CurveAggregationSpec:
    episode_reader: Literal["sequence", "series"]
    metrics: tuple[CurveMetricSpec, ...]
    primary_metric: str
    x_name: str
    default_smoothing_window: int = 1
    warn_non_completed: bool = False


@dataclass(frozen=True)
class FinalMetricSpec:
    name: str
    value: str
    transform: ValueTransform = "identity"


@dataclass(frozen=True)
class FinalAggregationSpec:
    metrics: tuple[FinalMetricSpec, ...]
    source: Literal["final", "best", "auto"] = "final"
    warn_non_completed: bool = False


@dataclass(frozen=True)
class AblationSpec:
    matrix_id: str
    manifest_filename: str
    default_output_root: str
    variants: tuple[VariantSpec, ...]
    seeds: tuple[int, ...]
    cli: CLIConfig
    run_id_template: str
    experiment_tag_template: str
    matrix_config: Mapping[str, object]
    training_signature: Mapping[str, object]
    curve: CurveAggregationSpec
    final: FinalAggregationSpec
    training_overrides: Mapping[str, object] = field(default_factory=dict)
    selection_arg: str | None = None
    all_variants_in_manifest: bool = False
    run_label_template: str = "{variant_label}"
    schema_version: int = ABLATION_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class AblationRun:
    run_id: str
    variant: VariantSpec
    repeat_index: int
    seed: int
    experiment_tag: str
    train_args: argparse.Namespace
    training_spec: TrainingRunSpec
    artifacts: ArtifactLayout

    @property
    def spec(self) -> TrainingRunSpec:
        return self.training_spec

    @property
    def training_run_spec(self) -> TrainingRunSpec:
        return self.training_spec

    @property
    def method(self) -> VariantSpec:
        return self.variant

    @property
    def ablation(self) -> VariantSpec:
        return self.variant

    @property
    def step_distance(self) -> float:
        return float(self.variant.manifest["step_distance"])

    @property
    def reward_diagnostics_path(self) -> str:
        return str(self.artifacts.episodes)

    @property
    def evaluation_history_path(self) -> str:
        return str(self.artifacts.evaluations)

    @property
    def final_metrics_path(self) -> str:
        return str(self.artifacts.metrics_final)


def _transform(values: object, transform: ValueTransform) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if transform == "abs":
        return np.abs(array)
    if transform == "j_to_kj":
        return array / 1000.0
    if transform == "bool":
        return array.astype(np.float64)
    return array


class AblationDriver:
    """Own the common CLI, matrix, manifest, training and aggregation flow."""

    def __init__(
        self,
        spec: AblationSpec,
        *,
        train_experiment: Callable[..., TrainingRunSpec] = train_single_experiment,
        evaluate_experiment: Callable[[TrainingRunSpec], tuple[str, str]] = (
            evaluate_final_training_run
        ),
    ) -> None:
        self.spec = spec
        self.train_experiment = train_experiment
        self.evaluate_experiment = evaluate_experiment

    def build_arg_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.spec.cli.description)
        subparsers = parser.add_subparsers(dest="command", required=True)
        train = subparsers.add_parser("train", help=self.spec.cli.train_help)
        show = subparsers.add_parser("show", help=self.spec.cli.show_help)
        for target, arguments in (
            (train, self.spec.cli.train_arguments),
            (show, self.spec.cli.show_arguments),
        ):
            for argument in arguments:
                target.add_argument(*argument.flags, **dict(argument.kwargs))
        return parser

    @staticmethod
    def _resolve_value(value: object, args: argparse.Namespace) -> object:
        if not isinstance(value, ArgRef):
            return value
        resolved = getattr(args, value.name)
        return value.cast(resolved) if value.cast is not None else resolved

    def _resolve_mapping(
        self, mapping: Mapping[str, object], args: argparse.Namespace
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in mapping.items():
            if isinstance(value, VariantPayloads):
                result[key] = [dict(variant.manifest) for variant in self.spec.variants]
            elif isinstance(value, VariantValues):
                result[key] = [
                    variant.manifest[value.name] for variant in self.spec.variants
                ]
            elif isinstance(value, SeedValues):
                result[key] = [value.cast(seed) for seed in self.spec.seeds]
            else:
                result[key] = self._resolve_value(value, args)
        return result

    def resolve_variants(
        self, requested: Sequence[object] | None = None
    ) -> tuple[VariantSpec, ...]:
        if requested is None:
            return self.spec.variants
        resolved: list[VariantSpec] = []
        seen: set[str] = set()
        for raw in requested:
            if isinstance(raw, VariantSpec):
                raw = raw.id
            match = next(
                (
                    variant
                    for variant in self.spec.variants
                    if raw == variant.id or raw in variant.manifest.values()
                ),
                None,
            )
            if match is None:
                allowed = ", ".join(variant.id for variant in self.spec.variants)
                raise ValueError(
                    f"Unknown {self.spec.matrix_id} variant {raw!r}: {allowed}"
                )
            if match.id not in seen:
                seen.add(match.id)
                resolved.append(match)
        return tuple(resolved)

    def selected_variants(self, args: argparse.Namespace) -> tuple[VariantSpec, ...]:
        requested = (
            getattr(args, self.spec.selection_arg)
            if self.spec.selection_arg is not None
            else None
        )
        return self.resolve_variants(requested)

    def _training_args(
        self,
        args: argparse.Namespace,
        variant: VariantSpec,
        repeat_index: int,
        seed: int,
    ) -> argparse.Namespace:
        result = build_default_training_args()
        result.output_root = str(Path(args.output_root) / "runs")
        for name in (
            "schedule_time_s",
            "step_distance",
            "reward_discount",
            "num_envs",
            "rollout_steps_per_update",
            "training_episodes",
            "evaluation_interval_rollouts",
            "device",
        ):
            if hasattr(args, name):
                setattr(result, name, getattr(args, name))
        common = {
            "run_mode": "reproduce",
            "enable_tb": False,
            "enable_monitor": True,
            "enable_auto_analysis": False,
            "enable_best_evaluation_artifacts": False,
            "enable_safety_truncation_histogram": False,
            "evaluation_deterministic": True,
            "dry_run": False,
        }
        for overrides in (common, self.spec.training_overrides, variant.training):
            for name, value in overrides.items():
                setattr(result, name, self._resolve_value(value, args))
        format_values = {
            "variant_id": variant.id,
            "repeat_index": repeat_index,
            "repeat_number": repeat_index + 1,
            "seed": seed,
        }
        result.experiment_tag = self.spec.experiment_tag_template.format(
            **format_values
        )
        result.seed = seed
        return result

    def resolve_runs(
        self,
        args: argparse.Namespace,
        requested: Sequence[object] | None = None,
    ) -> list[AblationRun]:
        variants = (
            self.selected_variants(args)
            if requested is None
            else self.resolve_variants(requested)
        )
        runs: list[AblationRun] = []
        for variant in variants:
            for repeat_index, seed in enumerate(self.spec.seeds):
                train_args = self._training_args(args, variant, repeat_index, seed)
                initial_spec = resolve_training_run_spec(train_args)
                train_args.evaluation_history_path = str(
                    Path(initial_spec.final_output_dir) / "evaluations.npz"
                )
                training_spec = resolve_training_run_spec(train_args)
                format_values = {
                    "matrix_id": self.spec.matrix_id,
                    "variant_id": variant.id,
                    "repeat_index": repeat_index,
                    "repeat_number": repeat_index + 1,
                    "seed": seed,
                }
                runs.append(
                    AblationRun(
                        run_id=self.spec.run_id_template.format(**format_values),
                        variant=variant,
                        repeat_index=repeat_index,
                        seed=int(seed),
                        experiment_tag=str(train_args.experiment_tag),
                        train_args=train_args,
                        training_spec=training_spec,
                        artifacts=ArtifactLayout.from_training_spec(training_spec),
                    )
                )
        return runs

    def training_signature(self, args: argparse.Namespace) -> dict[str, object]:
        return self._resolve_mapping(self.spec.training_signature, args)

    def matrix_config(self, args: argparse.Namespace) -> dict[str, object]:
        return self._resolve_mapping(self.spec.matrix_config, args)

    @staticmethod
    def _status(
        statuses: Mapping[str, ManifestStatusUpdate | Mapping[str, object]], run_id: str
    ) -> ManifestStatusUpdate:
        value = statuses.get(run_id)
        if value is None:
            return ManifestStatusUpdate(status="pending")
        if isinstance(value, ManifestStatusUpdate):
            return value
        return ManifestStatusUpdate.from_mapping(value, context=f"status[{run_id}]")

    def build_manifest(
        self,
        args: argparse.Namespace,
        runs: Sequence[AblationRun],
        statuses: Mapping[str, ManifestStatusUpdate | Mapping[str, object]]
        | None = None,
    ) -> AblationManifest:
        status_map = statuses or {}
        records = []
        for run in runs:
            status = self._status(status_map, run.run_id)
            records.append(
                AblationRunRecord(
                    run_id=run.run_id,
                    variant_id=run.variant.id,
                    variant=dict(run.variant.manifest),
                    repeat_index=run.repeat_index,
                    seed=run.seed,
                    experiment_tag=run.experiment_tag,
                    artifacts=ArtifactRefs.from_mapping(artifact_paths(run.artifacts)),
                    status=status.status,
                    error_message=status.error_message,
                    training_budget=status.training_budget,
                )
            )
        return build_manifest_payload(
            matrix_id=self.spec.matrix_id,
            matrix_config=self.matrix_config(args),
            training_signature=self.training_signature(args),
            runs=records,
            schema_version=self.spec.schema_version,
        ).with_output_root(str(args.output_root))

    def manifest_store(self, output_root: str | os.PathLike[str]) -> ManifestStore:
        return ManifestStore(
            output_root,
            matrix_id=self.spec.matrix_id,
            filename=self.spec.manifest_filename,
            schema_version=self.spec.schema_version,
        )

    def load_manifest(self, output_root: str | os.PathLike[str]) -> AblationManifest:
        return self.manifest_store(output_root).load()

    def validate_manifest(
        self, manifest: AblationManifest, args: argparse.Namespace
    ) -> None:
        self.manifest_store(args.output_root).validate_compatibility(
            manifest,
            matrix_config=self.matrix_config(args),
            training_signature=self.training_signature(args),
        )

    def run_train(self, args: argparse.Namespace) -> int:
        selected_runs = self.resolve_runs(args)
        manifest_runs_all = (
            self.resolve_runs(args, [variant.id for variant in self.spec.variants])
            if self.spec.all_variants_in_manifest
            else selected_runs
        )
        return execute_matrix(
            runs=selected_runs,
            store=self.manifest_store(args.output_root),
            build_manifest=lambda statuses: self.build_manifest(
                args, manifest_runs_all, statuses
            ),
            run_id_of=lambda run: run.run_id,
            required_artifacts=lambda run: canonical_artifacts_complete(run.artifacts),
            train_one=lambda run: self.train_experiment(
                run.train_args, spec=run.training_spec
            ),
            evaluate_one=lambda _run, training_spec: self.evaluate_experiment(
                training_spec
            ),
            validate_existing=lambda manifest: self.validate_manifest(manifest, args),
            resume=bool(getattr(args, "resume", False)),
            force_new=bool(getattr(args, "force_new", False)),
            dry_run=bool(args.dry_run),
            print_run=lambda run, action: print(
                f"{action:4s} "
                + self.spec.run_label_template.format(
                    variant_id=run.variant.id,
                    variant_label=run.variant.label,
                    seed=run.seed,
                    repeat_number=run.repeat_index + 1,
                    output_dir=run.training_spec.output_dir,
                    **dict(run.variant.manifest),
                )
            ),
        )

    @staticmethod
    def _entry_matches_variant(entry: AblationRunRecord, variant: VariantSpec) -> bool:
        if entry.variant_id == variant.id:
            return True
        return all(
            entry.variant.get(key) == value for key, value in variant.manifest.items()
        )

    def build_curve_aggregates(
        self,
        manifest: AblationManifest | Mapping[str, object],
        selected: Sequence[object] | None = None,
        *,
        episode_smoothing_window: int | None = None,
    ) -> tuple[list[CurveAggregate], list[str]]:
        variants = self.resolve_variants(selected)
        smoothing_window = (
            self.spec.curve.default_smoothing_window
            if episode_smoothing_window is None
            else episode_smoothing_window
        )
        if smoothing_window < 1:
            raise ValueError("episode_smoothing_window must be >= 1")
        aggregates: list[CurveAggregate] = []
        warnings: list[str] = []
        for variant in variants:
            run_series: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
            for entry in manifest_runs(manifest):
                if not self._entry_matches_variant(entry, variant):
                    continue
                if entry.status != "completed":
                    if self.spec.curve.warn_non_completed:
                        warnings.append(
                            f"Skipped {variant.label}, repeat={entry.repeat_index} "
                            f"due to status={entry.status}."
                        )
                    continue
                try:
                    episode_artifact = None
                    evaluation = None
                    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    for metric in self.spec.curve.metrics:
                        if metric.source == "episode":
                            if episode_artifact is None:
                                artifact = load_reward_diagnostics_artifact(
                                    Path(entry.artifacts.path_for("episodes"))
                                )
                                episode_artifact = (
                                    extract_complete_episode_sequence(artifact)
                                    if self.spec.curve.episode_reader == "sequence"
                                    else extract_complete_episode_series(artifact)
                                )
                            axis = _transform(
                                getattr(episode_artifact, metric.axis), "identity"
                            )
                            data = _transform(
                                getattr(episode_artifact, metric.value),
                                metric.transform,
                            )
                        else:
                            if evaluation is None:
                                evaluation = load_evaluation_history(
                                    entry.artifacts.path_for("evaluations")
                                )
                            axis = _transform(
                                getattr(evaluation, metric.axis), "identity"
                            )
                            data = _transform(
                                getattr(evaluation, metric.value), metric.transform
                            )
                            if metric.success_only:
                                success = np.asarray(evaluation.success, dtype=bool)
                                data = np.where(success, data, np.nan)
                        if axis.shape != data.shape:
                            raise ValueError(
                                f"{metric.name} and {metric.axis} have different shapes"
                            )
                        if metric.smooth:
                            if data.size < smoothing_window:
                                raise ValueError(
                                    f"fewer than {smoothing_window} complete episodes"
                                )
                            kernel = np.ones(smoothing_window) / smoothing_window
                            data = np.convolve(data, kernel, mode="valid")
                            axis = axis[smoothing_window - 1 :]
                        values[metric.name] = (axis, data)
                    run_series.append(values)
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    warnings.append(f"Skipped {variant.label} curve: {exc}")
            if not run_series:
                continue
            metric_stats: dict[str, MetricStats] = {}
            axes: dict[str, np.ndarray] = {}
            for metric in self.spec.curve.metrics:
                series = [run[metric.name] for run in run_series]
                if metric.alignment == "indexed":
                    reference, mean, std, counts = aggregate_indexed_series(series)
                else:
                    if metric.alignment == "exact_range":
                        first = int(min(x[0] for x, _ in series))
                        last = int(max(x[-1] for x, _ in series))
                        reference = np.arange(first, last + 1, dtype=np.float64)
                    else:
                        reference = np.unique(np.concatenate([x for x, _ in series]))
                    matrix = np.vstack(
                        [align_exact(reference, x, values) for x, values in series]
                    )
                    mean, std, counts = aggregate_matrix(matrix)
                metric_stats[metric.name] = MetricStats(mean, std, counts)
                axes[metric.name] = reference
            primary_axis = axes.pop(self.spec.curve.primary_metric)
            aggregates.append(
                CurveAggregate(
                    variant_id=variant.id,
                    x_name=self.spec.curve.x_name,
                    x=primary_axis,
                    metrics=metric_stats,
                    valid_run_count=len(run_series),
                    label=variant.label,
                    color=variant.color,
                    axes=axes,
                )
            )
        return aggregates, warnings

    def resolve_metric_source(
        self, manifest: AblationManifest | Mapping[str, object]
    ) -> str:
        if self.spec.final.source != "auto":
            return self.spec.final.source
        for entry in manifest_runs(manifest):
            if (
                entry.status == "completed"
                and entry.artifacts.metrics_best is not None
                and Path(entry.artifacts.metrics_best).is_file()
            ):
                return "best"
        return "final"

    def build_final_aggregates(
        self,
        manifest: AblationManifest | Mapping[str, object],
        selected: Sequence[object] | None = None,
        *,
        metric_source: str | None = None,
    ) -> tuple[list[FinalMetricAggregate], list[str]]:
        variants = self.resolve_variants(selected)
        source = metric_source or self.resolve_metric_source(manifest)
        artifact_name = "metrics_best" if source == "best" else "metrics_final"
        aggregates: list[FinalMetricAggregate] = []
        warnings: list[str] = []
        for variant in variants:
            values = {metric.name: [] for metric in self.spec.final.metrics}
            successes: list[float] = []
            for entry in manifest_runs(manifest):
                if not self._entry_matches_variant(entry, variant):
                    continue
                if entry.status != "completed":
                    if self.spec.final.warn_non_completed:
                        warnings.append(
                            f"Skipped {source} metrics for {variant.label}, "
                            f"repeat={entry.repeat_index} due to status={entry.status}."
                        )
                    continue
                try:
                    metrics = load_evaluation_metrics(
                        Path(entry.artifacts.path_for(artifact_name))
                    )
                    successes.append(float(metrics.success))
                    for metric in self.spec.final.metrics:
                        value = _transform(
                            [getattr(metrics, metric.value)], metric.transform
                        )[0]
                        values[metric.name].append(float(value))
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    warnings.append(f"Skipped {variant.label} {source} metrics: {exc}")
            if not successes:
                continue
            stats: dict[str, MetricStats] = {}
            for name, items in values.items():
                mean, std, count = aggregate_matrix(
                    np.asarray(items, dtype=np.float64).reshape(-1, 1)
                )
                stats[name] = MetricStats(mean, std, count)
            aggregates.append(
                FinalMetricAggregate(
                    variant_id=variant.id,
                    metrics=stats,
                    valid_run_count=len(successes),
                    success_rate=float(np.mean(successes)),
                    label=variant.label,
                    color=variant.color,
                )
            )
        return aggregates, warnings


def execute_matrix(
    *,
    runs: Sequence[Any],
    store: ManifestStore,
    build_manifest: Callable[
        [Mapping[str, ManifestStatusUpdate]], AblationManifest | Mapping[str, object]
    ],
    run_id_of: Callable[[Any], str],
    required_artifacts: Callable[[Any], bool],
    train_one: Callable[[Any], Any],
    evaluate_one: Callable[[Any, Any], Any],
    validate_existing: Callable[[AblationManifest], None] | None = None,
    resume: bool,
    force_new: bool = False,
    dry_run: bool,
    print_run: Callable[[Any, str], None] | None = None,
) -> int:
    """Execute runs sequentially with an explicit manifest state machine.

    The runner is intentionally single-writer.  Atomic manifest replacement
    protects the file from partial writes, while callers must not launch two
    matrix runners against the same output root concurrently.
    """
    if resume and force_new:
        raise ValueError("--resume and --force-new cannot be used together")

    existing: AblationManifest | None = None
    existing_on_disk = store.exists()
    if existing_on_disk and not force_new:
        existing = store.load()
        if resume and validate_existing is not None:
            validate_existing(existing)
        elif not resume:
            raise FileExistsError(
                f"Manifest already exists: {store.path}; use --resume or --force-new"
            )

    if resume and existing is not None:
        manifest_run_ids = {entry.run_id for entry in existing.runs}
        requested_run_ids = {run_id_of(run) for run in runs}
        missing_run_ids = requested_run_ids - manifest_run_ids
        if missing_run_ids:
            missing = ", ".join(sorted(str(run_id) for run_id in missing_run_ids))
            raise ValueError(
                f"Existing {store.matrix_id} manifest is missing run ids: {missing}"
            )

    if dry_run:
        for run in runs:
            if print_run is not None:
                print_run(run, "DRY")
        return 0

    if force_new and existing_on_disk:
        store.archive_existing()

    statuses: dict[str, ManifestStatusUpdate] = (
        status_map(existing) if existing is not None and resume else {}
    )

    def save_manifest() -> None:
        built = build_manifest(statuses)
        if isinstance(built, AblationManifest):
            # The typed builder is authoritative.  Reapplying the status map
            # also keeps generic test/build callbacks from mutating DTO views.
            built = built.with_statuses(statuses)
        store.save_atomic(built)

    save_manifest()

    for run in runs:
        run_id = run_id_of(run)
        previous = statuses.get(run_id)
        if (
            resume
            and previous is not None
            and previous.status == "completed"
            and required_artifacts(run)
        ):
            if print_run is not None:
                print_run(run, "SKIP")
            continue

        if print_run is not None:
            print_run(run, "RUN")
        statuses[run_id] = ManifestStatusUpdate(status="running")
        save_manifest()
        try:
            trained = train_one(run)
            evaluate_one(run, trained)
            if not required_artifacts(run):
                raise RuntimeError(
                    f"required artifacts are incomplete for run_id={run_id}"
                )
        except Exception as exc:
            statuses[run_id] = ManifestStatusUpdate(
                status="failed", error_message=str(exc)
            )
            save_manifest()
            print(f"FAILED run_id={run_id}: {exc}")
            return 1

        completed_status = ManifestStatusUpdate(status="completed")
        metadata = getattr(trained, "run_metadata", None)
        training_budget = getattr(metadata, "training_budget", None)
        if training_budget is not None:
            completed_status = ManifestStatusUpdate(
                status="completed", training_budget=training_budget
            )
        statuses[run_id] = completed_status
        save_manifest()

    return 0
