"""Strict TOML definitions for paper training matrices."""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path

from mtto.workflows.train import TrainConfig

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KEYS = {
    "name",
    "scenario",
    "line_dir",
    "tasks",
    "task",
    "output_root",
    "seeds",
}
TRAIN_KEYS = {field.name for field in dataclasses.fields(TrainConfig)} - {"seed"}
NULLABLE_TRAIN_KEYS = {
    name
    for name, hint in typing.get_type_hints(TrainConfig).items()
    if name in TRAIN_KEYS and type(None) in typing.get_args(hint)
}


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    overrides: dict[str, object]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    scenario: Path
    line_dir: Path
    tasks: Path
    task: str
    output_root: Path
    seeds: tuple[int, ...]
    train: dict[str, object]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class PlannedRun:
    run_label: str
    config: TrainConfig
    variant: Variant
    seed: int


def _require_keys(data: dict[str, object], expected: set[str], section: str) -> None:
    if set(data) != expected:
        raise ValueError(
            f"{section}: missing {sorted(expected - set(data))}, "
            f"extra {sorted(set(data) - expected)}"
        )


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Load a complete experiment definition without implicit train defaults."""
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    _require_keys(data, {"experiment", "train", "variants"}, "root")
    experiment = data["experiment"]
    train = data["train"]
    _require_keys(experiment, EXPERIMENT_KEYS, "experiment")
    if not experiment["seeds"]:
        raise ValueError("seeds must be nonempty")
    extra = set(train) - TRAIN_KEYS
    if extra:
        raise ValueError(f"train: extra {sorted(extra)}")
    variants = []
    for raw in data["variants"]:
        overrides = {
            key: value for key, value in raw.items() if key not in {"id", "label"}
        }
        extra = set(overrides) - TRAIN_KEYS
        if "id" not in raw or "label" not in raw or extra:
            raise ValueError(f"variant: missing id/label or extra {sorted(extra)}")
        merged = {**dict.fromkeys(NULLABLE_TRAIN_KEYS), **train, **overrides}
        missing = TRAIN_KEYS - set(merged)
        if missing:
            raise ValueError(f"variant {raw['id']}: missing {sorted(missing)}")
        TrainConfig(seed=experiment["seeds"][0], **merged)
        variants.append(Variant(raw["id"], raw["label"], overrides))
    if not variants:
        raise ValueError("variants must be nonempty")
    if len({variant.id for variant in variants}) != len(variants):
        raise ValueError("variant ids must be unique")
    if len(set(experiment["seeds"])) != len(experiment["seeds"]):
        raise ValueError("seeds must be unique")
    return ExperimentSpec(
        name=experiment["name"],
        scenario=ROOT / experiment["scenario"],
        line_dir=ROOT / experiment["line_dir"],
        tasks=ROOT / experiment["tasks"],
        task=experiment["task"],
        output_root=ROOT / experiment["output_root"],
        seeds=tuple(experiment["seeds"]),
        train={**dict.fromkeys(NULLABLE_TRAIN_KEYS), **train},
        variants=tuple(variants),
    )


def expand_matrix(spec: ExperimentSpec) -> tuple[PlannedRun, ...]:
    """Expand methods and seeds in source order."""
    return tuple(
        PlannedRun(
            run_label=f"{spec.name}__{variant.id}__seed{seed:04d}",
            config=TrainConfig(seed=seed, **(spec.train | variant.overrides)),
            variant=variant,
            seed=seed,
        )
        for variant in spec.variants
        for seed in spec.seeds
    )
