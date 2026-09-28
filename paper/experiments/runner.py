"""Resume and reuse completed paper training and evaluation runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mtto
from mtto.domain.scenario import Scenario, Task
from mtto.io.artifacts import (
    BEST_DIR,
    POLICY_ZIP,
    RUN_JSON,
    ArtifactError,
    CompletedRun,
    RunKind,
    RunRecord,
    canonical_json,
    file_sha256,
    read_completed_run,
    task_to_json,
)
from mtto.io.scenario import load_scenario, load_tasks
from mtto.workflows import (
    evaluate as evaluate_workflow,
    train as train_workflow,
)
from mtto.workflows.evaluate import EvaluateConfig, evaluation_record_config
from mtto.workflows.train import training_record_config
from paper.experiments.spec import ExperimentSpec, PlannedRun, expand_matrix

PAPER_JSON = "paper.json"


class ExperimentStopped(RuntimeError):
    """A completed training run could not reach its requested budget."""


@dataclass(frozen=True)
class RunResult:
    planned: PlannedRun
    directory: Path
    reused: bool


@dataclass(frozen=True)
class PlannedEvaluation:
    run_label: str
    policy_run_dir: Path
    task: Task
    config: EvaluateConfig


@dataclass(frozen=True)
class EvaluationRunResult:
    planned: PlannedEvaluation
    directory: Path
    reused: bool


def git_state() -> tuple[str, bool]:
    """Capture provenance at the beginning of a matrix run."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def reuse_key(record: RunRecord, policy_sha256: str | None = None) -> str:
    """Hash the exact persisted run inputs, with optional evaluated policy identity."""
    inputs = {
        "config": record.config,
        "scenario_hash": record.scenario_hash,
        "task": record.task,
        "mtto_version": record.mtto_version,
    }
    if policy_sha256 is not None:
        inputs["policy_sha256"] = policy_sha256
    return hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()


def _record_key(record: RunRecord, source_dir: Path | None = None) -> str:
    policy_sha = (
        file_sha256(source_dir / record.config["source_policy_path"])
        if source_dir is not None
        else None
    )
    return reuse_key(record, policy_sha)


def _read_paper(
    path: Path,
    record: RunRecord,
    source_dir: Path | None = None,
    *,
    preserve_stale_key: bool = False,
) -> dict[str, object]:
    paper = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(paper, dict) or set(paper) != {
        "reuse_key",
        "git_commit",
        "dirty",
    }:
        raise ValueError("Invalid paper.json fields")
    if not isinstance(paper["dirty"], bool):
        raise ValueError("Invalid paper.json provenance")
    if not preserve_stale_key and paper["reuse_key"] != _record_key(record, source_dir):
        raise ValueError("Invalid paper.json provenance")
    return paper


def _write_paper(path: Path, content: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".paper_", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(content))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _expected_key(planned: PlannedRun, scenario_hash: str, task: dict) -> str:
    return reuse_key(
        RunRecord(
            run_id="",
            kind="rl_train",
            config=training_record_config(planned.config),
            scenario_hash=scenario_hash,
            task=task,
            policy_io_version=None,
            mtto_version=mtto.__version__,
            created_at="",
        )
    )


def _execute_one(
    output_root: Path,
    run_label: str,
    expected: str,
    provenance: tuple[str, bool],
    execute: Callable[[Path], object],
    source_dir: Path | None = None,
) -> tuple[Path, bool, CompletedRun]:
    commit, dirty = provenance
    pattern = re.compile(rf"{re.escape(run_label)}__(\d{{2}})$")
    existing = []
    for path in output_root.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            existing.append((int(match.group(1)), path))
    max_number = max((number for number, _ in existing), default=0)
    reusable = None
    for _, path in sorted(existing):
        if not (path / RUN_JSON).exists():
            shutil.rmtree(path)
            continue
        try:
            completed = read_completed_run(path)
        except ArtifactError:
            shutil.rmtree(path)
            continue
        paper_path = path / PAPER_JSON
        if not paper_path.exists():
            shutil.rmtree(path)
            continue
        try:
            paper = _read_paper(
                paper_path,
                completed.record,
                source_dir,
                preserve_stale_key=source_dir is not None,
            )
        except ValueError:
            shutil.rmtree(path)
            continue
        if (
            paper["reuse_key"] == expected
            and _record_key(completed.record, source_dir) == expected
            and (dirty or not paper["dirty"])
        ):
            reusable = path
    if reusable is not None:
        return reusable, True, read_completed_run(reusable)
    path = output_root / f"{run_label}__{max_number + 1:02d}"
    execute(path)
    completed = read_completed_run(path)
    _write_paper(
        path / PAPER_JSON,
        {
            "reuse_key": _record_key(completed.record, source_dir),
            "git_commit": commit,
            "dirty": dirty,
        },
    )
    return path, False, completed


def completed_matrix(spec: ExperimentSpec) -> tuple[Path, ...]:
    """Read the latest valid result for each planned run without training."""
    scenario = load_scenario(spec.scenario, spec.line_dir)
    task = task_to_json(load_tasks(spec.tasks)[spec.task])
    directories = []
    for planned in expand_matrix(spec):
        expected = _expected_key(planned, scenario.scenario_hash, task)
        pattern = re.compile(rf"{re.escape(planned.run_label)}__(\d{{2}})$")
        matches = []
        for path in spec.output_root.iterdir():
            match = pattern.fullmatch(path.name)
            if not match or not path.is_dir() or path.is_symlink():
                continue
            try:
                completed = read_completed_run(path)
            except ArtifactError:
                continue
            paper_path = path / PAPER_JSON
            if not paper_path.exists():
                continue
            try:
                paper = _read_paper(paper_path, completed.record)
            except ValueError:
                continue
            if paper["reuse_key"] == expected:
                matches.append((int(match.group(1)), path))
        if not matches:
            raise FileNotFoundError(f"No completed run for {planned.run_label}")
        directories.append(max(matches)[1])
    return tuple(directories)


def execute_matrix(spec: ExperimentSpec) -> tuple[RunResult, ...]:
    """Delete only interrupted matching directories and reuse complete results."""
    commit, dirty = git_state()
    scenario = load_scenario(spec.scenario, spec.line_dir)
    task = load_tasks(spec.tasks)[spec.task]
    spec.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for planned in expand_matrix(spec):
        expected = _expected_key(planned, scenario.scenario_hash, task_to_json(task))
        path, reused, completed = _execute_one(
            spec.output_root,
            planned.run_label,
            expected,
            (commit, dirty),
            lambda output, planned=planned: train_workflow.train(
                scenario, task, planned.config, output, run_id=str(uuid.uuid4())
            ),
        )
        training = completed.payload.result.training
        if not training.target_reached:
            raise ExperimentStopped(f"Training budget not reached: {path}")
        results.append(RunResult(planned=planned, directory=path, reused=reused))
    return tuple(results)


def execute_evaluations(
    output_root: Path,
    scenario: Scenario,
    planned_runs: tuple[PlannedEvaluation, ...],
) -> tuple[EvaluationRunResult, ...]:
    """Run or reuse evaluations with the training recovery rules."""
    provenance = git_state()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for planned in planned_runs:
        expected = _expected_evaluation_key(planned, scenario)
        path, reused, _ = _execute_one(
            output_root,
            planned.run_label,
            expected,
            provenance,
            lambda output, planned=planned: evaluate_workflow.evaluate(
                scenario,
                planned.task,
                planned.policy_run_dir,
                planned.config,
                output,
                run_id=str(uuid.uuid4()),
            ),
            planned.policy_run_dir,
        )
        results.append(EvaluationRunResult(planned, path, reused))
    return tuple(results)


def _expected_evaluation_key(planned: PlannedEvaluation, scenario: Scenario) -> str:
    source = read_completed_run(planned.policy_run_dir)
    if source.record.kind != RunKind.RL_TRAIN:
        raise ValueError(f"Source is not a training run: {planned.policy_run_dir}")
    if planned.config.use_best and source.payload.best is None:
        raise ValueError(f"Source has no best policy: {planned.policy_run_dir}")
    policy_path = (
        planned.policy_run_dir / BEST_DIR / POLICY_ZIP
        if planned.config.use_best
        else planned.policy_run_dir / POLICY_ZIP
    )
    return reuse_key(
        RunRecord(
            run_id="",
            kind=RunKind.EVALUATION,
            config=evaluation_record_config(planned.config, source.record.run_id),
            scenario_hash=scenario.scenario_hash,
            task=task_to_json(planned.task),
            policy_io_version=None,
            mtto_version=mtto.__version__,
            created_at="",
        ),
        file_sha256(policy_path),
    )


def completed_evaluations(
    output_root: Path,
    scenario: Scenario,
    planned_runs: tuple[PlannedEvaluation, ...],
) -> tuple[Path, ...]:
    """Find matching complete evaluation artifacts without executing workflows."""
    directories = []
    for planned in planned_runs:
        expected = _expected_evaluation_key(planned, scenario)
        pattern = re.compile(rf"{re.escape(planned.run_label)}__(\d{{2}})$")
        matches = []
        for path in output_root.iterdir():
            match = pattern.fullmatch(path.name)
            if not match or not path.is_dir() or path.is_symlink():
                continue
            try:
                completed = read_completed_run(path)
            except ArtifactError:
                continue
            paper_path = path / PAPER_JSON
            if not paper_path.exists():
                continue
            try:
                paper = _read_paper(
                    paper_path, completed.record, planned.policy_run_dir
                )
            except ValueError:
                continue
            if paper["reuse_key"] == expected:
                matches.append((int(match.group(1)), path))
        if not matches:
            raise FileNotFoundError(f"No completed evaluation for {planned.run_label}")
        directories.append(max(matches)[1])
    return tuple(directories)
