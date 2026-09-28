"""Command line interface for MTTO workflows."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tomllib
import types
import typing
import warnings
from pathlib import Path
from typing import Any

from tqdm import TqdmExperimentalWarning

from mtto.domain.scenario import Scenario, Task
from mtto.evaluation.quality import QualityReport
from mtto.io.artifacts import ArtifactError
from mtto.io.scenario import load_scenario, load_tasks
from mtto.workflows.analysis import AnalysisConfig, analyze_training
from mtto.workflows.dp import DPConfig, solve_dp
from mtto.workflows.evaluate import EvaluateConfig, evaluate
from mtto.workflows.train import Progress, TrainConfig, train

# tqdm.rich emits TqdmExperimentalWarning on import; suppress it as SB3's
# ProgressBarCallback does (only this warning category).
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.rich import tqdm  # noqa: E402

__all__ = ["build_parser", "main"]


def _is_float_type(hint: Any) -> bool:
    if hint is float:
        return True
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        return float in typing.get_args(hint)
    return False


def _allows_none_type(hint: Any) -> bool:
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        return type(None) in typing.get_args(hint)
    return False


def _load_config_from_toml[T](
    path: Path, section: str, dataclass_type: type[T], *, strict: bool = True
) -> T:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if section not in data:
        raise ValueError(f"Config file missing [{section}] section: {path}")
    raw = data[section]
    if not isinstance(raw, dict):
        raise ValueError(f"Section [{section}] in {path} must be a table")

    type_hints = typing.get_type_hints(dataclass_type)
    dataclass_fields = {f.name: f for f in dataclasses.fields(dataclass_type)}

    extra_keys = set(raw.keys()) - set(dataclass_fields.keys())
    if extra_keys:
        raise ValueError(
            f"Unknown config key(s) in [{section}]: {', '.join(sorted(extra_keys))}"
        )

    kwargs: dict[str, Any] = {}
    if strict:
        for name, field in dataclass_fields.items():
            hint = type_hints.get(name, field.type)
            if name not in raw:
                if not _allows_none_type(hint):
                    raise ValueError(
                        f"Missing required config key in [{section}]: {name}"
                    )
                kwargs[name] = None
            else:
                val = raw[name]
                if (
                    _is_float_type(hint)
                    and isinstance(val, int)
                    and not isinstance(val, bool)
                ):
                    val = float(val)
                kwargs[name] = val
        return dataclass_type(**kwargs)
    else:
        for k, v in raw.items():
            hint = type_hints.get(k)
            if (
                hint is not None
                and _is_float_type(hint)
                and isinstance(v, int)
                and not isinstance(v, bool)
            ):
                v = float(v)
            kwargs[k] = v
        return dataclass_type(**kwargs)


def _load_scenario_and_task(args: argparse.Namespace) -> tuple[Scenario, Task]:
    scenario = load_scenario(args.scenario, args.line_dir)
    tasks = load_tasks(args.tasks)
    if args.task not in tasks:
        available = ", ".join(sorted(tasks.keys()))
        raise KeyError(f"Task '{args.task}' not found. Available tasks: {available}")
    task = tasks[args.task]
    if args.schedule_time is not None:
        task = dataclasses.replace(task, schedule_time_s=args.schedule_time)
    return scenario, task


def _print_quality_summary(
    run_id: str, output_dir: Path, quality: QualityReport
) -> None:
    arr_err = quality.metrics.arrival_time_error_s
    arr_err_str = f"{arr_err:.2f} s" if arr_err is not None else "N/A"
    print(
        f"Run ID: {run_id}\n"
        f"Output directory: {output_dir}\n"
        f"Completed: {quality.completed}\n"
        f"Feasible: {quality.feasible}\n"
        f"Run time: {quality.metrics.run_time_s:.2f} s\n"
        f"Arrival time error: {arr_err_str}\n"
        f"Stop error: {quality.metrics.stop_error_m:.2f} m\n"
        f"Total energy: {quality.metrics.total_energy_kj:.2f} kJ"
    )


def _print_analysis_summary(output_root: Path, output_paths: dict[str, str]) -> None:
    print(f"Output directory: {output_root}")
    for kind, path in sorted(output_paths.items()):
        print(f"  {kind}: {path}")


def _cmd_train(args: argparse.Namespace) -> int:
    scenario, task = _load_scenario_and_task(args)
    config = _load_config_from_toml(args.config, "train", TrainConfig, strict=True)

    pbar: tqdm | None = None

    def _on_progress(p: Progress) -> None:
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=p.total_timesteps, desc="Training", unit="steps")
        pbar.n = min(p.training_timesteps, p.total_timesteps)
        pbar.set_postfix(episodes=p.completed_episodes)
        pbar.refresh()

    try:
        result = train(
            scenario=scenario,
            task=task,
            config=config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            tensorboard_log_dir=args.tensorboard_dir,
            progress=_on_progress,
        )
    finally:
        if pbar is not None:
            pbar.close()

    _print_quality_summary(result.run_id, args.output_dir, result.quality)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    scenario, task = _load_scenario_and_task(args)
    config = EvaluateConfig(
        use_best=args.use_best,
        deterministic=not args.stochastic,
        device=args.device,
    )
    result = evaluate(
        scenario=scenario,
        task=task,
        policy_run_dir=args.policy_run,
        config=config,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    _print_quality_summary(result.run_id, args.output_dir, result.quality)
    return 0


def _cmd_dp(args: argparse.Namespace) -> int:
    scenario, task = _load_scenario_and_task(args)
    config = _load_config_from_toml(args.config, "dp", DPConfig, strict=True)
    result = solve_dp(
        scenario=scenario,
        task=task,
        config=config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        run_id=args.run_id,
    )
    _print_quality_summary(result.run_id, args.output_dir, result.quality)
    return 0


def _cmd_analyze_training(args: argparse.Namespace) -> int:
    config = _load_config_from_toml(
        args.config, "analysis", AnalysisConfig, strict=True
    )
    result = analyze_training(
        config=config,
        output_root=args.output_root,
        run_name=args.run_name,
        train_run_dir=args.train_run,
        tensorboard_log_root=args.tensorboard_root,
        tensorboard_run_name=args.tensorboard_run,
    )
    _print_analysis_summary(args.output_root, result.output_paths)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtto",
        description="Maglev Train Trajectory Optimization CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to scenario TOML specification",
    )
    common_parser.add_argument(
        "--line-dir",
        type=Path,
        required=True,
        help="Path to line data directory containing JSON definitions",
    )
    common_parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="Path to tasks TOML specification",
    )
    common_parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Operating task name",
    )
    common_parser.add_argument(
        "--schedule-time",
        type=float,
        default=None,
        help="Optional schedule time override in seconds",
    )

    # train
    p_train = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="Train RL policy for a trajectory optimization task",
    )
    p_train.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to TOML config file with [train] table "
            "(omit keys allowing None to set them to None)"
        ),
    )
    p_train.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training run artifacts",
    )
    p_train.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory",
    )
    p_train.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID (defaults to generated UUID)",
    )
    p_train.set_defaults(func=_cmd_train)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate",
        parents=[common_parser],
        help="Evaluate a trained RL policy on a task",
    )
    p_eval.add_argument(
        "--policy-run",
        type=Path,
        required=True,
        help="Path to source completed training run directory",
    )
    p_eval.add_argument(
        "--use-best",
        action="store_true",
        default=False,
        help="Use best checkpoint policy rather than final policy",
    )
    p_eval.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic evaluation policy (default is deterministic)",
    )
    p_eval.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (default: cpu)",
    )
    p_eval.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation run artifacts",
    )
    p_eval.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID (defaults to generated UUID)",
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    # dp
    p_dp = subparsers.add_parser(
        "dp",
        parents=[common_parser],
        help="Solve trajectory optimization via dynamic programming",
    )
    p_dp.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to TOML config file with [dp] table "
            "(omit keys allowing None to set them to None)"
        ),
    )
    p_dp.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for DP run artifacts",
    )
    p_dp.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional disk cache directory for DP precomputed transitions",
    )
    p_dp.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID (defaults to generated UUID)",
    )
    p_dp.set_defaults(func=_cmd_dp)

    # analyze-training
    p_analysis = subparsers.add_parser(
        "analyze-training",
        help="Analyze training diagnostics and generate report",
    )
    p_analysis.add_argument(
        "--train-run",
        type=Path,
        required=True,
        help="Path to completed RL train run directory",
    )
    p_analysis.add_argument(
        "--tensorboard-root",
        type=Path,
        default=None,
        help="Optional TensorBoard root directory",
    )
    p_analysis.add_argument(
        "--tensorboard-run",
        type=str,
        default=None,
        help="Optional TensorBoard run name",
    )
    p_analysis.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for report identification",
    )
    p_analysis.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for generated analysis reports",
    )
    p_analysis.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to TOML config file with [analysis] table "
            "(omit keys allowing None to set them to None; "
            "see paper/specs/analysis.toml for an example)"
        ),
    )
    p_analysis.set_defaults(func=_cmd_analyze_training)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        FileExistsError,
        FileNotFoundError,
        ValueError,
        ArtifactError,
        RuntimeError,
        KeyError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
