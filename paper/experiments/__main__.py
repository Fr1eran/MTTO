"""Command line entry point for paper experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper.experiments import method_ablation, schedule_change, step_time
from paper.experiments.runner import completed_matrix
from paper.experiments.spec import load_experiment_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper experiment runner")
    experiments = parser.add_subparsers(dest="experiment", required=True)
    defaults = {
        "method_ablation": "paper/specs/method_ablation.toml",
        "step_time": "paper/specs/step_time.toml",
        "schedule_change": "paper/specs/schedule_change.toml",
    }
    for name, default_spec in defaults.items():
        experiment = experiments.add_parser(name)
        actions = experiment.add_subparsers(dest="action", required=True)
        for action in ("run", "summarize", "figures"):
            command = actions.add_parser(action)
            command.add_argument("--spec", type=Path, default=Path(default_spec))
            if action != "run":
                command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.experiment == "schedule_change":
        if args.action == "run":
            results = schedule_change.run(args.spec)
            for result in results:
                print(
                    f"{'reused' if result.reused else 'evaluated'}: {result.directory}"
                )
            return
        spec = schedule_change.load_schedule_spec(args.spec)
        run_dirs = schedule_change.completed_runs(spec)
        summary = schedule_change.summarize(spec, run_dirs)
        if args.action == "summarize":
            schedule_change.write_summary(summary, args.output)
        else:
            from paper.plotting.ablation import require_clean, schedule_change_figure

            source_spec = load_experiment_spec(spec.source_spec)
            require_clean((*completed_matrix(source_spec), *run_dirs))
            scenario, _ = schedule_change.planned_evaluations(spec)
            schedule_change_figure(summary, scenario, args.output)
    else:
        module = method_ablation if args.experiment == "method_ablation" else step_time
        if args.action == "run":
            results = module.run(args.spec)
            for result in results:
                print(f"{'reused' if result.reused else 'trained'}: {result.directory}")
            return
        spec = load_experiment_spec(args.spec)
        run_dirs = completed_matrix(spec)
        summary = module.summarize(spec, run_dirs)
        if args.action == "summarize":
            module.write_summary(summary, args.output)
        else:
            from paper.plotting.ablation import (
                method_figures,
                require_clean,
                step_time_figure,
            )

            require_clean(run_dirs)
            if args.experiment == "method_ablation":
                method_figures(summary, spec, args.output)
            else:
                step_time_figure(summary, spec, args.output)
    print(f"{args.action} written to {args.output}")


if __name__ == "__main__":
    main()
