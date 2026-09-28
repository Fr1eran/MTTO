"""Paper figures and shared paper scenario paths."""

from dataclasses import replace
from pathlib import Path

from mtto.domain.scenario import Scenario, Task
from mtto.io.scenario import load_scenario, load_tasks

ROOT = Path(__file__).resolve().parents[2]
LINE_DIR = ROOT / "paper/data/line"


def load_paper_scenario() -> Scenario:
    return load_scenario(ROOT / "paper/specs/scenario.toml", LINE_DIR)


def load_paper_task(schedule_time_s: float | None = None) -> Task:
    task = load_tasks(ROOT / "paper/specs/tasks.toml")["longyang_to_airport"]
    if schedule_time_s is not None:
        return replace(task, schedule_time_s=float(schedule_time_s))
    return task
