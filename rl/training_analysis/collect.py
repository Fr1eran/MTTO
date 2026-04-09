from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


@dataclass(frozen=True)
class ScalarSeries:
    tag: str
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray


def list_run_directories(log_root: str | Path) -> list[Path]:
    root = Path(log_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime
    )


def resolve_run_directory(log_root: str | Path, run_name: str | None = None) -> Path:
    root = Path(log_root)
    if run_name:
        candidate = Path(run_name)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Run directory not found: {candidate}")
        return candidate

    run_dirs = list_run_directories(root)
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard run directories found in: {root}")
    return run_dirs[-1]


def _sort_and_keep_latest_by_step(
    steps: np.ndarray,
    values: np.ndarray,
    wall_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps.size == 0:
        return steps, values, wall_times

    order = np.argsort(steps, kind="stable")
    steps_sorted = steps[order]
    values_sorted = values[order]
    wall_times_sorted = wall_times[order]

    rev_steps = steps_sorted[::-1]
    _, rev_unique_indices = np.unique(rev_steps, return_index=True)
    keep = steps_sorted.size - 1 - rev_unique_indices
    keep.sort()

    return steps_sorted[keep], values_sorted[keep], wall_times_sorted[keep]


def load_scalar_series_from_run(run_dir: str | Path) -> dict[str, ScalarSeries]:
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    accumulator = event_accumulator.EventAccumulator(
        str(run_path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    series_map: dict[str, ScalarSeries] = {}

    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        if not events:
            continue

        steps = np.asarray([event.step for event in events], dtype=np.int64)
        values = np.asarray([event.value for event in events], dtype=np.float64)
        wall_times = np.asarray([event.wall_time for event in events], dtype=np.float64)
        steps, values, wall_times = _sort_and_keep_latest_by_step(
            steps,
            values,
            wall_times,
        )
        if steps.size == 0:
            continue

        series_map[tag] = ScalarSeries(
            tag=tag,
            steps=steps,
            values=values,
            wall_times=wall_times,
        )

    return series_map
