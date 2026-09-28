"""TensorBoard event log reading routines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from mtto.rl.training_analysis.collect import (
    ScalarSeries,
    _sort_and_keep_latest_by_step,
)

__all__ = [
    "list_run_directories",
    "resolve_run_directory",
    "load_scalar_series_from_run",
]


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
        if candidate.exists() and candidate.is_dir():
            return candidate

        # SB3 会在 tb_log_name 后追加 _1、_2 等后缀
        # 因此精确匹配失败时按前缀查找最新的匹配目录
        run_dirs = list_run_directories(root)
        run_name_lower = candidate.name.lower()
        matching = [
            d
            for d in run_dirs
            if d.name.lower() == run_name_lower
            or d.name.lower().startswith(run_name_lower + "_")
        ]
        if matching:
            return matching[-1]
        raise FileNotFoundError(f"Run directory not found: {candidate}")

    run_dirs = list_run_directories(root)
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard run directories found in: {root}")
    return run_dirs[-1]


def load_scalar_series_from_run(run_dir: str | Path) -> dict[str, ScalarSeries]:
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    accumulator = event_accumulator.EventAccumulator(
        str(run_path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _ = accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    if not isinstance(scalar_tags, list):
        scalar_tags = []
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
