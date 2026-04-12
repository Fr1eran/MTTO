from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


@dataclass(frozen=True)
class ScalarSeries:
    tag: str
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray


DEFAULT_SAMPLING_HEALTH_TAGS = [
    "rollout/ep_rew_mean",
    "state/episode_id",
    "constraint/is_truncated",
    "constraint/violation_code",
    "rewards/safety",
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


def compute_sampling_health(
    series_map: dict[str, ScalarSeries],
    *,
    key_tags: list[str] | None = None,
) -> dict[str, Any]:
    tags = key_tags or DEFAULT_SAMPLING_HEALTH_TAGS
    available_tags = [tag for tag in tags if tag in series_map]

    if not available_tags:
        return {
            "available": False,
            "total_step_span": 0.0,
            "tag_metrics": {},
            "summary": {},
        }

    global_min_step = min(int(np.min(series_map[tag].steps)) for tag in available_tags)
    global_max_step = max(int(np.max(series_map[tag].steps)) for tag in available_tags)
    total_step_span = max(1, global_max_step - global_min_step)

    tag_metrics: dict[str, dict[str, float]] = {}
    samples_per_10k_values: list[float] = []
    mean_gap_values: list[float] = []
    p95_gap_values: list[float] = []
    max_gap_values: list[float] = []

    for tag in available_tags:
        steps = series_map[tag].steps.astype(np.int64)
        sample_count = int(steps.size)
        if sample_count <= 1:
            mean_gap = 0.0
            p95_gap = 0.0
            max_gap = 0.0
        else:
            gaps = np.diff(steps).astype(np.float64)
            mean_gap = float(np.mean(gaps))
            p95_gap = float(np.quantile(gaps, 0.95))
            max_gap = float(np.max(gaps))

        samples_per_10k = float(sample_count) * 10000.0 / float(total_step_span)
        samples_per_10k_values.append(samples_per_10k)
        mean_gap_values.append(mean_gap)
        p95_gap_values.append(p95_gap)
        max_gap_values.append(max_gap)

        tag_metrics[tag] = {
            "sample_count": float(sample_count),
            "mean_step_gap": mean_gap,
            "p95_step_gap": p95_gap,
            "max_step_gap": max_gap,
            "samples_per_10k_steps": samples_per_10k,
            "step_start": float(int(steps[0])) if sample_count > 0 else 0.0,
            "step_end": float(int(steps[-1])) if sample_count > 0 else 0.0,
        }

    summary = {
        "observed_tag_count": float(len(available_tags)),
        "min_sample_count": float(
            min(int(tag_metrics[tag]["sample_count"]) for tag in available_tags)
        ),
        "mean_samples_per_10k_steps": float(np.mean(samples_per_10k_values)),
        "max_mean_step_gap": float(np.max(mean_gap_values)),
        "max_p95_step_gap": float(np.max(p95_gap_values)),
        "max_max_step_gap": float(np.max(max_gap_values)),
    }

    return {
        "available": True,
        "total_step_span": float(total_step_span),
        "tag_metrics": tag_metrics,
        "summary": summary,
    }
