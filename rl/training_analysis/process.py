from __future__ import annotations

from typing import Any

import numpy as np

from .collect import ScalarSeries


def sanitize_tag(tag: str) -> str:
    return tag.replace("/", "__")


def exponential_moving_average(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values

    alpha = float(np.clip(alpha, 1e-6, 1.0))
    ema = np.empty_like(values)
    ema[0] = values[0]
    for idx in range(1, values.size):
        ema[idx] = alpha * values[idx] + (1.0 - alpha) * ema[idx - 1]
    return ema


def coefficient_of_variation(values: np.ndarray, eps: float = 1e-8) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    return std_value / max(abs(mean_value), eps)


def linear_slope(steps: np.ndarray, values: np.ndarray) -> float:
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if steps.size < 2 or values.size < 2:
        return 0.0
    if float(np.max(steps) - np.min(steps)) < 1e-12:
        return 0.0
    slope = np.polyfit(steps, values, 1)[0]
    return float(slope)


def series_window_stats(
    series: ScalarSeries, start_step: int, end_step: int
) -> dict[str, float]:
    mask = (series.steps >= start_step) & (series.steps < end_step)
    if not np.any(mask):
        return {"count": 0.0}

    values = series.values[mask]
    steps = series.steps[mask]
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "last": float(values[-1]),
        "slope": float(linear_slope(steps, values)),
        "cv": float(coefficient_of_variation(values)),
    }


def build_step_windows(max_step: int, window_size: int) -> list[tuple[int, int]]:
    if max_step < 0:
        return []
    window = max(1, int(window_size))
    windows: list[tuple[int, int]] = []
    start = 0
    while start <= max_step:
        end = start + window
        windows.append((start, end))
        start = end
    return windows


def build_step_snapshots(
    series_map: dict[str, ScalarSeries],
    selected_tags: list[str],
    step_window_size: int,
) -> list[dict[str, Any]]:
    present_tags = [tag for tag in selected_tags if tag in series_map]
    if not present_tags:
        return []

    max_step = max(int(series_map[tag].steps[-1]) for tag in present_tags)
    snapshots: list[dict[str, Any]] = []

    for index, (start_step, end_step) in enumerate(
        build_step_windows(max_step, step_window_size)
    ):
        metrics: dict[str, dict[str, float]] = {}
        sample_count = 0
        for tag in present_tags:
            stats = series_window_stats(series_map[tag], start_step, end_step)
            if stats.get("count", 0.0) <= 0.0:
                continue
            metrics[tag] = stats
            sample_count += int(stats["count"])

        if not metrics:
            continue

        snapshots.append(
            {
                "window_type": "step",
                "window_index": index,
                "step_start": start_step,
                "step_end": end_step,
                "sample_count": sample_count,
                "metrics": metrics,
            }
        )

    return snapshots


def build_episode_snapshots(
    series_map: dict[str, ScalarSeries],
    selected_tags: list[str],
    episode_window_size: int,
) -> list[dict[str, Any]]:
    episode_tag = "state/episode_id"
    if episode_tag not in series_map:
        return []

    episode_series = series_map[episode_tag]
    if episode_series.values.size == 0:
        return []

    present_tags = [tag for tag in selected_tags if tag in series_map]
    if not present_tags:
        return []

    episode_ids = np.rint(episode_series.values).astype(np.int64)
    min_episode = int(np.min(episode_ids))
    max_episode = int(np.max(episode_ids))
    window = max(1, int(episode_window_size))

    snapshots: list[dict[str, Any]] = []
    window_index = 0

    for episode_start in range(min_episode, max_episode + 1, window):
        episode_end = episode_start + window
        episode_mask = (episode_ids >= episode_start) & (episode_ids < episode_end)
        if not np.any(episode_mask):
            continue

        step_start = int(np.min(episode_series.steps[episode_mask]))
        step_end = int(np.max(episode_series.steps[episode_mask])) + 1

        metrics: dict[str, dict[str, float]] = {}
        sample_count = 0
        for tag in present_tags:
            stats = series_window_stats(series_map[tag], step_start, step_end)
            if stats.get("count", 0.0) <= 0.0:
                continue
            metrics[tag] = stats
            sample_count += int(stats["count"])

        if not metrics:
            continue

        snapshots.append(
            {
                "window_type": "episode",
                "window_index": window_index,
                "episode_start": episode_start,
                "episode_end": episode_end,
                "step_start": step_start,
                "step_end": step_end,
                "sample_count": sample_count,
                "metrics": metrics,
            }
        )
        window_index += 1

    return snapshots


def align_tags_to_reference_steps(
    series_map: dict[str, ScalarSeries],
    tags: list[str],
    reference_tag: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    available_tags = [tag for tag in tags if tag in series_map]
    if not available_tags:
        return np.asarray([], dtype=np.int64), {}

    if reference_tag is None or reference_tag not in series_map:
        reference_tag = available_tags[0]

    reference_steps = series_map[reference_tag].steps.astype(np.int64)
    aligned_values: dict[str, np.ndarray] = {}

    for tag in available_tags:
        series = series_map[tag]
        if series.steps.size == 0:
            continue

        if np.array_equal(series.steps, reference_steps):
            values = series.values.copy()
        else:
            values = np.interp(
                reference_steps.astype(np.float64),
                series.steps.astype(np.float64),
                series.values.astype(np.float64),
            )
        aligned_values[tag] = values

    return reference_steps, aligned_values
