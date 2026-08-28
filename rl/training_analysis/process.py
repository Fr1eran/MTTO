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


def trailing_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute SB3-style trailing arithmetic moving averages.

    The returned values align with ``x[window - 1:]``, matching
    ``stable_baselines3.common.results_plotter.window_func``.
    """
    array = np.asarray(values, dtype=np.float64)
    window_size = int(window)
    if window_size < 1:
        raise ValueError("window must be >= 1")
    if array.size < window_size:
        return np.empty(0, dtype=np.float64)
    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float64)
    return np.convolve(array, kernel, mode="valid")


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
