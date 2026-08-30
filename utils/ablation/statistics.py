"""Numerical alignment and aggregation primitives for ablation reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rl.training_analysis.process import trailing_moving_average


def align_exact(
    reference: NDArray[np.float64],
    keys: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Align values to exact reference keys without interpolation."""
    reference = np.asarray(reference, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if (
        reference.ndim != 1
        or keys.ndim != 1
        or values.ndim != 1
        or keys.size != values.size
    ):
        raise ValueError(
            "reference, keys and values must be one-dimensional; "
            "keys and values must be equally sized"
        )
    result = np.full(reference.shape, np.nan, dtype=np.float64)
    if keys.size == 0 or reference.size == 0:
        return result
    indices = np.searchsorted(reference, keys)
    valid = indices < reference.size
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        valid[valid_indices] = reference[indices[valid_indices]] == keys[valid_indices]
    result[indices[valid]] = values[valid]
    return result


def smooth_episode_curve(
    episode_numbers: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    window: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply trailing moving average and retain the corresponding x-axis."""
    if window < 1:
        raise ValueError("episode_smoothing_window must be >= 1")
    episodes = np.asarray(episode_numbers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if episodes.ndim != 1 or values.ndim != 1 or episodes.size != values.size:
        raise ValueError("episode_numbers and values must be one-dimensional and equal")
    if values.size < window:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return episodes[window - 1 :], trailing_moving_average(values, window)


def aggregate_matrix(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return NaN-aware mean, sample std and point-wise valid counts."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    finite_matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    counts = np.sum(np.isfinite(finite_matrix), axis=0, dtype=np.int64)
    means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    if np.any(valid):
        means[valid] = np.nansum(finite_matrix[:, valid], axis=0) / counts[valid]
    stds = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    multiple = counts >= 2
    if np.any(multiple):
        stds[multiple] = np.nanstd(finite_matrix[:, multiple], axis=0, ddof=1)
    stds[counts == 1] = 0.0
    return means, stds, counts


def aggregate_indexed_series(
    series: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Aggregate series by their index, preserving missing tails."""
    if not series:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.int64)
    normalized: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for x_values, values in series:
        x_array = np.asarray(x_values, dtype=np.float64)
        value_array = np.asarray(values, dtype=np.float64)
        if (
            x_array.ndim != 1
            or value_array.ndim != 1
            or x_array.size != value_array.size
        ):
            raise ValueError("indexed series must contain equally sized 1-D arrays")
        normalized.append((x_array, value_array))

    max_length = max(values.size for _, values in normalized)
    if max_length == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.int64)
    x_matrix = np.full((len(normalized), max_length), np.nan, dtype=np.float64)
    value_matrix = np.full_like(x_matrix, np.nan)
    for row, (x_values, values) in enumerate(normalized):
        count = min(x_values.size, max_length)
        x_matrix[row, :count] = x_values[:count]
        value_matrix[row, :count] = values[:count]
    x_mean, _, _ = aggregate_matrix(x_matrix)
    mean, std, counts = aggregate_matrix(value_matrix)
    return x_mean, mean, std, counts
