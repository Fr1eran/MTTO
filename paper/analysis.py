"""Numerical alignment and aggregation primitives for ablation reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

SAFETY_MARGIN_EPS_MPS = 1e-6


def is_successful_evaluation(*, termination_reason: str | None) -> bool:
    """Return whether an evaluation ended in normal task completion."""
    return termination_reason == "STOPPED_IN_ZONE"


def is_precise_evaluation(
    *, success: bool, stop_error_m: float, stop_error_limit_m: float
) -> bool:
    """Return the canonical stopping-accuracy status (inclusive limit)."""
    if stop_error_limit_m < 0.0:
        raise ValueError("stop_error_limit_m must be non-negative")
    return bool(success and abs(float(stop_error_m)) <= stop_error_limit_m)


def is_punctual_evaluation(
    *, precise_arrival: bool, time_error_s: float, time_error_limit_s: float
) -> bool:
    """Return the canonical punctuality status (exclusive limit)."""
    if time_error_limit_s < 0.0:
        raise ValueError("time_error_limit_s must be non-negative")
    return bool(precise_arrival and abs(float(time_error_s)) < time_error_limit_s)


def is_safe_evaluation(
    *,
    min_safety_margin_mps: float,
    safety_violation_count: int,
    safety_margin_eps_mps: float = SAFETY_MARGIN_EPS_MPS,
) -> bool:
    """Return the canonical safety status for one evaluated trajectory."""
    if safety_violation_count < 0:
        raise ValueError("safety_violation_count must be non-negative")
    if safety_margin_eps_mps < 0.0:
        raise ValueError("safety_margin_eps_mps must be non-negative")
    return bool(
        min_safety_margin_mps >= -safety_margin_eps_mps and safety_violation_count == 0
    )


def is_feasible_evaluation(
    *,
    success: bool,
    precise_arrival: bool,
    punctual_arrival: bool,
    safe: bool,
) -> bool:
    """Return the canonical strict-feasibility status."""
    return bool(success and precise_arrival and punctual_arrival and safe)


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
    """Smooth a curve with expanding warm-up and a fixed trailing window.

    The first ``window - 1`` points use every observation available so far.
    Once the window is full, each point uses the latest ``window`` observations.
    The returned x-axis therefore always retains the original episode alignment.
    """
    if window < 1:
        raise ValueError("episode_smoothing_window must be >= 1")
    episodes = np.asarray(episode_numbers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if episodes.ndim != 1 or values.ndim != 1 or episodes.size != values.size:
        raise ValueError("episode_numbers and values must be one-dimensional and equal")
    if values.size == 0:
        return episodes.copy(), values.copy()
    order = np.argsort(episodes, kind="stable")
    episodes = episodes[order]
    values = values[order]
    window_size = int(window)
    totals = np.convolve(values, np.ones(window_size, dtype=np.float64), mode="full")[
        : values.size
    ]
    counts = np.minimum(np.arange(1, values.size + 1, dtype=np.float64), window_size)
    return episodes.copy(), totals / counts


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


def aggregate_step_binned_series(
    series: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    bin_width: int,
    axis_max: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Aggregate each run within fixed environment-transition bins."""
    if bin_width <= 0 or axis_max <= 0:
        raise ValueError("bin_width and axis_max must be positive")
    ends = np.arange(bin_width, axis_max + 1, bin_width, dtype=np.float64)
    if ends.size == 0 or ends[-1] < axis_max:
        ends = np.append(ends, float(axis_max))
    matrix = np.full((len(series), ends.size), np.nan, dtype=np.float64)
    starts = np.concatenate(([0.0], ends[:-1]))
    for row, (steps, values) in enumerate(series):
        x = np.asarray(steps, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        for column, (start, end) in enumerate(zip(starts, ends, strict=True)):
            mask = (x > start) & (x <= end) & np.isfinite(y)
            if np.any(mask):
                matrix[row, column] = float(np.mean(y[mask]))
    mean, std, counts = aggregate_matrix(matrix)
    return ends, mean, std, counts


STRICT_STOP_ERROR_LIMIT_M = 0.3
STRICT_TIME_ERROR_LIMIT_S = 10.0
DEFAULT_SAFETY_MARGIN_EPS_MPS = SAFETY_MARGIN_EPS_MPS


@dataclass(frozen=True)
class ConstraintThresholds:
    """Numerical thresholds used by the primary task-feasibility gate."""

    stop_error_limit_m: float = STRICT_STOP_ERROR_LIMIT_M
    time_error_limit_s: float = STRICT_TIME_ERROR_LIMIT_S
    safety_margin_eps_mps: float = DEFAULT_SAFETY_MARGIN_EPS_MPS

    def __post_init__(self) -> None:
        if self.stop_error_limit_m < 0.0:
            raise ValueError("stop_error_limit_m must be non-negative")
        if self.time_error_limit_s < 0.0:
            raise ValueError("time_error_limit_s must be non-negative")
        if self.safety_margin_eps_mps < 0.0:
            raise ValueError("safety_margin_eps_mps must be non-negative")


@dataclass(frozen=True)
class ConstraintAssessment:
    """Explicit pass/fail result for one evaluated trajectory."""

    success: bool
    precise_arrival: bool
    punctual_arrival: bool
    safe: bool
    feasible: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "precise_arrival": self.precise_arrival,
            "punctual_arrival": self.punctual_arrival,
            "safe": self.safe,
            "feasible": self.feasible,
            "failure_reasons": list(self.failure_reasons),
        }


def _as_bool(metrics: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = metrics.get(key, default)
    return bool(value)


def _as_float(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        result = float(value)
    except TypeError, ValueError:
        return float(default)
    return result if np.isfinite(result) else float(default)


def _require_bool(metrics: Mapping[str, Any], key: str) -> bool:
    value = metrics.get(key)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"metrics field {key!r} must be a boolean")
    return bool(value)


def _require_float(metrics: Mapping[str, Any], key: str) -> float:
    value = _as_float(metrics, key, np.nan)
    if not np.isfinite(value):
        raise ValueError(f"metrics field {key!r} must be a finite number")
    return value


def _require_non_negative_int(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"metrics field {key!r} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"metrics field {key!r} must be non-negative")
    return result


def assess_constraints(
    metrics: Mapping[str, Any],
    *,
    thresholds: ConstraintThresholds | None = None,
) -> ConstraintAssessment:
    """Evaluate stopping, punctuality, and safety constraints."""

    canonical_assessment = thresholds is None
    limits = thresholds or ConstraintThresholds(
        stop_error_limit_m=_require_float(metrics, "strict_stop_error_limit_m"),
        time_error_limit_s=_require_float(metrics, "strict_time_error_limit_s"),
    )
    raw_reason = metrics.get("termination_reason")
    termination_reason = str(raw_reason) if raw_reason is not None else None
    success = is_successful_evaluation(
        termination_reason=termination_reason,
    )
    stop_error = abs(_as_float(metrics, "stop_error_m", np.inf))
    time_error = abs(_as_float(metrics, "time_error_s", np.inf))
    precise_arrival = is_precise_evaluation(
        success=success,
        stop_error_m=stop_error,
        stop_error_limit_m=limits.stop_error_limit_m,
    )
    punctual_arrival = is_punctual_evaluation(
        precise_arrival=precise_arrival,
        time_error_s=time_error,
        time_error_limit_s=limits.time_error_limit_s,
    )

    min_margin = _require_float(metrics, "min_safety_margin_mps")
    violation_count = _require_non_negative_int(metrics, "safety_violation_count")
    safe = is_safe_evaluation(
        min_safety_margin_mps=min_margin,
        safety_violation_count=violation_count,
        safety_margin_eps_mps=limits.safety_margin_eps_mps,
    )
    feasible = is_feasible_evaluation(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        safe=safe,
    )
    if canonical_assessment:
        if _require_bool(metrics, "success") != success:
            raise ValueError(
                "metrics success field disagrees with canonical assessment"
            )
        if _require_bool(metrics, "precise_arrival") != precise_arrival:
            raise ValueError(
                "metrics precise_arrival field disagrees with canonical assessment"
            )
        if _require_bool(metrics, "punctual_arrival") != punctual_arrival:
            raise ValueError(
                "metrics punctual_arrival field disagrees with canonical assessment"
            )
        if _require_bool(metrics, "safe") != safe:
            raise ValueError("metrics safe field disagrees with canonical assessment")
        if _require_bool(metrics, "feasible") != feasible:
            raise ValueError(
                "metrics feasible field disagrees with canonical assessment"
            )

    failures: list[str] = []
    if not success:
        failures.append(
            termination_reason.lower()
            if termination_reason
            else "not_successful_arrival"
        )
    if success and not precise_arrival:
        failures.append("stop_error")
    if precise_arrival and not punctual_arrival:
        failures.append("time_error")
    if not safe:
        failures.append("safety_violation")

    return ConstraintAssessment(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        safe=safe,
        feasible=feasible,
        failure_reasons=tuple(failures),
    )
