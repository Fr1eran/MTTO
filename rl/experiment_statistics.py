"""Shared task-constraint checks for the final publication experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

STRICT_STOP_ERROR_LIMIT_M = 0.3
STRICT_TIME_ERROR_LIMIT_S = 10.0
DEFAULT_SAFETY_MARGIN_EPS_MPS = 1e-6


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
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def assess_constraints(
    metrics: Mapping[str, Any],
    *,
    thresholds: ConstraintThresholds | None = None,
) -> ConstraintAssessment:
    """Evaluate stopping, punctuality, and safety constraints."""

    limits = thresholds or ConstraintThresholds()
    success = (
        _as_bool(metrics, "success")
        and _as_bool(metrics, "terminated", default=True)
        and not _as_bool(metrics, "truncated", default=False)
    )
    stop_error = abs(_as_float(metrics, "stop_error_m", np.inf))
    time_error = abs(_as_float(metrics, "time_error_s", np.inf))
    precise_arrival = success and stop_error <= limits.stop_error_limit_m
    punctual_arrival = precise_arrival and time_error < limits.time_error_limit_s

    min_margin = _as_float(metrics, "min_safety_margin_mps", 0.0)
    violation_count = int(max(0.0, _as_float(metrics, "safety_violation_count", 0.0)))
    safe = min_margin >= -limits.safety_margin_eps_mps and violation_count == 0

    failures: list[str] = []
    if not success:
        failures.append(
            "truncated"
            if _as_bool(metrics, "truncated", default=False)
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
        feasible=not failures,
        failure_reasons=tuple(failures),
    )
