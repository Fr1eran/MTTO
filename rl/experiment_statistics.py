"""Shared task-constraint checks for the final publication experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.evaluation import (
    SAFETY_MARGIN_EPS_MPS,
    EvaluationMetrics,
    is_feasible_evaluation,
    is_precise_evaluation,
    is_punctual_evaluation,
    is_safe_evaluation,
    is_successful_evaluation,
)

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
    metrics: EvaluationMetrics | Mapping[str, Any],
    *,
    thresholds: ConstraintThresholds | None = None,
) -> ConstraintAssessment:
    """Evaluate stopping, punctuality, and safety constraints."""

    canonical_assessment = thresholds is None
    limits = thresholds or ConstraintThresholds(
        stop_error_limit_m=_require_float(metrics, "strict_stop_error_limit_m"),
        time_error_limit_s=_require_float(metrics, "strict_time_error_limit_s"),
    )
    success = is_successful_evaluation(
        terminated=_as_bool(metrics, "terminated", default=True),
        truncated=_as_bool(metrics, "truncated", default=False),
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
        feasible=feasible,
        failure_reasons=tuple(failures),
    )
