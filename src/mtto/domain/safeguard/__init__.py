from __future__ import annotations

from mtto.domain.safeguard.curves import (
    CalculationInputs,
    Safeguard,
    SafeguardCurveConfig,
    SafeGuardCurves,
    SafeguardParams,
    build_safeguard,
    calculate_curves,
)
from mtto.domain.safeguard.dynamic_limits import (
    SPS,
    SafeguardViolation,
    SPSState,
    ViolationKind,
    current_stopping_point,
    dynamic_limit_violation,
    dynamic_limits,
    latest_intervention_points,
    max_speed,
    min_speed,
)
from mtto.domain.safeguard.static_region import (
    StaticRegion,
    detect_any_danger,
    detect_danger,
    intersecting_danger_points,
)

__all__ = [
    "SafeguardParams",
    "Safeguard",
    "build_safeguard",
    "calculate_curves",
    "SafeguardCurveConfig",
    "CalculationInputs",
    "SafeGuardCurves",
    "SPS",
    "SPSState",
    "dynamic_limits",
    "min_speed",
    "max_speed",
    "current_stopping_point",
    "latest_intervention_points",
    "ViolationKind",
    "SafeguardViolation",
    "dynamic_limit_violation",
    "StaticRegion",
    "detect_danger",
    "detect_any_danger",
    "intersecting_danger_points",
]
