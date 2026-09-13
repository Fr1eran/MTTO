import pytest

from rl.experiment_statistics import ConstraintThresholds, assess_constraints


def _metrics(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "success": True,
        "terminated": True,
        "truncated": False,
        "stop_error_m": 0.1,
        "time_error_s": 2.0,
        "min_safety_margin_mps": 0.2,
        "safety_violation_count": 0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 10.0,
    }
    values.update(overrides)
    success = bool(
        values["success"] and values["terminated"] and not values["truncated"]
    )
    precise = success and abs(float(values["stop_error_m"])) <= float(
        values["strict_stop_error_limit_m"]
    )
    punctual = precise and abs(float(values["time_error_s"])) < float(
        values["strict_time_error_limit_s"]
    )
    safe = (
        float(values["min_safety_margin_mps"]) >= -1e-6
        and int(values["safety_violation_count"]) == 0
    )
    values["success"] = success
    values["precise_arrival"] = precise
    values["punctual_arrival"] = punctual
    values["safe"] = safe
    values["feasible"] = success and precise and punctual and safe
    return values


def test_assess_constraints_is_feasibility_first() -> None:
    result = assess_constraints(_metrics())
    assert result.feasible
    assert result.failure_reasons == ()

    late = assess_constraints(_metrics(time_error_s=-10.0))
    assert not late.feasible
    assert late.failure_reasons == ("time_error",)

    truncated = assess_constraints(_metrics(truncated=True))
    assert not truncated.feasible
    assert "truncated" in truncated.failure_reasons


def test_custom_thresholds_are_applied() -> None:
    thresholds = ConstraintThresholds(
        stop_error_limit_m=0.5,
        time_error_limit_s=12.0,
        safety_margin_eps_mps=1e-4,
    )
    result = assess_constraints(
        _metrics(stop_error_m=-0.4, time_error_s=-11.0),
        thresholds=thresholds,
    )
    assert result.feasible


@pytest.mark.parametrize(
    ("margin_mps", "violation_count", "expected_safe"),
    (
        (-0.5e-6, 0, True),
        (-0.5e-6, 1, False),
        (-2.0e-6, 0, False),
    ),
)
def test_assess_constraints_uses_canonical_safety_boundary(
    margin_mps: float,
    violation_count: int,
    expected_safe: bool,
) -> None:
    result = assess_constraints(
        _metrics(
            min_safety_margin_mps=margin_mps,
            safety_violation_count=violation_count,
        )
    )

    assert result.safe is expected_safe


def test_assess_constraints_uses_inclusive_stop_and_exclusive_time_limits() -> None:
    assert assess_constraints(_metrics(stop_error_m=0.3)).precise_arrival
    assert not assess_constraints(_metrics(time_error_s=10.0)).punctual_arrival


def test_assess_constraints_requires_v2_safety_fields() -> None:
    metrics = _metrics()
    del metrics["safety_violation_count"]

    with pytest.raises(ValueError, match="safety_violation_count"):
        assess_constraints(metrics)
