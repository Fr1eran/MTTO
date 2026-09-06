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
    }
    values.update(overrides)
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
