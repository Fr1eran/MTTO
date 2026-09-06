import numpy as np
import pytest

from rl.dspdl_distribution import DSPDLDistributionSolver


def test_distribution_solver_is_feasible_and_stops_on_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = DSPDLDistributionSolver(relative_entropy_bound=0.02)
    current = np.asarray([0.3, 0.3, 0.2, 0.2], dtype=np.float64)
    target = np.asarray([0.95, 0.02, 0.02, 0.01], dtype=np.float64)
    original = solver._distribution_at_dual
    calls = 0

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_distribution_at_dual", counted)
    candidate = solver.solve(
        context_values=np.asarray([8.0, 1.0, 0.0, -1.0]),
        current_distribution=current,
        target_distribution=target,
        alpha=0.5,
    )
    assert candidate.sum() == pytest.approx(1.0)
    assert solver.kl_divergence(candidate, current) <= 0.02 + solver.tolerance
    assert calls < solver.max_iterations


def test_equal_warmup_values_skip_dual_search(monkeypatch: pytest.MonkeyPatch) -> None:
    solver = DSPDLDistributionSolver(relative_entropy_bound=0.02)
    current = np.asarray([0.6, 0.4], dtype=np.float64)
    monkeypatch.setattr(
        solver,
        "_distribution_at_dual",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dual search should be skipped")
        ),
    )
    candidate = solver.solve(
        context_values=np.ones(2),
        current_distribution=current,
        target_distribution=np.asarray([0.9, 0.1]),
        alpha=0.0,
    )
    assert candidate == pytest.approx(current)
