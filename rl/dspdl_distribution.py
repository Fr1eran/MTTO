"""Generic KL-constrained distribution updates for DSPDL variants."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["DSPDLDistributionSolver"]


class DSPDLDistributionSolver:
    """Solve a finite DSPDL KL-constrained distribution update.

    The solver is shared by curriculum variants.  It contains no assumption
    about how context values are produced; each evaluator variant supplies the
    values used by the distribution update.
    """

    def __init__(
        self,
        *,
        relative_entropy_bound: float,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
    ) -> None:
        if relative_entropy_bound <= 0.0:
            raise ValueError("relative_entropy_bound must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.relative_entropy_bound = float(relative_entropy_bound)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def solve(
        self,
        *,
        context_values: NDArray[np.float64],
        current_distribution: NDArray[np.float64],
        target_distribution: NDArray[np.float64],
        alpha: float,
    ) -> NDArray[np.float64]:
        values = np.asarray(context_values, dtype=np.float64)
        current = np.asarray(current_distribution, dtype=np.float64)
        target = np.asarray(target_distribution, dtype=np.float64)
        if values.shape != current.shape or target.shape != current.shape:
            raise ValueError("DSPDL solver inputs must have matching shapes")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if alpha == 0.0 and np.allclose(
            values, values[0], rtol=0.0, atol=self.tolerance
        ):
            return current.copy()

        log_target = np.log(target)
        log_current = np.log(current)
        if alpha > 0.0:
            unconstrained = self._distribution_at_dual(
                values, alpha, 0.0, log_target, log_current
            )
            if (
                self.kl_divergence(unconstrained, current)
                <= self.relative_entropy_bound + self.tolerance
            ):
                return unconstrained

        lower = 0.0
        upper = 1.0
        while (
            self.kl_divergence(
                self._distribution_at_dual(
                    values, alpha, upper, log_target, log_current
                ),
                current,
            )
            > self.relative_entropy_bound + self.tolerance
        ):
            upper *= 2.0
            if upper > 1e12:
                raise RuntimeError("could not satisfy the DSPDL relative-entropy bound")

        feasible = self._distribution_at_dual(
            values, alpha, upper, log_target, log_current
        )
        for _ in range(self.max_iterations):
            middle = (lower + upper) / 2.0
            candidate = self._distribution_at_dual(
                values, alpha, middle, log_target, log_current
            )
            candidate_kl = self.kl_divergence(candidate, current)
            if abs(candidate_kl - self.relative_entropy_bound) <= self.tolerance:
                return candidate
            if candidate_kl > self.relative_entropy_bound:
                lower = middle
            else:
                upper = middle
                feasible = candidate
            if upper - lower <= np.finfo(np.float64).eps * max(1.0, upper):
                break
        return feasible

    @staticmethod
    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        smallest = np.finfo(np.float64).tiny
        safe_left = np.maximum(left, smallest)
        safe_right = np.maximum(right, smallest)
        return float(np.sum(safe_left * (np.log(safe_left) - np.log(safe_right))))

    @staticmethod
    def _distribution_at_dual(
        values: np.ndarray,
        alpha: float,
        dual: float,
        log_target: np.ndarray,
        log_current: np.ndarray,
    ) -> NDArray[np.float64]:
        denominator = alpha + dual
        if denominator <= 0.0:
            raise ValueError("DSPDL dual denominator must be positive")
        logits = (
            values / denominator
            + alpha / denominator * log_target
            + dual / denominator * log_current
        )
        logits -= float(np.max(logits))
        distribution = np.maximum(np.exp(logits), np.finfo(np.float64).tiny).astype(
            np.float64
        )
        return distribution / float(np.sum(distribution))
