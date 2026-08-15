"""Versioned sampling from an immutable DSPDL context pool."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rl.context_pool import Context, ContextPool

__all__ = ["ContextSampler"]


class ContextSampler:
    """Own one environment's versioned DSPDL context distribution."""

    def __init__(
        self,
        *,
        context_pool: ContextPool,
        initial_distribution: NDArray[np.floating] | list[float],
        seed: int | None = None,
    ) -> None:
        self.context_pool = context_pool
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._distribution: NDArray[np.float64] = self._validate_distribution(
            initial_distribution
        )
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def context_count(self) -> int:
        return self.context_pool.context_count

    @property
    def distribution(self) -> NDArray[np.float64]:
        return self._distribution.copy()

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def update_distribution(
        self,
        distribution: NDArray[np.floating] | list[float],
        *,
        version: int,
    ) -> None:
        if not isinstance(version, (int, np.integer)):
            raise TypeError("version must be an integer")
        new_version = int(version)
        if new_version <= self._version:
            raise ValueError("sampling distribution version must increase")
        validated = self._validate_distribution(distribution)
        self._distribution = validated
        self._version = new_version

    def sample(self) -> Context:
        index = int(
            self._rng.choice(self.context_pool.context_count, p=self._distribution)
        )
        return self.context_pool.context_at(index)

    def _validate_distribution(
        self, distribution: NDArray[np.floating] | list[float]
    ) -> NDArray[np.float64]:
        values = np.asarray(distribution, dtype=np.float64)
        if values.ndim != 1 or values.size != self.context_pool.context_count:
            raise ValueError(
                "distribution must have one finite non-negative value per context"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("distribution must be finite and non-negative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("distribution must sum to a positive value")
        return values / total
