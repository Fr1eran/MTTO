"""Versioned sampling from an immutable DSPDL context pool."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rl.context_pool import Context, ContextPool

__all__ = ["ContextSampler", "CurriculumDistributionState"]


class CurriculumDistributionState:
    """Own one immutable, versioned curriculum distribution."""

    def __init__(
        self,
        *,
        context_count: int,
        initial_distribution: NDArray[np.floating] | list[float],
    ) -> None:
        if context_count <= 0:
            raise ValueError("context_count must be positive")
        self._context_count = int(context_count)
        self._distribution = self._validate_distribution(initial_distribution)
        self._version = 0

    @property
    def context_count(self) -> int:
        return self._context_count

    @property
    def version(self) -> int:
        return self._version

    @property
    def distribution(self) -> NDArray[np.float64]:
        return self._distribution.copy()

    def update(
        self,
        distribution: NDArray[np.floating] | list[float],
        *,
        version: int,
    ) -> None:
        if not isinstance(version, (int, np.integer)):
            raise TypeError("sampling distribution version must be an integer")
        new_version = int(version)
        if new_version <= self._version:
            raise ValueError("sampling distribution version must increase")
        validated = self._validate_distribution(distribution)
        self._distribution = validated
        self._version = new_version

    def _validate_distribution(
        self, distribution: NDArray[np.floating] | list[float]
    ) -> NDArray[np.float64]:
        values = np.asarray(distribution, dtype=np.float64)
        if values.ndim != 1 or values.size != self._context_count:
            raise ValueError(
                "distribution must have one finite non-negative value per context"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("distribution must be finite and non-negative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("distribution must sum to a positive value")
        normalized = np.asarray(values / total, dtype=np.float64)
        normalized.flags.writeable = False
        return normalized


class ContextSampler:
    """Own one environment's versioned DSPDL context distribution."""

    def __init__(
        self,
        *,
        context_pool: ContextPool,
        initial_distribution: NDArray[np.floating] | list[float] | None = None,
        distribution_state: CurriculumDistributionState | None = None,
        seed: int | None = None,
    ) -> None:
        self.context_pool = context_pool
        self._rng: np.random.Generator = np.random.default_rng(seed)
        if (initial_distribution is None) == (distribution_state is None):
            raise ValueError(
                "exactly one of initial_distribution and distribution_state is required"
            )
        if distribution_state is None:
            assert initial_distribution is not None
            distribution_state = CurriculumDistributionState(
                context_count=context_pool.context_count,
                initial_distribution=initial_distribution,
            )
        elif distribution_state.context_count != context_pool.context_count:
            raise ValueError("distribution state and context pool sizes differ")
        self._distribution_state = distribution_state

    @property
    def distribution_state(self) -> CurriculumDistributionState:
        return self._distribution_state

    @property
    def version(self) -> int:
        return self._distribution_state.version

    @property
    def context_count(self) -> int:
        return self.context_pool.context_count

    @property
    def distribution(self) -> NDArray[np.float64]:
        return self._distribution_state.distribution

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def update_distribution(
        self,
        distribution: NDArray[np.floating] | list[float],
        *,
        version: int,
    ) -> None:
        self._distribution_state.update(distribution, version=version)

    def sample(self) -> Context:
        index = int(
            self._rng.choice(
                self.context_pool.context_count,
                p=self._distribution_state._distribution,
            )
        )
        return self.context_pool.context_at(index)
