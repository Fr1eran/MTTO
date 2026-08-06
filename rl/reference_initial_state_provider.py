"""Versioned sampling of reconstructed reference initial states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl.reference_trajectory_sampler import (
    ReferenceTrajectorySampler,
    ReferenceTrajectoryState,
)

__all__ = ["ReferenceInitialStateProvider", "ReferenceInitialStateSample"]


@dataclass(frozen=True, slots=True)
class ReferenceInitialStateSample:
    """One provider draw, including the distribution that produced it."""

    trajectory_state: ReferenceTrajectoryState
    sample_id: int
    distribution_version: int

    @property
    def runtime_state(self):
        return self.trajectory_state.runtime_state

    @property
    def reference_index(self) -> int:
        return self.trajectory_state.reference_index


class ReferenceInitialStateProvider:
    """Own one environment's reference-state sampling distribution."""

    def __init__(
        self,
        *,
        sampler: ReferenceTrajectorySampler,
        initial_weights: NDArray[np.floating] | list[float],
        seed: int | None = None,
    ) -> None:
        self.sampler: ReferenceTrajectorySampler = sampler
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._weights: NDArray[np.float64] = self._validate_and_normalize_weights(
            initial_weights
        )
        self._version: int = 0
        self._next_sample_id: int = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def eligible_node_count(self) -> int:
        return self.sampler.eligible_node_count

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights.copy()

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def set_sampling_distribution(
        self,
        weights: NDArray[np.floating] | list[float],
        *,
        version: int,
    ) -> None:
        if not isinstance(version, (int, np.integer)):
            raise TypeError("version must be an integer")
        new_version = int(version)
        if new_version <= self._version:
            raise ValueError("sampling distribution version must increase")
        self._weights = self._validate_and_normalize_weights(weights)
        self._version = new_version

    def sample(self) -> ReferenceInitialStateSample:
        all_weights = np.concatenate(
            (self._weights, np.asarray([0.0], dtype=np.float64))
        )
        trajectory_state = self.sampler.sample(self._rng, weights=all_weights)
        result = ReferenceInitialStateSample(
            trajectory_state=trajectory_state,
            sample_id=self._next_sample_id,
            distribution_version=self._version,
        )
        self._next_sample_id += 1
        return result

    def _validate_and_normalize_weights(
        self, weights: NDArray[np.floating] | list[float]
    ) -> NDArray[np.float64]:
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or values.size != self.sampler.eligible_node_count:
            raise ValueError(
                "weights must have one finite non-negative value per eligible "
                + "reference node"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("weights must be finite and non-negative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return values / total
