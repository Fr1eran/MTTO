from dataclasses import replace

import numpy as np
import pytest

from model.ocs.stopping_points_stepping import SPSState
from rl.context_pool import Context, ContextPool
from rl.context_sampler import ContextSampler
from rl.operational_state import OperationalState


def _state(index: int) -> OperationalState:
    return OperationalState(
        position_m=float(index),
        speed_mps=0.0,
        acceleration_mps2=0.0,
        operation_time_s=0.0,
        redundant_operation_time_s=0.0,
        energy_consumption_kj=0.0,
        slope_permille=0.0,
        min_speed_mps=0.0,
        max_speed_mps=10.0,
        stop_error_m=0.0,
        sps_state=SPSState(),
        step_count=index,
    )


def _pool() -> ContextPool:
    return ContextPool(tuple(Context(i, float(3 - i), _state(i)) for i in range(3)))


def test_context_sampler_is_seed_reproducible() -> None:
    first = ContextSampler(
        context_pool=_pool(), initial_distribution=[0.2, 0.3, 0.5], seed=17
    )
    second = ContextSampler(
        context_pool=_pool(), initial_distribution=[0.2, 0.3, 0.5], seed=17
    )

    assert [first.sample().context_index for _ in range(10)] == [
        second.sample().context_index for _ in range(10)
    ]


def test_context_sampler_updates_versioned_distribution() -> None:
    sampler = ContextSampler(
        context_pool=_pool(), initial_distribution=[1.0, 0.0, 0.0], seed=1
    )
    sampler.update_distribution([0.0, 1.0, 0.0], version=1)

    assert sampler.version == 1
    assert sampler.sample().context_index == 1
    assert sampler.distribution == pytest.approx([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="must increase"):
        sampler.update_distribution([1.0, 0.0, 0.0], version=1)


def test_invalid_update_does_not_change_distribution_or_version() -> None:
    sampler = ContextSampler(
        context_pool=_pool(), initial_distribution=[1.0, 0.0, 0.0], seed=1
    )

    with pytest.raises(ValueError):
        sampler.update_distribution([0.0, -1.0, 2.0], version=1)

    assert sampler.version == 0
    assert sampler.distribution == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "distribution",
    ([1.0, 0.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]),
)
def test_context_sampler_rejects_invalid_distribution(
    distribution: list[float],
) -> None:
    with pytest.raises(ValueError):
        _ = ContextSampler(
            context_pool=_pool(), initial_distribution=distribution, seed=1
        )


def test_context_pool_rejects_non_contiguous_indices() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _ = ContextPool((Context(1, 1.0, replace(_state(0), step_count=1)),))
