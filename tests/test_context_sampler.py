from dataclasses import replace

import numpy as np
import pytest

from model.ocs.stopping_points_stepping import SPSState
from rl.context_pool import Context, ContextPool
from rl.context_sampler import ContextSampler, CurriculumDistributionState
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


def test_context_samplers_share_distribution_but_keep_independent_rngs() -> None:
    pool = _pool()
    state = CurriculumDistributionState(
        context_count=pool.context_count,
        initial_distribution=[1.0, 0.0, 0.0],
    )
    first = ContextSampler(context_pool=pool, distribution_state=state, seed=1)
    second = ContextSampler(context_pool=pool, distribution_state=state, seed=2)

    state.update([0.0, 0.0, 1.0], version=1)

    assert first.distribution_state is second.distribution_state is state
    assert first.version == second.version == 1
    assert first.sample().context_index == second.sample().context_index == 2


def test_distribution_state_is_read_only_and_invalid_update_is_atomic() -> None:
    state = CurriculumDistributionState(
        context_count=3,
        initial_distribution=[2.0, 1.0, 1.0],
    )
    external = state.distribution
    external[0] = 0.0

    assert state.distribution == pytest.approx([0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        state.update([0.0, -1.0, 2.0], version=1)
    assert state.version == 0
    assert state.distribution == pytest.approx([0.5, 0.25, 0.25])


def test_context_sampler_requires_exactly_one_distribution_source() -> None:
    pool = _pool()
    state = CurriculumDistributionState(
        context_count=pool.context_count,
        initial_distribution=[1.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="exactly one"):
        _ = ContextSampler(context_pool=pool, seed=1)
    with pytest.raises(ValueError, match="exactly one"):
        _ = ContextSampler(
            context_pool=pool,
            initial_distribution=[1.0, 0.0, 0.0],
            distribution_state=state,
            seed=1,
        )


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
