from types import SimpleNamespace

import numpy as np
import pytest

from rl.reference_initial_state_provider import ReferenceInitialStateProvider


class _Sampler:
    eligible_node_count = 3

    def sample(self, rng, *, weights):
        index = int(rng.choice(np.arange(3), p=np.asarray(weights)[:3]))
        return SimpleNamespace(reference_index=index)


def test_provider_samples_only_eligible_nodes_and_is_reproducible() -> None:
    first = ReferenceInitialStateProvider(
        sampler=_Sampler(),  # type: ignore[arg-type]
        initial_weights=[0.2, 0.3, 0.5],
        seed=17,
    )
    second = ReferenceInitialStateProvider(
        sampler=_Sampler(),  # type: ignore[arg-type]
        initial_weights=[0.2, 0.3, 0.5],
        seed=17,
    )

    assert [first.sample().reference_index for _ in range(10)] == [
        second.sample().reference_index for _ in range(10)
    ]


def test_provider_updates_versioned_distribution() -> None:
    provider = ReferenceInitialStateProvider(
        sampler=_Sampler(),  # type: ignore[arg-type]
        initial_weights=[1.0, 0.0, 0.0],
        seed=1,
    )
    provider.set_sampling_distribution([0.0, 1.0, 0.0], version=1)

    assert provider.version == 1
    assert provider.sample().reference_index == 1
    with pytest.raises(ValueError, match="must increase"):
        provider.set_sampling_distribution([1.0, 0.0, 0.0], version=0)
    with pytest.raises(ValueError, match="must increase"):
        provider.set_sampling_distribution([1.0, 0.0, 0.0], version=1)


def test_provider_samples_include_monotonic_token_and_distribution_version() -> None:
    provider = ReferenceInitialStateProvider(
        sampler=_Sampler(),  # type: ignore[arg-type]
        initial_weights=[1.0, 0.0, 0.0],
        seed=1,
    )
    first = provider.sample()
    provider.set_sampling_distribution([0.0, 1.0, 0.0], version=1)
    second = provider.sample()

    assert (first.sample_id, first.reference_index, first.distribution_version) == (
        0,
        0,
        0,
    )
    assert (second.sample_id, second.reference_index, second.distribution_version) == (
        1,
        1,
        1,
    )


@pytest.mark.parametrize(
    "weights",
    ([1.0, 0.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]),
)
def test_provider_rejects_invalid_weights(weights) -> None:
    with pytest.raises(ValueError):
        ReferenceInitialStateProvider(
            sampler=_Sampler(),  # type: ignore[arg-type]
            initial_weights=weights,
            seed=1,
        )
