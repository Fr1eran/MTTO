from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.completion_critic import (
    CompletionBuffer,
    CompletionCritic,
    CompletionCriticTrainer,
    CompletionDSPDLCallback,
    CompletionDSPDLConfig,
    CompletionTrajectoryAccumulator,
)
from rl.context_pool import Context, ContextPool
from rl.operational_state import OperationalState


def _spaces() -> tuple[gym.spaces.Box, gym.spaces.Box]:
    return (
        gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
    )


def _critic() -> CompletionCritic:
    observation_space, action_space = _spaces()
    return CompletionCritic(
        observation_space,
        action_space,
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs={},
        normalize_images=True,
        net_arch=[8, 8],
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs={"eps": 1e-5},
        learning_rate=3e-3,
        device="cpu",
    )


def _context_pool() -> ContextPool:
    distances = [6000.0, 3000.0, 1000.0, 300.0]
    return ContextPool(
        tuple(
            Context(
                index,
                distance,
                cast(
                    OperationalState,
                    cast(object, SimpleNamespace(position_m=distance)),
                ),
            )
            for index, distance in enumerate(distances)
        )
    )


def test_completion_accumulator_labels_all_decision_states_and_spans_drains() -> None:
    accumulator = CompletionTrajectoryAccumulator(observation_shape=(2,))
    accumulator.begin_episode(np.asarray([0.0, 1.0], dtype=np.float32))
    accumulator.record_transition(np.asarray([1.0, 2.0], dtype=np.float32), done=False)
    empty = accumulator.drain()
    assert cast(np.ndarray, empty["observations"]).shape == (0, 2)

    accumulator.record_transition(
        np.asarray([9.0, 9.0], dtype=np.float32),
        done=True,
        completion=0.75,
    )
    payload = accumulator.drain()
    np.testing.assert_allclose(payload["observations"], [[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(payload["completion_targets"], [0.75, 0.75])
    np.testing.assert_allclose(payload["episode_completions"], [0.75])


def test_completion_accumulator_keeps_active_episode_across_distribution_update() -> (
    None
):
    accumulator = CompletionTrajectoryAccumulator(observation_shape=(1,))
    accumulator.begin_episode(np.asarray([0.0], dtype=np.float32))
    accumulator.switch_version(1)
    accumulator.record_transition(
        np.asarray([1.0], dtype=np.float32), done=True, completion=0.5
    )

    payload = accumulator.drain()
    np.testing.assert_allclose(payload["completion_targets"], [0.5])
    assert accumulator.accepted_version == 1


def test_completion_accumulator_treats_manual_reset_as_zero_completion() -> None:
    accumulator = CompletionTrajectoryAccumulator(observation_shape=(1,))
    accumulator.begin_episode(np.asarray([0.0], dtype=np.float32))
    accumulator.begin_episode(np.asarray([1.0], dtype=np.float32))

    payload = accumulator.drain()
    np.testing.assert_allclose(payload["observations"], [[0.0]])
    np.testing.assert_allclose(payload["completion_targets"], [0.0])
    np.testing.assert_allclose(payload["episode_completions"], [0.0])


def test_completion_buffer_grows_and_returns_each_state_once_per_epoch() -> None:
    observation_space, action_space = _spaces()
    buffer = CompletionBuffer(2, observation_space, action_space, device="cpu")
    observations = np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0
    targets = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    buffer.add_batch(observations, targets)

    batches = list(buffer.get(3, rng=np.random.default_rng(7)))
    assert buffer.size() == 4
    assert sum(batch.observations.shape[0] for batch in batches) == 4
    observed_targets = (
        th.cat([batch.completion_targets for batch in batches]).numpy().reshape(-1)
    )
    assert sorted(observed_targets.tolist()) == pytest.approx(sorted(targets.tolist()))


def test_completion_critic_has_bounded_output_and_trainer_reduces_loss() -> None:
    critic = _critic()
    observation_space, action_space = _spaces()
    buffer = CompletionBuffer(8, observation_space, action_space, device="cpu")
    observations = np.asarray(
        [[-1.0, -1.0, -1.0], [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    targets = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    buffer.add_batch(observations, targets)
    with th.inference_mode():
        before_predictions = critic(th.as_tensor(observations))
        before = float(
            th.nn.functional.mse_loss(
                before_predictions, th.as_tensor(targets).reshape(-1, 1)
            )
        )
    assert th.all(before_predictions >= 0.0)
    assert th.all(before_predictions <= 1.0)

    trainer = CompletionCriticTrainer(
        critic,
        lr_schedule=lambda _: 3e-3,
        batch_size=4,
        n_epochs=30,
        loss_coef=0.5,
        max_grad_norm=0.5,
        rng=np.random.default_rng(3),
    )
    metrics = trainer.train(buffer, progress_remaining=1.0)
    assert metrics is not None
    with th.inference_mode():
        after = float(
            th.nn.functional.mse_loss(
                critic(th.as_tensor(observations)),
                th.as_tensor(targets).reshape(-1, 1),
            )
        )
    assert after < before
    assert buffer.size() == 0
    assert metrics.learning_rate == pytest.approx(3e-3)


def test_completion_callback_builds_an_independent_ppo_style_value_branch() -> None:
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=4,
        n_epochs=3,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [16], "vf": [12, 8]}},
    )
    callback = CompletionDSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.ones((4, 4), dtype=np.float32),
        config=CompletionDSPDLConfig(),
        seed=11,
    )
    policy_parameters = [
        parameter.detach().clone() for parameter in model.policy.parameters()
    ]
    callback.init_callback(model)
    callback._on_training_start()

    assert callback._critic is not None
    assert callback._trainer is not None
    assert callback._critic.mlp_extractor.latent_dim_vf == 8
    assert callback._trainer.batch_size == model.batch_size
    assert callback._trainer.n_epochs == model.n_epochs
    assert callback._trainer.loss_coef == pytest.approx(model.vf_coef)
    for before, after in zip(policy_parameters, model.policy.parameters(), strict=True):
        assert th.equal(before, after)
    env.close()


def test_completion_callback_alpha_uses_completion_ema_and_is_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = CompletionDSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.ones((4, 2), dtype=np.float32),
        config=CompletionDSPDLConfig(),
    )
    callback.model = cast(
        object,
        SimpleNamespace(
            device=th.device("cpu"),
            logger=None,
            get_env=lambda: SimpleNamespace(),
        ),
    )
    callback._critic = cast(
        CompletionCritic,
        lambda observations: th.full((observations.shape[0], 1), 0.5),
    )
    callback._completion_ema = 0.8
    captured: list[float] = []

    def solve(**kwargs: object) -> np.ndarray:
        captured.append(float(kwargs["alpha"]))
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    target_kl = callback._solver.kl_divergence(
        callback.initial_context_distribution(),
        callback.target_context_distribution,
    )
    expected = np.clip(
        callback._config.zeta
        * callback._completion_ema
        / max(target_kl, callback._config.target_kl_stop),
        callback._config.alpha_min,
        callback._config.alpha_max,
    )
    callback._maybe_update_curriculum()

    assert captured == pytest.approx([expected])


class _CompletionEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.accumulator = CompletionTrajectoryAccumulator(observation_shape=(2,))
        self.steps = 0
        self.distribution_updates: list[tuple[np.ndarray, int]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self.steps = 0
        observation = np.asarray([0.0, 0.0], dtype=np.float32)
        self.accumulator.begin_episode(observation)
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.steps += 1
        done = self.steps >= 2
        observation = np.asarray([self.steps / 2.0, 0.5], dtype=np.float32)
        self.accumulator.record_transition(
            observation,
            done=done,
            completion=1.0 if done else None,
        )
        return observation, 0.0, done, False, {}

    def drain_completion_trajectories(self) -> dict[str, object]:
        return self.accumulator.drain()

    def set_dspdl_distribution(self, distribution: np.ndarray, *, version: int) -> None:
        self.accumulator.switch_version(version)
        self.distribution_updates.append((np.asarray(distribution), version))

    def disable_dspdl_accumulator(self) -> None:
        self.accumulator.disable()


def test_completion_callback_runs_through_ppo_rollout_lifecycle() -> None:
    raw_env = _CompletionEnv()
    env = DummyVecEnv([lambda: raw_env])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
    )
    callback = CompletionDSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.asarray(
            [[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        config=CompletionDSPDLConfig(),
        seed=5,
    )

    model.learn(total_timesteps=12, callback=callback)

    assert callback.completion_ema == pytest.approx(1.0)
    assert raw_env.distribution_updates
    assert raw_env.distribution_updates[0][1] == 1
    env.close()
