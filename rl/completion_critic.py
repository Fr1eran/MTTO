"""Task-completion value fitting and DSPDL curriculum control."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, cast, override

import gymnasium as gym
import numpy as np
import torch as th
from numpy.typing import NDArray
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel, BasePolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.utils import explained_variance, update_learning_rate

from rl.context_pool import ContextPool
from rl.context_sampler import CurriculumDistributionState
from rl.dspdl import DSPDLDistributionSolver

__all__ = [
    "CompletionBuffer",
    "CompletionCritic",
    "CompletionCriticTrainer",
    "CompletionDSPDLCallback",
    "CompletionDSPDLConfig",
    "CompletionTrajectoryAccumulator",
    "completion_critic_metadata",
]


@dataclass(frozen=True)
class CompletionDSPDLConfig:
    """Configuration for task-completion based discrete SPDL."""

    initial_gaussian_std_m: float = 800.0
    initial_peak_remaining_distance_m: float = 3000.0
    initial_uniform_mass: float = 0.01
    target_uniform_mass: float = 0.05
    target_kl_stop: float = 0.1
    relative_entropy_bound: float = 0.02
    zeta: float = 1.0
    completion_floor: float = 0.1
    completion_ema_alpha: float = 0.1
    alpha_min: float = 0.01
    alpha_max: float = 0.05
    update_interval_rollouts: int = 2
    success_base: float = 0.6
    stopping_weight: float = 0.25
    punctuality_weight: float = 0.15


class CompletionTrajectoryAccumulator:
    """Collect completed decision-state trajectories inside one environment."""

    def __init__(
        self,
        *,
        observation_shape: tuple[int, ...],
        success_base: float = 0.6,
        stopping_weight: float = 0.25,
        punctuality_weight: float = 0.15,
    ) -> None:
        if not observation_shape or any(int(size) <= 0 for size in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        self._observation_shape = tuple(int(size) for size in observation_shape)
        weights = (
            float(success_base),
            float(stopping_weight),
            float(punctuality_weight),
        )
        if min(weights) < 0.0 or not np.isclose(sum(weights), 1.0):
            raise ValueError(
                "task-completion weights must be non-negative and sum to one"
            )
        self._completion_weights = weights
        self._enabled = True
        self._accepted_version = 0
        self._active_observations: list[NDArray[np.float32]] = []
        self._completed_observations: list[NDArray[np.float32]] = []
        self._completed_targets: list[NDArray[np.float32]] = []
        self._episode_completions: list[float] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def accepted_version(self) -> int:
        return self._accepted_version

    @property
    def completion_weights(self) -> tuple[float, float, float]:
        return self._completion_weights

    def begin_episode(self, observation: NDArray[np.floating]) -> None:
        if not self._enabled:
            return
        if self._active_observations:
            # A manual Gym reset is an implicit truncation of the active task.
            self._finalize_episode(0.0)
        self._active_observations.append(self._validate_observation(observation))

    def record_transition(
        self,
        next_observation: NDArray[np.floating],
        *,
        done: bool,
        completion: float | None = None,
    ) -> None:
        if not self._enabled:
            return
        if not self._active_observations:
            raise RuntimeError("completion trajectory has not been started")
        if not done:
            if completion is not None:
                raise ValueError("completion is only valid for a finished episode")
            self._active_observations.append(
                self._validate_observation(next_observation)
            )
            return
        if completion is None or not np.isfinite(completion):
            raise ValueError("finished episode completion must be finite")
        target = float(completion)
        if not 0.0 <= target <= 1.0:
            raise ValueError("episode completion must be within [0, 1]")
        self._finalize_episode(target)

    def validate_version_update(self, version: int) -> int:
        if not isinstance(version, (int, np.integer)):
            raise TypeError("version must be an integer")
        new_version = int(version)
        if new_version <= self._accepted_version:
            raise ValueError("accumulator version must increase")
        return new_version

    def switch_version(self, version: int) -> None:
        self._accepted_version = self.validate_version_update(version)

    def drain(self) -> dict[str, object]:
        if not self._enabled:
            return self._empty_payload()
        if self._completed_observations:
            observations = np.concatenate(self._completed_observations, axis=0)
            targets = np.concatenate(self._completed_targets, axis=0)
        else:
            observations = np.empty((0, *self._observation_shape), dtype=np.float32)
            targets = np.empty(0, dtype=np.float32)
        payload: dict[str, object] = {
            "observations": observations,
            "completion_targets": targets,
            "episode_completions": np.asarray(
                self._episode_completions, dtype=np.float32
            ),
        }
        self._completed_observations.clear()
        self._completed_targets.clear()
        self._episode_completions.clear()
        return payload

    def disable(self) -> None:
        self._enabled = False
        self._active_observations.clear()
        self._completed_observations.clear()
        self._completed_targets.clear()
        self._episode_completions.clear()

    def _validate_observation(
        self, observation: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        result = np.asarray(observation, dtype=np.float32)
        if result.shape != self._observation_shape:
            raise ValueError("completion observation has an invalid shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("completion observation must be finite")
        return result.copy()

    def _finalize_episode(self, target: float) -> None:
        observations = np.stack(self._active_observations, axis=0)
        self._completed_observations.append(observations)
        self._completed_targets.append(
            np.full(observations.shape[0], target, dtype=np.float32)
        )
        self._episode_completions.append(float(target))
        self._active_observations.clear()

    def _empty_payload(self) -> dict[str, object]:
        return {
            "observations": np.empty((0, *self._observation_shape), dtype=np.float32),
            "completion_targets": np.empty(0, dtype=np.float32),
            "episode_completions": np.empty(0, dtype=np.float32),
        }


class CompletionSamples(NamedTuple):
    observations: th.Tensor
    completion_targets: th.Tensor


class CompletionBuffer(BaseBuffer):
    """Short-lived supervised buffer containing completed trajectory states."""

    observations: NDArray[np.float32]
    completion_targets: NDArray[np.float32]

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        *,
        device: th.device | str = "auto",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("completion buffer size must be positive")
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(
                "CompletionBuffer currently requires a Box observation space"
            )
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=1,
        )
        self.observations = np.empty(
            (self.buffer_size, *self.obs_shape), dtype=np.float32
        )
        self.completion_targets = np.empty(self.buffer_size, dtype=np.float32)

    def add_batch(
        self,
        observations: NDArray[np.floating],
        completion_targets: NDArray[np.floating],
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32)
        targets = np.asarray(completion_targets, dtype=np.float32).reshape(-1)
        if obs.ndim != len(self.obs_shape) + 1 or obs.shape[1:] != self.obs_shape:
            raise ValueError("completion observation batch has an invalid shape")
        if obs.shape[0] != targets.shape[0]:
            raise ValueError("completion observations and targets have different sizes")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(targets)):
            raise ValueError("completion training batch must be finite")
        if np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("completion targets must be within [0, 1]")
        required = self.pos + obs.shape[0]
        if required > self.buffer_size:
            self._grow(required)
        self.observations[self.pos : required] = obs
        self.completion_targets[self.pos : required] = targets
        self.pos = required

    def get(
        self, batch_size: int, *, rng: np.random.Generator
    ) -> Generator[CompletionSamples]:
        size = self.size()
        if size <= 0:
            return
        if batch_size <= 0:
            raise ValueError("completion batch size must be positive")
        indices = rng.permutation(size)
        for start in range(0, size, batch_size):
            yield self._get_samples(indices[start : start + batch_size])

    @override
    def _get_samples(
        self, batch_inds: NDArray[np.integer], env: Any = None
    ) -> CompletionSamples:
        del env
        return CompletionSamples(
            observations=self.to_torch(self.observations[batch_inds]),
            completion_targets=self.to_torch(
                self.completion_targets[batch_inds].reshape(-1, 1)
            ),
        )

    def add(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError("use add_batch() for CompletionBuffer")

    def _grow(self, required: int) -> None:
        new_size = max(required, self.buffer_size * 2)
        observations = np.empty((new_size, *self.obs_shape), dtype=np.float32)
        targets = np.empty(new_size, dtype=np.float32)
        observations[: self.pos] = self.observations[: self.pos]
        targets[: self.pos] = self.completion_targets[: self.pos]
        self.observations = observations
        self.completion_targets = targets
        self.buffer_size = new_size


class CompletionCritic(BaseModel):
    """PPO-style value branch with a sigmoid task-completion output."""

    optimizer: th.optim.Optimizer

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        *,
        features_extractor_class: type,
        features_extractor_kwargs: dict[str, Any],
        normalize_images: bool,
        net_arch: list[int],
        activation_fn: type[th.nn.Module],
        ortho_init: bool,
        optimizer_class: type[th.optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
        learning_rate: float,
        device: th.device | str,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch={"pi": [], "vf": list(net_arch)},
            activation_fn=activation_fn,
            device=device,
        )
        self.completion_net = th.nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if ortho_init:
            self.features_extractor.apply(
                partial(BasePolicy.init_weights, gain=np.sqrt(2))
            )
            self.mlp_extractor.apply(partial(BasePolicy.init_weights, gain=np.sqrt(2)))
            self.completion_net.apply(partial(BasePolicy.init_weights, gain=1.0))
        self.to(device)
        self.optimizer = optimizer_class(
            self.parameters(), lr=float(learning_rate), **optimizer_kwargs
        )

    @override
    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.extract_features(observations, self.features_extractor)
        latent = self.mlp_extractor.forward_critic(features)
        return th.sigmoid(self.completion_net(latent))


@dataclass(frozen=True)
class CompletionTrainingMetrics:
    loss: float
    explained_variance: float
    learning_rate: float


class CompletionCriticTrainer:
    """Train a CompletionCritic with PPO critic update conventions."""

    def __init__(
        self,
        critic: CompletionCritic,
        *,
        lr_schedule: Any,
        batch_size: int,
        n_epochs: int,
        loss_coef: float,
        max_grad_norm: float,
        rng: np.random.Generator,
    ) -> None:
        self.critic = critic
        self.lr_schedule = lr_schedule
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.loss_coef = float(loss_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.rng = rng
        if self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("completion batch size and epoch count must be positive")
        if self.loss_coef < 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("completion loss coefficient and grad norm are invalid")

    def train(
        self, buffer: CompletionBuffer, *, progress_remaining: float
    ) -> CompletionTrainingMetrics | None:
        if buffer.size() <= 0:
            return None
        learning_rate = float(self.lr_schedule(float(progress_remaining)))
        update_learning_rate(self.critic.optimizer, learning_rate)
        self.critic.set_training_mode(True)
        losses: list[float] = []
        for _ in range(self.n_epochs):
            for samples in buffer.get(self.batch_size, rng=self.rng):
                predictions = self.critic(samples.observations)
                value_loss = th.nn.functional.mse_loss(
                    predictions, samples.completion_targets
                )
                loss = self.loss_coef * value_loss
                self.critic.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic.optimizer.step()
                losses.append(float(value_loss.detach().cpu()))
        with th.inference_mode():
            observations = buffer.to_torch(buffer.observations[: buffer.size()])
            predictions = self.critic(observations).detach().cpu().numpy().reshape(-1)
        targets = buffer.completion_targets[: buffer.size()].astype(np.float32)
        variance = float(explained_variance(predictions, targets))
        if not np.isfinite(variance):
            variance = 0.0
        metrics = CompletionTrainingMetrics(
            loss=float(np.mean(losses)),
            explained_variance=variance,
            learning_rate=learning_rate,
        )
        buffer.reset()
        return metrics


def _value_net_arch(policy: ActorCriticPolicy) -> list[int]:
    net_arch = policy.net_arch
    if isinstance(net_arch, dict):
        return [int(width) for width in net_arch.get("vf", [])]
    return [int(width) for width in net_arch]


def completion_critic_metadata(model: Any) -> dict[str, object]:
    policy = cast(ActorCriticPolicy, model.policy)
    return {
        "net_arch": _value_net_arch(policy),
        "activation_fn": policy.activation_fn.__name__,
        "features_extractor_class": policy.features_extractor_class.__name__,
        "optimizer_class": policy.optimizer_class.__name__,
        "optimizer_kwargs": dict(policy.optimizer_kwargs),
        "ortho_init": bool(policy.ortho_init),
        "normalize_images": bool(policy.normalize_images),
        "learning_rate_schedule": "ppo.lr_schedule",
        "initial_learning_rate": float(model.lr_schedule(1.0)),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "loss_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "output_activation": "Sigmoid",
        "inherits_from_ppo": True,
        "persistent": False,
    }


class CompletionDSPDLCallback(BaseCallback):
    """Update DSPDL from a PPO-style task-completion value approximator."""

    def __init__(
        self,
        *,
        context_pool: ContextPool,
        context_observations: NDArray[np.float32],
        config: CompletionDSPDLConfig,
        seed: int | None = None,
        solver: DSPDLDistributionSolver | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        observations = np.asarray(context_observations, dtype=np.float32)
        if (
            observations.ndim != 2
            or observations.shape[0] != context_pool.context_count
        ):
            raise ValueError("context_observations must have one row per context")
        if not np.all(np.isfinite(observations)):
            raise ValueError("context_observations must be finite")
        self._context_pool = context_pool
        self._context_observations = observations
        self._config = config
        self._validate_config()
        self._solver = solver or DSPDLDistributionSolver(
            relative_entropy_bound=config.relative_entropy_bound
        )
        self._start_index = int(np.argmax(context_pool.remaining_distances_m))
        self._target_distribution = self._build_target_distribution()
        self._distribution_state = CurriculumDistributionState(
            context_count=context_pool.context_count,
            initial_distribution=self._build_initial_distribution(),
        )
        self._rollouts_since_update_attempt = 0
        self._converged = False
        self._completion_ema: float | None = None
        self._seed = 0 if seed is None else int(seed) + 1_000_003
        self._rng = np.random.default_rng(self._seed)
        self._critic: CompletionCritic | None = None
        self._trainer: CompletionCriticTrainer | None = None
        self._buffer: CompletionBuffer | None = None

    def initial_context_distribution(self) -> NDArray[np.float64]:
        return self._distribution_state.distribution

    @property
    def distribution_state(self) -> CurriculumDistributionState:
        return self._distribution_state

    @property
    def target_context_distribution(self) -> NDArray[np.float64]:
        return self._target_distribution.copy()

    @property
    def completion_ema(self) -> float | None:
        return self._completion_ema

    @override
    def _on_training_start(self) -> None:
        policy = cast(ActorCriticPolicy, self.model.policy)
        with th.random.fork_rng(devices=[]):
            th.manual_seed(self._seed)
            critic = CompletionCritic(
                self.model.observation_space,
                self.model.action_space,
                features_extractor_class=policy.features_extractor_class,
                features_extractor_kwargs=dict(policy.features_extractor_kwargs),
                normalize_images=bool(policy.normalize_images),
                net_arch=_value_net_arch(policy),
                activation_fn=policy.activation_fn,
                ortho_init=bool(policy.ortho_init),
                optimizer_class=policy.optimizer_class,
                optimizer_kwargs=dict(policy.optimizer_kwargs),
                learning_rate=float(self.model.lr_schedule(1.0)),
                device=self.model.device,
            )
        self._critic = critic
        self._trainer = CompletionCriticTrainer(
            critic,
            lr_schedule=self.model.lr_schedule,
            batch_size=int(self.model.batch_size),
            n_epochs=int(self.model.n_epochs),
            loss_coef=float(self.model.vf_coef),
            max_grad_norm=float(self.model.max_grad_norm),
            rng=self._rng,
        )
        initial_capacity = max(
            1, int(self.model.n_steps) * int(self.training_env.num_envs)
        )
        self._buffer = CompletionBuffer(
            initial_capacity,
            self.model.observation_space,
            self.model.action_space,
            device=self.model.device,
        )

    @override
    def _on_rollout_start(self) -> None:
        self._train_completion_critic()
        if (
            not self._converged
            and self._rollouts_since_update_attempt
            >= self._config.update_interval_rollouts
        ):
            self._rollouts_since_update_attempt = 0
            self._maybe_update_curriculum()

    @override
    def _on_rollout_end(self) -> None:
        if self._converged:
            return
        self._drain_completion_trajectories()
        self._rollouts_since_update_attempt += 1

    @override
    def _on_step(self) -> bool:
        return True

    @override
    def _on_training_end(self) -> None:
        self._context_observations = np.empty((0, 0), dtype=np.float32)
        if self._buffer is not None:
            self._buffer.reset()
        self._critic = None
        self._trainer = None
        self._buffer = None

    def _train_completion_critic(self) -> None:
        if self._trainer is None or self._buffer is None:
            return
        metrics = self._trainer.train(
            self._buffer,
            progress_remaining=float(self.model._current_progress_remaining),
        )
        if metrics is None:
            return
        self._record_scalar("completion/loss", metrics.loss)
        self._record_scalar("completion/explained_variance", metrics.explained_variance)
        self._record_scalar("completion/learning_rate", metrics.learning_rate)

    def _drain_completion_trajectories(self) -> None:
        if self._buffer is None:
            raise RuntimeError("completion buffer is not initialized")
        payloads = self.training_env.env_method("drain_completion_trajectories")
        episode_completions: list[float] = []
        for raw_payload in payloads:
            if not isinstance(raw_payload, dict):
                raise TypeError("completion trajectory payload must be a dictionary")
            observations = np.asarray(raw_payload["observations"], dtype=np.float32)
            targets = np.asarray(raw_payload["completion_targets"], dtype=np.float32)
            episodes = np.asarray(raw_payload["episode_completions"], dtype=np.float32)
            if observations.shape[0] != targets.size:
                raise ValueError("completion trajectory payload shapes differ")
            if observations.shape[0] > 0:
                self._buffer.add_batch(observations, targets)
            episode_completions.extend(float(value) for value in episodes)
        if episode_completions:
            batch_mean = float(np.mean(episode_completions))
            if self._completion_ema is None:
                self._completion_ema = batch_mean
            else:
                eta = self._config.completion_ema_alpha
                self._completion_ema = (
                    1.0 - eta
                ) * self._completion_ema + eta * batch_mean

    def _maybe_update_curriculum(self) -> None:
        if self._converged or self._critic is None:
            return
        target_kl = self._solver.kl_divergence(
            self._distribution_state.distribution, self._target_distribution
        )
        if target_kl <= self._config.target_kl_stop:
            self._mark_converged()
            return
        tensor = th.as_tensor(
            self._context_observations,
            dtype=th.float32,
            device=self.model.device,
        )
        with th.inference_mode():
            values = self._critic(tensor).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(values)):
            return
        completion = (
            self._config.completion_floor
            if self._completion_ema is None
            else max(self._config.completion_floor, self._completion_ema)
        )
        alpha = (
            self._config.zeta * completion / max(target_kl, self._config.target_kl_stop)
        )
        alpha = float(np.clip(alpha, self._config.alpha_min, self._config.alpha_max))
        self._record_scalar("dspdl/alpha", alpha)
        try:
            current_distribution = self._distribution_state.distribution
            candidate = self._solver.solve(
                context_values=values.astype(np.float64),
                current_distribution=current_distribution,
                target_distribution=self._target_distribution,
                alpha=alpha,
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            if self.verbose:
                print(f"Completion DSPDL update skipped: {exc}")
            return
        if (
            candidate.shape != current_distribution.shape
            or not np.all(np.isfinite(candidate))
            or np.any(candidate <= 0.0)
            or not np.isclose(float(np.sum(candidate)), 1.0)
        ):
            return
        reaches_target = (
            self._solver.kl_divergence(candidate, self._target_distribution)
            <= self._config.target_kl_stop
        )
        if not np.allclose(
            candidate, current_distribution, rtol=1e-10, atol=1e-12
        ):
            next_version = self._distribution_state.version + 1
            self.training_env.env_method(
                "validate_dspdl_version", next_version
            )
            self._distribution_state.update(candidate, version=next_version)
            self.training_env.env_method("commit_dspdl_version", next_version)
        if reaches_target:
            self._mark_converged()

    def _build_initial_distribution(self) -> NDArray[np.float64]:
        remaining = self._context_pool.remaining_distances_m
        gaussian = np.exp(
            -0.5
            * np.square(
                (remaining - self._config.initial_peak_remaining_distance_m)
                / self._config.initial_gaussian_std_m
            )
        )
        gaussian /= float(np.sum(gaussian))
        uniform = np.full_like(gaussian, 1.0 / gaussian.size)
        result = (
            1.0 - self._config.initial_uniform_mass
        ) * gaussian + self._config.initial_uniform_mass * uniform
        return result / float(np.sum(result))

    def _build_target_distribution(self) -> NDArray[np.float64]:
        start = np.zeros(self._context_pool.context_count, dtype=np.float64)
        start[self._start_index] = 1.0
        uniform = np.full_like(start, 1.0 / start.size)
        result = (
            1.0 - self._config.target_uniform_mass
        ) * start + self._config.target_uniform_mass * uniform
        return result / float(np.sum(result))

    def _mark_converged(self) -> None:
        if self._converged:
            return
        self._converged = True
        self.training_env.env_method("disable_dspdl_accumulator")

    def _record_scalar(self, key: str, value: float) -> None:
        logger = getattr(self.model, "logger", None)
        record = getattr(logger, "record", None)
        if callable(record):
            record(key, value)

    def _validate_config(self) -> None:
        cfg = self._config
        if (
            not np.isfinite(cfg.initial_gaussian_std_m)
            or cfg.initial_gaussian_std_m <= 0
        ):
            raise ValueError("Completion DSPDL initial Gaussian std must be positive")
        maximum = float(np.max(self._context_pool.remaining_distances_m))
        if not 0.0 <= cfg.initial_peak_remaining_distance_m <= maximum:
            raise ValueError(
                "Completion DSPDL initial peak is outside the context range"
            )
        if not 0.0 < cfg.initial_uniform_mass < 1.0:
            raise ValueError(
                "Completion DSPDL initial uniform mass must be within (0, 1)"
            )
        if not 0.0 < cfg.target_uniform_mass < 1.0:
            raise ValueError(
                "Completion DSPDL target uniform mass must be within (0, 1)"
            )
        if cfg.target_kl_stop <= 0.0 or cfg.relative_entropy_bound <= 0.0:
            raise ValueError("Completion DSPDL KL bounds must be positive")
        if (
            not np.isfinite(cfg.zeta)
            or cfg.zeta < 0.0
            or not np.isfinite(cfg.completion_floor)
            or not 0.0 <= cfg.completion_floor <= 1.0
        ):
            raise ValueError("Completion DSPDL completion scale is invalid")
        if not np.isfinite(cfg.completion_ema_alpha) or not (
            0.0 < cfg.completion_ema_alpha <= 1.0
        ):
            raise ValueError("Completion DSPDL EMA alpha must be within (0, 1]")
        if not np.all(np.isfinite([cfg.alpha_min, cfg.alpha_max])) or not (
            0.0 <= cfg.alpha_min <= cfg.alpha_max
        ):
            raise ValueError("Completion DSPDL alpha bounds are invalid")
        if cfg.update_interval_rollouts <= 0:
            raise ValueError("Completion DSPDL update interval must be positive")
        weights = cfg.success_base + cfg.stopping_weight + cfg.punctuality_weight
        if (
            not np.isclose(weights, 1.0)
            or min(cfg.success_base, cfg.stopping_weight, cfg.punctuality_weight) < 0.0
        ):
            raise ValueError("Completion DSPDL task-completion weights must sum to one")
