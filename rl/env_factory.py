import numpy as np
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.context_pool import ContextPool
from rl.context_sampler import ContextSampler
from rl.dspdl import DSPDLEpisodeAccumulator
from rl.mtto_env import MTTOEnv
from rl.reward_calculator import RewardConfig
from rl.reward_diagnostics import RewardDiagnosticsAccumulator
from rl.safety_statistics import SafetyTruncationBuffer


def make_env(
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    step_distance: float,
    compact_training_info: bool = False,
    enable_trajectory_tracking: bool = False,
    render_mode: str | None = None,
    reward_config: RewardConfig | None = None,
    context_pool: ContextPool | None = None,
    initial_context_distribution: NDArray[np.floating] | list[float] | None = None,
    context_sampling_seed: int | None = None,
    enable_dspdl_accumulator: bool = False,
    enable_safety_truncation_tracking: bool = False,
    reward_diagnostics_worker_rank: int | None = None,
    reward_diagnostics_rollout_capacity: int | None = None,
):
    if (reward_diagnostics_worker_rank is None) != (
        reward_diagnostics_rollout_capacity is None
    ):
        raise ValueError(
            "reward diagnostics worker rank and rollout capacity must be set together"
        )
    env = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=gamma,
        step_distance=step_distance,
        compact_training_info=compact_training_info,
        enable_trajectory_tracking=enable_trajectory_tracking,
        render_mode=render_mode,
        reward_config=reward_config,
        safety_truncation_buffer=(
            SafetyTruncationBuffer() if enable_safety_truncation_tracking else None
        ),
        reward_diagnostics_accumulator=(
            RewardDiagnosticsAccumulator(
                worker_rank=reward_diagnostics_worker_rank,
                rollout_capacity=reward_diagnostics_rollout_capacity,
            )
            if reward_diagnostics_worker_rank is not None
            and reward_diagnostics_rollout_capacity is not None
            else None
        ),
    )
    if context_pool is not None:
        if initial_context_distribution is None:
            raise ValueError(
                "initial_context_distribution is required with context_pool"
            )
        context_sampler = ContextSampler(
            context_pool=context_pool,
            initial_distribution=initial_context_distribution,
            seed=context_sampling_seed,
        )
        env.context_sampler = context_sampler
        if enable_dspdl_accumulator:
            env.dspdl_accumulator = DSPDLEpisodeAccumulator(
                context_count=context_pool.context_count,
                gamma=gamma,
            )
    return env
