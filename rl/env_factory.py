import numpy as np
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.completion_critic import CompletionDSPDLConfig, CompletionTrajectoryAccumulator
from rl.context_pool import ContextPool
from rl.context_sampler import ContextSampler, CurriculumDistributionState
from rl.dspdl import DSPDLStatisticsHub
from rl.mtto_env import MTTOEnv
from rl.operational_stepper import OperationalStepper
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
    stepper: OperationalStepper | None = None,
    context_pool: ContextPool | None = None,
    initial_context_distribution: NDArray[np.floating] | list[float] | None = None,
    curriculum_distribution_state: CurriculumDistributionState | None = None,
    context_sampling_seed: int | None = None,
    dspdl_statistics_hub: DSPDLStatisticsHub | None = None,
    curriculum_env_rank: int | None = None,
    enable_completion_accumulator: bool = False,
    completion_config: CompletionDSPDLConfig | None = None,
    enable_safety_truncation_tracking: bool = False,
    reward_diagnostics_worker_rank: int | None = None,
    reward_diagnostics_rollout_capacity: int | None = None,
):
    if (dspdl_statistics_hub is None) != (curriculum_env_rank is None):
        raise ValueError(
            "DSPDL statistics hub and curriculum environment rank must be set together"
        )
    if dspdl_statistics_hub is not None and enable_completion_accumulator:
        raise ValueError(
            "traditional and completion DSPDL statistics are mutually exclusive"
        )
    if (reward_diagnostics_worker_rank is None) != (
        reward_diagnostics_rollout_capacity is None
    ):
        raise ValueError(
            "reward diagnostics worker rank and rollout capacity must be set together"
        )
    context_sampler: ContextSampler | None = None
    if context_pool is not None:
        if (initial_context_distribution is None) == (
            curriculum_distribution_state is None
        ):
            raise ValueError(
                "exactly one curriculum distribution source is required "
                "with context_pool"
            )
        if (
            dspdl_statistics_hub is not None
            and dspdl_statistics_hub.context_count != context_pool.context_count
        ):
            raise ValueError(
                "statistics hub context count must match the context pool"
            )
        context_sampler = ContextSampler(
            context_pool=context_pool,
            initial_distribution=initial_context_distribution,
            distribution_state=curriculum_distribution_state,
            seed=context_sampling_seed,
        )
    elif (
        initial_context_distribution is not None
        or curriculum_distribution_state is not None
        or dspdl_statistics_hub is not None
    ):
        raise ValueError("curriculum components require context_pool")
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
        stepper=stepper,
        context_sampler=context_sampler,
        dspdl_statistics_hub=dspdl_statistics_hub,
        curriculum_env_rank=curriculum_env_rank,
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
        if enable_completion_accumulator:
            config = completion_config or CompletionDSPDLConfig()
            env.completion_accumulator = CompletionTrajectoryAccumulator(
                observation_shape=env.observation_space.shape,
                success_base=config.success_base,
                stopping_weight=config.stopping_weight,
                punctuality_weight=config.punctuality_weight,
            )
    return env
