from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.mtto_env import MTTOEnv, RewardConfig
from rl.reference_initial_state_provider import ReferenceInitialStateProvider
from rl.reference_trajectory_sampler import (
    ReferenceTrajectory,
    ReferenceTrajectorySampler,
)


def make_env(
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    max_step_distance: float,
    enable_diagnostics: bool = True,
    diagnostics_interval_steps: int = 1,
    enable_trajectory_tracking: bool = False,
    render_mode: str | None = None,
    reward_config: RewardConfig | None = None,
    reference_trajectory: ReferenceTrajectory | None = None,
    reference_initial_state_weights=None,
    reference_initial_state_seed: int | None = None,
):
    env = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=gamma,
        max_step_distance=max_step_distance,
        enable_diagnostics=enable_diagnostics,
        diagnostics_interval_steps=diagnostics_interval_steps,
        enable_trajectory_tracking=enable_trajectory_tracking,
        render_mode=render_mode,
        reward_config=reward_config,
    )
    if reference_trajectory is not None:
        if reference_initial_state_weights is None:
            raise ValueError(
                "reference_initial_state_weights are required with reference_trajectory"
            )
        sampler = ReferenceTrajectorySampler(
            reference_trajectory,
            stepper=env.stepper,
        )
        env.set_reference_initial_state_provider(
            ReferenceInitialStateProvider(
                sampler=sampler,
                initial_weights=reference_initial_state_weights,
                seed=reference_initial_state_seed,
            )
        )
    return env
