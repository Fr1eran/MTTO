from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
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
    enable_safety_truncation_tracking: bool = False,
    reward_diagnostics_worker_rank: int | None = None,
    reward_diagnostics_rollout_capacity: int | None = None,
) -> MTTOEnv:
    if (reward_diagnostics_worker_rank is None) != (
        reward_diagnostics_rollout_capacity is None
    ):
        raise ValueError(
            "reward diagnostics worker rank and rollout capacity must be set together"
        )
    return MTTOEnv(
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
