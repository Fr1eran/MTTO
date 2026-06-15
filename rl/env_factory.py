import os

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.mtto_env import DEFAULT_PUNCTUALITY_DP_CURVE_DIR, MTTOEnv, RewardConfig


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
    punctuality_dp_curve_dir: str | os.PathLike[str] | None = (
        DEFAULT_PUNCTUALITY_DP_CURVE_DIR
    ),
    punctuality_reference_match_tolerance: float = 1e-3,
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
        punctuality_dp_curve_dir=punctuality_dp_curve_dir,
        punctuality_reference_match_tolerance=punctuality_reference_match_tolerance,
    )
    return env
