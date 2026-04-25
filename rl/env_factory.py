from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from gymnasium.wrappers import FlattenObservation

from rl.mtto_env import MTTOEnv


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
    )
    env = FlattenObservation(env)
    return env
