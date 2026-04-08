from model.safe_guard_utility import SafeGuardUtility
from model.task import Task
from model.track import Track
from model.vehicle import Vehicle
from gymnasium.wrappers import FlattenObservation

from rl.mtto_env import MTTOEnv


def make_env(
    vehicle: Vehicle,
    track: Track,
    safeguard_utility: SafeGuardUtility,
    task: Task,
    gamma: float,
    max_step_distance: float,
    render_mode: str | None = None,
):
    env = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        task=task,
        gamma=gamma,
        max_step_distance=max_step_distance,
        render_mode=render_mode,
    )
    env = FlattenObservation(env)
    return env

