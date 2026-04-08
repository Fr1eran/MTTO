from model.SafeGuard import SafeGuardUtility
from model.Task import Task
from model.Track import Track
from model.Vehicle import Vehicle
from gymnasium.wrappers import FlattenObservation

from rl.MTTOEnv import MTTOEnv


def make_env(
    vehicle: Vehicle,
    track: Track,
    safeguardutility: SafeGuardUtility,
    task: Task,
    gamma: float,
    max_step_distance: float,
    render_mode: str | None = None,
):
    env = MTTOEnv(
        vehicle=vehicle,
        track=track,
        safeguardutility=safeguardutility,
        task=task,
        gamma=gamma,
        max_step_distance=max_step_distance,
        render_mode=render_mode,
    )
    env = FlattenObservation(env)
    return env
