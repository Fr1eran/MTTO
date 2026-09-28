from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mtto.domain.speed_profile import SpeedProfile
from mtto.rl.state import TerminationReason

if TYPE_CHECKING:
    from mtto.rl.env import MTTOEnv


def calculate_route_completion_ratio(
    *, start_position_m: float, target_position_m: float, final_position_m: float
) -> float:
    """Return clipped net progress along the route's running direction."""
    route_delta = float(target_position_m) - float(start_position_m)
    if not np.isfinite(route_delta) or route_delta == 0.0:
        raise ValueError("start and target positions must define a finite route")
    progress = (
        (float(final_position_m) - float(start_position_m))
        * np.sign(route_delta)
        / abs(route_delta)
    )
    if not np.isfinite(progress):
        raise ValueError("final_position_m must be finite")
    return float(np.clip(progress, 0.0, 1.0))


@dataclass(frozen=True, slots=True, eq=False)
class RLRun:
    profile: SpeedProfile
    termination_reason: TerminationReason
    total_reward: float
    steps: int
    deterministic: bool


def run_policy(policy: Any, env: MTTOEnv, *, deterministic: bool = True) -> RLRun:
    state = env.initial_state()
    states = [state]
    total_reward = 0.0
    observation_buffer = np.empty(
        env.observation_builder.OBSERVATION_DIM, dtype=np.float32
    )

    while True:
        observation = env.observation_builder.build(state, out=observation_buffer)
        if hasattr(policy, "predict"):
            action, _ = policy.predict(observation, deterministic=deterministic)
        else:
            action = policy(observation)
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        acceleration = env.observation_builder.denormalize_action(action_value)
        result = env.transition(state, acceleration)
        total_reward += env.reward_calculator.calculate(state, result, env.task).total
        states.append(result.step_end_state)
        state = result.next_state
        if result.termination_reason is not None:
            break

    profile = SpeedProfile.from_arrays(
        [node.s_m for node in states],
        [node.v_mps for node in states],
        [node.t_s for node in states],
        [node.propulsion_energy_kj for node in states],
        [node.levitation_energy_kj for node in states],
    )
    return RLRun(
        profile, result.termination_reason, total_reward, len(states) - 1, deterministic
    )
