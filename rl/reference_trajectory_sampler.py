"""Reconstruct RL-compatible initial states from reference trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from model.common import calc_transition_to_speed_scalar_numba
from rl.operational_state import OperationalState
from rl.operational_stepper import OperationalStepper

__all__ = [
    "ReferenceTrajectory",
    "ReferenceTrajectorySampler",
    "ReferenceTrajectoryState",
]


_POSITION_ATOL_M = 1e-3
_SPEED_ATOL_MPS = 1e-4
_TIME_ATOL_S = 1e-3
_ACC_ATOL_MPS2 = 1e-9


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Validated, source-agnostic position, speed, and cumulative-time data."""

    position_m: NDArray[np.float64]
    speed_mps: NDArray[np.float64]
    cumulative_time_s: NDArray[np.float64]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        speed = np.asarray(self.speed_mps, dtype=np.float64)
        cumulative_time = np.asarray(self.cumulative_time_s, dtype=np.float64)
        if position.ndim != 1 or speed.ndim != 1 or cumulative_time.ndim != 1:
            raise ValueError("reference trajectory arrays must be one-dimensional")
        if not (position.size == speed.size == cumulative_time.size):
            raise ValueError("reference trajectory arrays must have equal length")
        if position.size < 2:
            raise ValueError("reference trajectory must contain at least two nodes")
        if not (
            np.all(np.isfinite(position))
            and np.all(np.isfinite(speed))
            and np.all(np.isfinite(cumulative_time))
        ):
            raise ValueError(
                "reference trajectory arrays must contain only finite values"
            )
        if np.any(speed < 0.0):
            raise ValueError("reference trajectory speeds must be non-negative")

        delta_position = np.diff(position)
        if not (np.all(delta_position > 0.0) or np.all(delta_position < 0.0)):
            raise ValueError(
                "reference trajectory positions must be strictly monotonic"
            )
        if np.any(np.diff(cumulative_time) <= 0.0):
            raise ValueError("reference trajectory cumulative times must increase")

        position = position.copy()
        speed = speed.copy()
        cumulative_time = cumulative_time.copy()
        position.flags.writeable = False
        speed.flags.writeable = False
        cumulative_time.flags.writeable = False
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "speed_mps", speed)
        object.__setattr__(self, "cumulative_time_s", cumulative_time)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ReferenceTrajectoryState:
    """One replayed RL-grid node and its complete runtime state."""

    reference_index: int
    position_m: float
    speed_mps: float
    operation_time_s: float
    remaining_distance_m: float
    runtime_state: OperationalState


class ReferenceTrajectorySampler:
    """Sample fully reconstructed states from a complete reference trajectory.

    The source trajectory is first checked for piecewise constant-acceleration
    consistency.  It is then sampled on the environment's action grid.  Since
    a grid step may span multiple source segments, its time is recomputed from
    its grid endpoints so that one :class:`OperationalStepper` transition can
    reproduce the state exactly.
    """

    def __init__(
        self,
        trajectory: ReferenceTrajectory,
        *,
        stepper: OperationalStepper,
    ) -> None:
        if not isinstance(trajectory, ReferenceTrajectory):
            raise TypeError("trajectory must be a ReferenceTrajectory")
        self._trajectory: ReferenceTrajectory = trajectory
        self._stepper: OperationalStepper = stepper
        self._validate_task_coverage()
        self._validate_source_timing()
        position, speed, operation_time = self._resample_on_rl_grid()
        self._states: tuple[ReferenceTrajectoryState, ...] = self._replay_states(
            position, speed, operation_time
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        position_m: NDArray[np.floating[Any]] | list[float],
        speed_mps: NDArray[np.floating[Any]] | list[float],
        cumulative_time_s: NDArray[np.floating[Any]] | list[float],
        stepper: OperationalStepper,
        metadata: Mapping[str, object] | None = None,
    ) -> ReferenceTrajectorySampler:
        """Build a sampler directly from a source-independent trajectory."""
        return cls(
            ReferenceTrajectory(
                position_m=np.asarray(position_m, dtype=np.float64),
                speed_mps=np.asarray(speed_mps, dtype=np.float64),
                cumulative_time_s=np.asarray(cumulative_time_s, dtype=np.float64),
                metadata={} if metadata is None else metadata,
            ),
            stepper=stepper,
        )

    @property
    def trajectory(self) -> ReferenceTrajectory:
        return self._trajectory

    @property
    def node_count(self) -> int:
        """Number of replayed nodes, including the terminal node."""
        return len(self._states)

    @property
    def eligible_node_count(self) -> int:
        """Number of nodes eligible for reset sampling (excludes terminal)."""
        return max(0, self.node_count - 1)

    @property
    def states(self) -> tuple[ReferenceTrajectoryState, ...]:
        return self._states

    def state_at(self, index: int) -> ReferenceTrajectoryState:
        """Return a replayed node, including the final terminal node."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")
        value = int(index)
        if not 0 <= value < self.node_count:
            raise IndexError(f"index {value} is outside [0, {self.node_count - 1}]")
        return self._states[value]

    def sample(
        self,
        rng: np.random.Generator,
        *,
        weights: NDArray[np.floating[Any]] | list[float] | None = None,
        index_range: tuple[int, int] | None = None,
        remaining_distance_range_m: tuple[float, float] | None = None,
    ) -> ReferenceTrajectoryState:
        """Sample a non-terminal replayed node from one optional selection range."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if index_range is not None and remaining_distance_range_m is not None:
            raise ValueError(
                "Specify at most one of index_range and remaining_distance_range_m."
            )

        candidates = np.arange(self.eligible_node_count, dtype=np.int64)
        if index_range is not None:
            lower, upper = self._validate_index_range(index_range)
            candidates = np.arange(lower, upper + 1, dtype=np.int64)
        elif remaining_distance_range_m is not None:
            lower, upper = self._validate_remaining_distance_range(
                remaining_distance_range_m
            )
            candidates = np.asarray(
                [
                    state.reference_index
                    for state in self._states[:-1]
                    if lower <= state.remaining_distance_m <= upper
                ],
                dtype=np.int64,
            )

        if candidates.size == 0:
            raise ValueError("The requested sampling range contains no eligible nodes.")
        probabilities = self._resolve_probabilities(weights, candidates)
        return self.state_at(int(rng.choice(candidates, p=probabilities)))

    def _validate_task_coverage(self) -> None:
        service = self._stepper.train_service
        position = self._trajectory.position_m
        speed = self._trajectory.speed_mps
        cumulative_time = self._trajectory.cumulative_time_s
        direction = self._stepper.direction
        if not np.isclose(
            position[0],
            service.start_position,
            atol=_POSITION_ATOL_M,
            rtol=0.0,
        ):
            raise ValueError("reference trajectory start position does not match task")
        if not np.isclose(
            position[-1],
            service.target_position,
            atol=_POSITION_ATOL_M,
            rtol=0.0,
        ):
            raise ValueError("reference trajectory target position does not match task")
        if np.any(direction * np.diff(position) <= 0.0):
            raise ValueError("reference trajectory direction does not match task")
        if not np.isclose(
            speed[0],
            service.start_speed,
            atol=_SPEED_ATOL_MPS,
            rtol=0.0,
        ):
            raise ValueError("reference trajectory start speed does not match task")
        if not np.isclose(speed[-1], 0.0, atol=_SPEED_ATOL_MPS, rtol=0.0):
            raise ValueError("reference trajectory must end at zero speed")
        if not np.isclose(
            cumulative_time[0],
            0.0,
            atol=_TIME_ATOL_S,
            rtol=0.0,
        ):
            raise ValueError("reference trajectory must start at operation time zero")

    def _validate_source_timing(self) -> None:
        position = self._trajectory.position_m
        speed = self._trajectory.speed_mps
        cumulative_time = self._trajectory.cumulative_time_s
        for index in range(position.size - 1):
            distance = abs(float(position[index + 1] - position[index]))
            duration = self._duration_to_speed(
                float(speed[index]),
                float(speed[index + 1]),
                distance,
            )
            recorded_duration = float(
                cumulative_time[index + 1] - cumulative_time[index]
            )
            if not np.isclose(duration, recorded_duration, atol=_TIME_ATOL_S, rtol=0.0):
                raise ValueError(
                    "reference trajectory cumulative time is inconsistent with "
                    + f"constant-acceleration segment {index}"
                )

    def _resample_on_rl_grid(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        service = self._stepper.train_service
        total_distance = self._stepper.whole_distance_m
        max_step = self._stepper.max_step_distance_m
        node_count = int(np.ceil(total_distance / max_step)) + 1
        travelled = np.minimum(
            np.arange(node_count, dtype=np.float64) * max_step,
            total_distance,
        )
        travelled[-1] = total_distance
        position = float(service.start_position) + self._stepper.direction * travelled
        speed = self._interpolate_source_speed(travelled)

        operation_time = np.zeros(node_count, dtype=np.float64)
        for index in range(node_count - 1):
            distance = float(travelled[index + 1] - travelled[index])
            duration = self._duration_to_speed(
                float(speed[index]),
                float(speed[index + 1]),
                distance,
            )
            operation_time[index + 1] = operation_time[index] + duration

        return position, speed, operation_time

    def _interpolate_source_speed(
        self, travelled_m: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        source_travelled = self._stepper.direction * (
            self._trajectory.position_m - self._trajectory.position_m[0]
        )
        source_speed = self._trajectory.speed_mps
        result = np.empty(travelled_m.size, dtype=np.float64)
        for index, distance in enumerate(travelled_m):
            if np.isclose(distance, source_travelled[-1], atol=_POSITION_ATOL_M):
                result[index] = float(source_speed[-1])
                continue
            segment = int(np.searchsorted(source_travelled, distance, side="right") - 1)
            segment = min(max(segment, 0), source_travelled.size - 2)
            begin_distance = float(source_travelled[segment])
            segment_distance = float(source_travelled[segment + 1] - begin_distance)
            begin_speed = float(source_speed[segment])
            end_speed = float(source_speed[segment + 1])
            acceleration = (end_speed**2 - begin_speed**2) / (2.0 * segment_distance)
            speed_sq = begin_speed**2 + 2.0 * acceleration * (
                float(distance) - begin_distance
            )
            if speed_sq < -(_SPEED_ATOL_MPS**2):
                raise ValueError("reference interpolation produced an invalid speed")
            result[index] = float(np.sqrt(max(speed_sq, 0.0)))
        return result

    def _replay_states(
        self,
        position: NDArray[np.float64],
        speed: NDArray[np.float64],
        operation_time: NDArray[np.float64],
    ) -> tuple[ReferenceTrajectoryState, ...]:
        state = self._stepper.reset()
        self._validate_replayed_state(
            state, position[0], speed[0], operation_time[0], 0
        )
        states: list[ReferenceTrajectoryState] = [
            self._make_reference_state(
                0, state, position[0], speed[0], operation_time[0]
            )
        ]

        for index in range(position.size - 1):
            distance = abs(float(position[index + 1] - position[index]))
            acceleration, _ = calc_transition_to_speed_scalar_numba(
                float(speed[index]),
                float(speed[index + 1]),
                distance,
            )
            self._validate_action_acceleration(float(acceleration), index)
            transition = self._stepper.advance(
                state,
                float(acceleration),
                requested_distance_m=distance,
            )
            is_final_transition = index == position.size - 2
            if is_final_transition:
                if not transition.terminated or transition.truncated:
                    raise ValueError(
                        "reference trajectory does not terminate successfully at target"
                    )
            elif transition.terminated or transition.truncated:
                raise ValueError(
                    f"reference trajectory becomes done before grid node {index + 1}"
                )

            state = transition.next_state
            self._validate_replayed_state(
                state,
                position[index + 1],
                speed[index + 1],
                operation_time[index + 1],
                index + 1,
            )
            states.append(
                self._make_reference_state(
                    index + 1,
                    state,
                    position[index + 1],
                    speed[index + 1],
                    operation_time[index + 1],
                )
            )

        return tuple(states)

    def _make_reference_state(
        self,
        index: int,
        runtime_state: OperationalState,
        position_m: float,
        speed_mps: float,
        operation_time_s: float,
    ) -> ReferenceTrajectoryState:
        return ReferenceTrajectoryState(
            reference_index=index,
            position_m=float(position_m),
            speed_mps=float(speed_mps),
            operation_time_s=float(operation_time_s),
            remaining_distance_m=abs(
                float(self._stepper.train_service.target_position) - float(position_m)
            ),
            runtime_state=runtime_state,
        )

    def _validate_replayed_state(
        self,
        state: OperationalState,
        expected_position: float,
        expected_speed: float,
        expected_time: float,
        index: int,
    ) -> None:
        if not np.isclose(
            state.position_m, expected_position, atol=_POSITION_ATOL_M, rtol=0.0
        ):
            raise ValueError(f"replayed position mismatch at grid node {index}")
        if not np.isclose(
            state.speed_mps, expected_speed, atol=_SPEED_ATOL_MPS, rtol=0.0
        ):
            raise ValueError(f"replayed speed mismatch at grid node {index}")
        if not np.isclose(
            state.operation_time_s,
            expected_time,
            atol=_TIME_ATOL_S,
            rtol=0.0,
        ):
            raise ValueError(f"replayed operation time mismatch at grid node {index}")

    def _validate_action_acceleration(self, acceleration: float, index: int) -> None:
        vehicle = self._stepper.vehicle
        if (
            not np.isfinite(acceleration)
            or acceleration < float(vehicle.max_dec) - _ACC_ATOL_MPS2
            or acceleration > float(vehicle.max_acc) + _ACC_ATOL_MPS2
        ):
            raise ValueError(
                "reference action acceleration is outside RL action bounds "
                f"at grid node {index}"
            )

    @staticmethod
    def _duration_to_speed(
        begin_speed: float, end_speed: float, distance: float
    ) -> float:
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("reference segment distance must be finite and positive")
        if begin_speed + end_speed <= _SPEED_ATOL_MPS:
            raise ValueError(
                "reference trajectory cannot traverse a segment at zero speed"
            )
        _, duration = calc_transition_to_speed_scalar_numba(
            begin_speed, end_speed, distance
        )
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("reference segment duration must be finite and positive")
        return float(duration)

    def _validate_index_range(self, index_range: tuple[int, int]) -> tuple[int, int]:
        if len(index_range) != 2:
            raise ValueError("index_range must contain exactly two indices")
        lower, upper = index_range
        if not isinstance(lower, (int, np.integer)) or not isinstance(
            upper, (int, np.integer)
        ):
            raise TypeError("index_range values must be integers")
        lower_value, upper_value = int(lower), int(upper)
        if lower_value > upper_value:
            raise ValueError("index_range lower bound must not exceed upper bound")
        if lower_value < 0 or upper_value >= self.eligible_node_count:
            raise ValueError(
                "index_range must stay within non-terminal reference node bounds"
            )
        return lower_value, upper_value

    @staticmethod
    def _validate_remaining_distance_range(
        remaining_distance_range_m: tuple[float, float],
    ) -> tuple[float, float]:
        if len(remaining_distance_range_m) != 2:
            raise ValueError(
                "remaining_distance_range_m must contain exactly two values"
            )
        lower, upper = map(float, remaining_distance_range_m)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0.0:
            raise ValueError(
                "remaining_distance_range_m must contain finite non-negative values"
            )
        if lower > upper:
            raise ValueError(
                "remaining_distance_range_m lower bound must not exceed upper bound"
            )
        return lower, upper

    def _resolve_probabilities(
        self,
        weights: NDArray[np.floating[Any]] | list[float] | None,
        candidates: NDArray[np.int64],
    ) -> NDArray[np.float64] | None:
        if weights is None:
            return None
        all_weights = np.asarray(weights, dtype=np.float64)
        if all_weights.ndim != 1 or all_weights.size != self.node_count:
            raise ValueError("weights must be one-dimensional with one value per node")
        if not np.all(np.isfinite(all_weights)) or np.any(all_weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        selected = all_weights[candidates]
        total = float(np.sum(selected))
        if total <= 0.0:
            raise ValueError(
                "weights over the requested sampling range must sum to > 0"
            )
        return selected / total
