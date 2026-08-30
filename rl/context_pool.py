"""Build immutable DSPDL contexts from a validated reference trajectory."""

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
    "Context",
    "ContextPool",
    "ContextPoolBuilder",
    "ReferenceTrajectory",
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


@dataclass(frozen=True, slots=True)
class Context:
    """One finite DSPDL task context and its materialized initial state."""

    context_index: int
    remaining_distance_m: float
    initial_state: OperationalState


@dataclass(frozen=True)
class ContextPool:
    """Immutable, index-aligned finite DSPDL context pool."""

    contexts: tuple[Context, ...]

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        if not contexts:
            raise ValueError("context pool must be non-empty")
        remaining = np.empty(len(contexts), dtype=np.float64)
        for index, context in enumerate(contexts):
            if context.context_index != index:
                raise ValueError("context indices must be contiguous and index-aligned")
            if (
                not np.isfinite(context.remaining_distance_m)
                or context.remaining_distance_m < 0
            ):
                raise ValueError(
                    "context remaining distance must be finite and non-negative"
                )
            remaining[index] = context.remaining_distance_m
        remaining.flags.writeable = False
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "_remaining_distances_m", remaining)

    @property
    def context_count(self) -> int:
        return len(self.contexts)

    @property
    def remaining_distances_m(self) -> NDArray[np.float64]:
        return self._remaining_distances_m

    def context_at(self, context_index: int) -> Context:
        if not isinstance(context_index, (int, np.integer)):
            raise TypeError("context_index must be an integer")
        index = int(context_index)
        if not 0 <= index < self.context_count:
            raise IndexError(
                f"context index {index} is outside [0, {self.context_count - 1}]"
            )
        return self.contexts[index]


class ContextPoolBuilder:
    """Reconstruct a finite context pool from a complete reference trajectory.

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
        self._position = position
        self._speed = speed
        self._operation_time = operation_time

    @classmethod
    def from_arrays(
        cls,
        *,
        position_m: NDArray[np.floating[Any]] | list[float],
        speed_mps: NDArray[np.floating[Any]] | list[float],
        cumulative_time_s: NDArray[np.floating[Any]] | list[float],
        stepper: OperationalStepper,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextPool:
        """Build a context pool directly from source-independent arrays."""
        return cls(
            ReferenceTrajectory(
                position_m=np.asarray(position_m, dtype=np.float64),
                speed_mps=np.asarray(speed_mps, dtype=np.float64),
                cumulative_time_s=np.asarray(cumulative_time_s, dtype=np.float64),
                metadata={} if metadata is None else metadata,
            ),
            stepper=stepper,
        ).build()

    @property
    def trajectory(self) -> ReferenceTrajectory:
        return self._trajectory

    def build(self) -> ContextPool:
        contexts = self._reconstruct_contexts(
            self._position, self._speed, self._operation_time
        )
        return ContextPool(contexts)

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
            0.0,
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
        max_step = self._stepper.step_distance_m
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

    def _reconstruct_contexts(
        self,
        position: NDArray[np.float64],
        speed: NDArray[np.float64],
        operation_time: NDArray[np.float64],
    ) -> tuple[Context, ...]:
        state = self._stepper.reset()
        self._validate_replayed_state(
            state, position[0], speed[0], operation_time[0], 0
        )
        contexts: list[Context] = [
            self._build_context(0, state, position[0], speed[0], operation_time[0])
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
            if not is_final_transition:
                contexts.append(
                    self._build_context(
                        index + 1,
                        state,
                        position[index + 1],
                        speed[index + 1],
                        operation_time[index + 1],
                    )
                )

        return tuple(contexts)

    def _build_context(
        self,
        index: int,
        runtime_state: OperationalState,
        position_m: float,
        speed_mps: float,
        operation_time_s: float,
    ) -> Context:
        del speed_mps, operation_time_s
        return Context(
            context_index=index,
            remaining_distance_m=abs(
                float(self._stepper.train_service.target_position) - float(position_m)
            ),
            initial_state=runtime_state,
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
