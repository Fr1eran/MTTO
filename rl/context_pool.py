"""Build immutable DSPL contexts from a validated reference trajectory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from model.ocs import SPSState
from rl.operational_state import OperationalState
from rl.operational_stepper import OperationalStepper

__all__ = [
    "Context",
    "ContextPool",
    "ContextPoolBuilder",
    "ReferenceTrajectory",
]


_POSITION_ATOL_M = 1e-3
_TARGET_POSITION_ATOL_M = 0.3
_STOPPED_SPEED_ATOL_MPS = 0.01
_SCHEDULE_TIME_ATOL_S = 10.0
_SAFETY_ATOL_MPS = 1e-6
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
    """One finite DSPL task context and its materialized initial state."""

    context_index: int
    remaining_distance_m: float
    initial_state: OperationalState


@dataclass(frozen=True)
class ContextPool:
    """Immutable, index-aligned finite DSPL context pool."""

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
    """Build uniformly spaced DSPL contexts from trajectory-prefix state."""

    def __init__(
        self,
        trajectory: ReferenceTrajectory,
        *,
        stepper: OperationalStepper,
        context_count: int,
    ) -> None:
        self._trajectory: ReferenceTrajectory = trajectory
        self._stepper: OperationalStepper = stepper
        if context_count <= 0:
            raise ValueError("context_count must be positive")
        self._context_count = int(context_count)
        self._position, self._speed, self._operation_time = (
            self._validate_and_normalize_source()
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
        context_count: int,
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
            context_count=context_count,
        ).build()

    @property
    def trajectory(self) -> ReferenceTrajectory:
        return self._trajectory

    def build(self) -> ContextPool:
        source_sps, source_energy, source_acceleration = self._replay_source_prefix()
        travelled = (
            np.arange(self._context_count, dtype=np.float64)
            * self._stepper.whole_distance_m
            / self._context_count
        )
        context_position = (
            float(self._stepper.train_service.start_position)
            + self._stepper.direction * travelled
        )
        source_travelled = self._stepper.direction * (
            self._position - self._position[0]
        )
        context_speed = np.interp(travelled, source_travelled, self._speed)
        context_time = np.interp(travelled, source_travelled, self._operation_time)
        contexts = tuple(
            self._materialize_context(
                context_index=index,
                travelled_m=float(distance),
                position_m=float(context_position[index]),
                speed_mps=float(context_speed[index]),
                operation_time_s=float(context_time[index]),
                source_travelled=source_travelled,
                source_sps=source_sps,
                source_energy=source_energy,
                source_acceleration=source_acceleration,
            )
            for index, distance in enumerate(travelled)
        )
        return ContextPool(contexts)

    def _validate_and_normalize_source(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        service = self._stepper.train_service
        position = self._trajectory.position_m.copy()
        speed = self._trajectory.speed_mps.copy()
        operation_time = self._trajectory.cumulative_time_s.copy()
        direction = self._stepper.direction
        if not np.isclose(
            position[0],
            service.start_position,
            atol=_POSITION_ATOL_M,
            rtol=0.0,
        ):
            raise ValueError("reference trajectory start position does not match task")
        target_error_m = abs(float(position[-1]) - float(service.target_position))
        if target_error_m >= _TARGET_POSITION_ATOL_M:
            raise ValueError(
                "reference trajectory target error must be less than 0.3 m"
            )
        if np.any(direction * np.diff(position) <= 0.0):
            raise ValueError("reference trajectory direction does not match task")
        if abs(float(speed[0])) > _STOPPED_SPEED_ATOL_MPS:
            raise ValueError("reference trajectory must start near zero speed")
        if abs(float(speed[-1])) > _STOPPED_SPEED_ATOL_MPS:
            raise ValueError("reference trajectory must end near zero speed")
        operation_time -= operation_time[0]
        if (
            abs(float(operation_time[-1]) - float(service.schedule_time))
            >= _SCHEDULE_TIME_ATOL_S
        ):
            raise ValueError(
                "reference trajectory total time error must be less than 10 s"
            )
        position[0] = float(service.start_position)
        position[-1] = float(service.target_position)
        speed[0] = speed[-1] = 0.0
        if np.any(direction * np.diff(position) <= 0.0):
            raise ValueError(
                "normalized reference trajectory direction does not match task"
            )
        return position, speed, operation_time

    def _replay_source_prefix(
        self,
    ) -> tuple[tuple[SPSState, ...], NDArray[np.float64], NDArray[np.float64]]:
        count = self._position.size
        sps_states: list[SPSState] = [self._stepper.sps.initial_state()]
        energy = np.zeros(count, dtype=np.float64)
        acceleration = np.zeros(count, dtype=np.float64)
        initial_state = self._stepper.build_state(
            position_m=float(self._position[0]),
            speed_mps=float(self._speed[0]),
            acceleration_mps2=0.0,
            operation_time_s=0.0,
            energy_consumption_kj=0.0,
            step_count=0,
            sps_state=sps_states[0],
        )
        self._validate_safe_speed(initial_state, "source node 0")
        for index in range(1, count):
            duration = float(
                self._operation_time[index] - self._operation_time[index - 1]
            )
            segment_acceleration = float(
                (self._speed[index] - self._speed[index - 1]) / duration
            )
            self._validate_action_acceleration(
                segment_acceleration, f"source segment {index - 1}"
            )
            distance = abs(float(self._position[index] - self._position[index - 1]))
            propulsion, levitation = self._stepper.ecc.calc_energy(
                begin_pos=float(self._position[index - 1]),
                begin_speed=float(self._speed[index - 1]),
                acc=segment_acceleration,
                distance=distance,
                direction=self._stepper.direction,
                operation_time=duration,
                vehicle=self._stepper.vehicle,
                track=self._stepper.track,
            )
            energy[index] = energy[index - 1] + float(propulsion + levitation)
            sps_state = self._stepper.sps.advance(
                sps_states[-1],
                position_m=float(self._position[index]),
                speed_mps=float(self._speed[index]),
                time_s=float(self._operation_time[index]),
            )
            sps_states.append(sps_state)
            acceleration[index] = segment_acceleration
            state = self._stepper.build_state(
                position_m=float(self._position[index]),
                speed_mps=float(self._speed[index]),
                acceleration_mps2=segment_acceleration,
                operation_time_s=float(self._operation_time[index]),
                energy_consumption_kj=float(energy[index]),
                step_count=0,
                sps_state=sps_state,
            )
            self._validate_safe_speed(state, f"source node {index}")
        return tuple(sps_states), energy, acceleration

    def _materialize_context(
        self,
        *,
        context_index: int,
        travelled_m: float,
        position_m: float,
        speed_mps: float,
        operation_time_s: float,
        source_travelled: NDArray[np.float64],
        source_sps: tuple[SPSState, ...],
        source_energy: NDArray[np.float64],
        source_acceleration: NDArray[np.float64],
    ) -> Context:
        source_index = int(
            np.searchsorted(source_travelled, travelled_m, side="right") - 1
        )
        source_index = max(0, min(source_index, source_travelled.size - 1))
        at_source_node = np.isclose(
            travelled_m, source_travelled[source_index], atol=_POSITION_ATOL_M, rtol=0.0
        )
        if at_source_node:
            acceleration = float(source_acceleration[source_index])
            energy = float(source_energy[source_index])
            sps_state = source_sps[source_index]
        else:
            duration = operation_time_s - float(self._operation_time[source_index])
            acceleration = (speed_mps - float(self._speed[source_index])) / duration
            self._validate_action_acceleration(
                acceleration, f"context {context_index} partial segment"
            )
            distance = travelled_m - float(source_travelled[source_index])
            propulsion, levitation = self._stepper.ecc.calc_energy(
                begin_pos=float(self._position[source_index]),
                begin_speed=float(self._speed[source_index]),
                acc=acceleration,
                distance=distance,
                direction=self._stepper.direction,
                operation_time=duration,
                vehicle=self._stepper.vehicle,
                track=self._stepper.track,
            )
            energy = float(source_energy[source_index] + propulsion + levitation)
            sps_state = self._stepper.sps.advance(
                source_sps[source_index],
                position_m=position_m,
                speed_mps=speed_mps,
                time_s=operation_time_s,
            )
        state = self._stepper.build_state(
            position_m=position_m,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration,
            operation_time_s=operation_time_s,
            energy_consumption_kj=energy,
            step_count=0,
            sps_state=sps_state,
        )
        self._validate_safe_speed(state, f"context {context_index}")
        return Context(
            context_index=context_index,
            remaining_distance_m=self._stepper.whole_distance_m - travelled_m,
            initial_state=state,
        )

    def _validate_action_acceleration(self, acceleration: float, source: str) -> None:
        vehicle = self._stepper.vehicle
        if (
            not np.isfinite(acceleration)
            or acceleration < float(vehicle.max_dec) - _ACC_ATOL_MPS2
            or acceleration > float(vehicle.max_acc) + _ACC_ATOL_MPS2
        ):
            raise ValueError(
                f"reference acceleration is outside vehicle bounds at {source}"
            )

    @staticmethod
    def _validate_safe_speed(state: OperationalState, source: str) -> None:
        if (
            state.speed_mps < state.min_speed_mps - _SAFETY_ATOL_MPS
            or state.speed_mps > state.max_speed_mps + _SAFETY_ATOL_MPS
        ):
            raise ValueError(f"reference speed violates safety bounds at {source}")
