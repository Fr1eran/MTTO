"""A source independent, node based speed profile."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mtto.domain.kinematics import segment_acceleration


@dataclass(frozen=True, slots=True, eq=False)
class SpeedProfile:
    position_m: NDArray[np.float64]
    speed_mps: NDArray[np.float64]
    time_s: NDArray[np.float64]
    segment_acceleration_mps2: NDArray[np.float64]
    propulsion_energy_kj: NDArray[np.float64]
    levitation_energy_kj: NDArray[np.float64]

    def __post_init__(self) -> None:
        nodes = (
            "position_m",
            "speed_mps",
            "time_s",
            "propulsion_energy_kj",
            "levitation_energy_kj",
        )
        fields = (*nodes, "segment_acceleration_mps2")
        for name in fields:
            value = getattr(self, name)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")

        count = self.position_m.size
        if count < 1:
            raise ValueError("position_m must contain at least one node")
        for name in nodes:
            if getattr(self, name).size != count:
                raise ValueError(f"{name} must have length {count}")
        if self.segment_acceleration_mps2.size != count - 1:
            raise ValueError("segment_acceleration_mps2 must have length N-1")
        if np.any(self.speed_mps < 0):
            raise ValueError("speed_mps must be nonnegative")
        for name in (
            "position_m",
            "time_s",
            "propulsion_energy_kj",
            "levitation_energy_kj",
        ):
            if np.any(np.diff(getattr(self, name)) < 0):
                raise ValueError(f"{name} must be nondecreasing")
        for name in ("time_s", "propulsion_energy_kj", "levitation_energy_kj"):
            if getattr(self, name)[0] != 0:
                raise ValueError(f"{name} must start at zero")

        for name in fields:
            getattr(self, name).flags.writeable = False

    @property
    def total_energy_kj(self) -> NDArray[np.float64]:
        return self.propulsion_energy_kj + self.levitation_energy_kj

    @classmethod
    def from_arrays(
        cls,
        position_m: ArrayLike,
        speed_mps: ArrayLike,
        time_s: ArrayLike,
        propulsion_energy_kj: ArrayLike,
        levitation_energy_kj: ArrayLike,
    ) -> SpeedProfile:
        position = np.array(position_m, dtype=np.float64)
        speed = np.array(speed_mps, dtype=np.float64)
        time = np.array(time_s, dtype=np.float64)
        propulsion = np.array(propulsion_energy_kj, dtype=np.float64)
        levitation = np.array(levitation_energy_kj, dtype=np.float64)
        if time.size != speed.size:
            raise ValueError("time_s must have the same length as speed_mps")
        acceleration = segment_acceleration(speed, time)
        return cls(position, speed, time, acceleration, propulsion, levitation)
