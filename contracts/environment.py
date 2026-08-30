"""Typed snapshots exposed by the simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import JSONMapping, from_dict, to_dict


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Termination state returned by one environment transition."""

    terminated: bool
    truncated: bool

    def to_mapping(self) -> JSONMapping:
        return to_dict(self)

    @classmethod
    def from_mapping(cls, payload: object) -> EpisodeOutcome:
        return from_dict(cls, payload, context="episode_outcome")


@dataclass(frozen=True, slots=True)
class EpisodeInfo:
    """Canonical episode snapshot with units encoded in field names."""

    position_m: float
    speed_mps: float
    stopping_point_index: int = field(metadata={"minimum": -1})
    operation_time_s: float
    redundant_operation_time_s: float
    energy_consumption_j: float
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float

    def __post_init__(self) -> None:
        for name in (
            "position_m",
            "speed_mps",
            "operation_time_s",
            "redundant_operation_time_s",
            "energy_consumption_j",
            "comfort_tav",
            "comfort_er_pct",
            "comfort_rms",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
        # ``-1`` is the domain sentinel used before a stopping point has been
        # selected.  It is valid in an intermediate Gym transition snapshot.
        if not isinstance(self.stopping_point_index, int) or isinstance(
            self.stopping_point_index, bool
        ):
            raise TypeError("stopping_point_index must be an integer")

    def to_mapping(self) -> JSONMapping:
        payload = to_dict(self)
        for name in payload.keys() - {"stopping_point_index"}:
            payload[name] = float(payload[name])  # type: ignore[arg-type]
        return payload

    @classmethod
    def from_mapping(cls, payload: object) -> EpisodeInfo:
        return from_dict(cls, payload, context="episode_info")
