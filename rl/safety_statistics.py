"""Worker-local buffering for safety-related truncation events."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from rl.operational_state import ViolationCode


class SafetyTruncationBatch(TypedDict):
    """Compact payload transferred from one environment at rollout boundaries."""

    position_m: NDArray[np.float32]
    violation_code: NDArray[np.int8]


class SafetyTruncationBuffer:
    """Collect only speed-bound truncations without per-step VecEnv telemetry."""

    def __init__(self) -> None:
        self._positions_m: list[float] = []
        self._violation_codes: list[int] = []

    def record(
        self,
        *,
        position_m: float,
        violation_code: ViolationCode | int,
        truncated: bool,
    ) -> None:
        code = ViolationCode(int(violation_code))
        if not truncated or code not in {
            ViolationCode.SPEED_LOW,
            ViolationCode.SPEED_HIGH,
        }:
            return
        position = float(position_m)
        if not np.isfinite(position):
            raise ValueError("safety truncation position must be finite")
        self._positions_m.append(position)
        self._violation_codes.append(int(code))

    def drain(self) -> SafetyTruncationBatch:
        payload: SafetyTruncationBatch = {
            "position_m": np.asarray(self._positions_m, dtype=np.float32),
            "violation_code": np.asarray(self._violation_codes, dtype=np.int8),
        }
        self._positions_m.clear()
        self._violation_codes.clear()
        return payload
