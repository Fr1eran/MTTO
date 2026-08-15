import numpy as np
import pytest

from rl.operational_state import ViolationCode
from rl.safety_statistics import SafetyTruncationBuffer


def test_safety_truncation_buffer_records_only_speed_bound_truncations() -> None:
    buffer = SafetyTruncationBuffer()

    buffer.record(
        position_m=100.0,
        violation_code=ViolationCode.SPEED_LOW,
        truncated=True,
    )
    buffer.record(
        position_m=200.0,
        violation_code=ViolationCode.SPEED_HIGH,
        truncated=True,
    )
    buffer.record(
        position_m=300.0,
        violation_code=ViolationCode.SPEED_LOW,
        truncated=False,
    )
    buffer.record(
        position_m=400.0,
        violation_code=ViolationCode.FAILED_STOP,
        truncated=True,
    )
    buffer.record(
        position_m=500.0,
        violation_code=ViolationCode.STEP_LIMIT,
        truncated=True,
    )

    batch = buffer.drain()

    assert batch["position_m"].dtype == np.float32
    assert batch["violation_code"].dtype == np.int8
    np.testing.assert_allclose(batch["position_m"], [100.0, 200.0])
    np.testing.assert_array_equal(batch["violation_code"], [2, 3])
    assert buffer.drain()["position_m"].size == 0


def test_safety_truncation_buffer_rejects_non_finite_recorded_position() -> None:
    buffer = SafetyTruncationBuffer()

    with pytest.raises(ValueError, match="finite"):
        buffer.record(
            position_m=np.nan,
            violation_code=ViolationCode.SPEED_LOW,
            truncated=True,
        )
