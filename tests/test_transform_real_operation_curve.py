from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.transform_real_operation_curve import (
    recompute_time_and_acceleration,
    save_transformed_operation_curve,
    transform_operation_curve_arrays,
)


def test_transform_operation_curve_arrays_maps_endpoints_and_midpoint() -> None:
    result = transform_operation_curve_arrays(
        source_position_m=[900.0, 1500.0, 29240.0],
        speed_kmh=[36.0, 36.0, 72.0],
        acc_mps2=[0.0, 1.0, -0.5],
        time_s=[0.0, 10.0, 433.8],
        start_position_m=135.0,
        target_position_m=29270.046,
    )

    position_m = result["position_m"]
    assert isinstance(position_m, np.ndarray)
    np.testing.assert_allclose(position_m[0], 135.0)
    np.testing.assert_allclose(position_m[-1], 29270.046)

    source_progress = (1500.0 - 900.0) / (29240.0 - 900.0)
    expected_mid = 135.0 + source_progress * (29270.046 - 135.0)
    np.testing.assert_allclose(position_m[1], expected_mid)

    np.testing.assert_allclose(result["speed_mps"], [10.0, 10.0, 20.0])
    assert np.asarray(result["time_s"])[0] == pytest.approx(0.0)
    assert np.asarray(result["time_s"])[-1] > 433.8
    assert np.asarray(result["acc_mps2"])[0] == pytest.approx(0.0)
    assert np.asarray(result["acc_mps2"])[1] == pytest.approx(0.0)
    np.testing.assert_allclose(result["source_position_m"], [900.0, 1500.0, 29240.0])
    np.testing.assert_allclose(result["source_time_s"], [0.0, 10.0, 433.8])
    np.testing.assert_allclose(result["source_acc_mps2"], [0.0, 1.0, -0.5])
    np.testing.assert_allclose(
        result["position_scale"],
        (29270.046 - 135.0) / (29240.0 - 900.0),
    )


def test_save_transformed_operation_curve_writes_expected_npz_keys(
    tmp_path: Path,
) -> None:
    result = transform_operation_curve_arrays(
        source_position_m=[900.0, 29240.0],
        speed_kmh=[36.0, 36.0],
        acc_mps2=[0.0, -1.0],
        time_s=[0.0, 433.8],
        start_position_m=135.0,
        target_position_m=29270.046,
    )
    output_path = tmp_path / "aligned_real_operation_curve.npz"

    saved_path = save_transformed_operation_curve(output_path, result)

    assert saved_path == output_path
    with np.load(output_path, allow_pickle=False) as data:
        assert {
            "position_m",
            "speed_mps",
            "acc_mps2",
            "time_s",
            "source_position_m",
            "source_time_s",
            "source_acc_mps2",
            "start_position_m",
            "target_position_m",
            "source_start_position_m",
            "source_target_position_m",
            "position_scale",
        }.issubset(data.files)
        np.testing.assert_allclose(data["position_m"], [135.0, 29270.046])
        np.testing.assert_allclose(data["speed_mps"], [10.0, 10.0])
        np.testing.assert_allclose(data["acc_mps2"], [0.0, 0.0])
        np.testing.assert_allclose(data["time_s"], [0.0, (29270.046 - 135.0) / 10.0])


def test_recompute_time_and_acceleration_uses_speed_and_scaled_position() -> None:
    time_s, acc_mps2 = recompute_time_and_acceleration(
        position_m=np.asarray([0.0, 10.0, 20.0]),
        speed_mps=np.asarray([10.0, 10.0, 20.0]),
        source_time_s=np.asarray([0.0, 1.0, 2.0]),
        position_scale=1.0,
    )

    np.testing.assert_allclose(time_s, [0.0, 1.0, 5.0 / 3.0])
    np.testing.assert_allclose(acc_mps2, [0.0, 0.0, 15.0])


def test_recompute_time_and_acceleration_falls_back_for_duplicate_positions() -> None:
    time_s, acc_mps2 = recompute_time_and_acceleration(
        position_m=np.asarray([0.0, 0.0, 20.0]),
        speed_mps=np.asarray([0.0, 1.0, 2.0]),
        source_time_s=np.asarray([0.0, 0.5, 1.0]),
        position_scale=2.0,
    )

    assert time_s[1] == pytest.approx(1.0)
    assert time_s[2] > time_s[1]
    assert np.all(np.isfinite(acc_mps2))


def test_transform_operation_curve_rejects_invalid_spans() -> None:
    with pytest.raises(ValueError, match="source position span must be positive"):
        _ = transform_operation_curve_arrays(
            source_position_m=[900.0, 900.0],
            speed_kmh=[0.0, 36.0],
            acc_mps2=[0.0, 1.0],
            time_s=[0.0, 1.0],
            start_position_m=135.0,
            target_position_m=29270.046,
        )

    with pytest.raises(
        ValueError, match="target_position_m must be greater than start_position_m"
    ):
        _ = transform_operation_curve_arrays(
            source_position_m=[900.0, 29240.0],
            speed_kmh=[0.0, 36.0],
            acc_mps2=[0.0, 1.0],
            time_s=[0.0, 1.0],
            start_position_m=135.0,
            target_position_m=135.0,
        )


def test_transform_operation_curve_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        _ = transform_operation_curve_arrays(
            source_position_m=[900.0, 29240.0],
            speed_kmh=[0.0],
            acc_mps2=[0.0, 1.0],
            time_s=[0.0, 1.0],
            start_position_m=135.0,
            target_position_m=29270.046,
        )
