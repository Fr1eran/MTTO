from pathlib import Path

import numpy as np
import pytest

from scripts.reproduce_dp import compute_comfort_metrics_from_trajectory
from utils.io_utils import load_optimized_curve_and_metrics, save_curve_and_metrics


def _build_trajectory_from_acc(
    acc_seq: list[float],
    *,
    v0: float = 1.0,
    ds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 1-D trajectory whose segment accelerations are acc_seq."""
    speed_sq = float(v0**2)
    speed = [float(v0)]
    for acc in acc_seq:
        speed_sq = speed_sq + 2.0 * float(acc) * ds
        speed.append(float(np.sqrt(max(speed_sq, 0.0))))

    pos = np.arange(len(speed), dtype=np.float64) * ds
    return pos, np.asarray(speed, dtype=np.float64)


def test_compute_comfort_metrics_matches_mtto_env_definition() -> None:
    acc_seq = [0.1, 0.5, -0.2]
    pos, speed = _build_trajectory_from_acc(acc_seq)

    metrics = compute_comfort_metrics_from_trajectory(
        pos_arr=pos,
        speed_arr=speed,
        max_acc_change=0.5,
    )

    # delta_acc = [0.1, 0.4, 0.7], with a0 = 0.0
    assert metrics["comfort_tav"] == pytest.approx(1.2)
    assert metrics["comfort_rms"] == pytest.approx(np.sqrt(0.22))
    assert metrics["comfort_er_pct"] == pytest.approx(100.0 / 3.0)


def test_compute_comfort_metrics_uses_strict_greater_than_threshold() -> None:
    # Segment accelerations are [0.5, 0.0], so delta_acc is [0.5, 0.5].
    pos = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    speed = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)

    metrics = compute_comfort_metrics_from_trajectory(
        pos_arr=pos,
        speed_arr=speed,
        max_acc_change=0.5,
    )

    # delta_acc = [0.5, 0.5], and 0.5 is NOT counted because the rule is strictly >
    assert metrics["comfort_er_pct"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "pos_arr,speed_arr",
    [
        ([0.0], [0.0]),
        ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
    ],
)
def test_compute_comfort_metrics_handles_short_or_flat_trajectory(
    pos_arr: list[float],
    speed_arr: list[float],
) -> None:
    metrics = compute_comfort_metrics_from_trajectory(
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        max_acc_change=0.75,
    )

    assert metrics["comfort_tav"] == pytest.approx(0.0)
    assert metrics["comfort_rms"] == pytest.approx(0.0)
    assert metrics["comfort_er_pct"] == pytest.approx(0.0)


def test_comfort_metrics_round_trip_in_metrics_json(tmp_path: Path) -> None:
    output_npz = tmp_path / "optimized_speed_curve.npz"
    comfort_metrics = {
        "comfort_tav": 1.234,
        "comfort_rms": 0.456,
        "comfort_er_pct": 12.5,
    }

    save_curve_and_metrics(
        pos_arr=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        speed_arr=np.asarray([0.0, 1.0, 1.5], dtype=np.float64),
        output_path=str(output_npz),
        metrics=comfort_metrics,
    )

    _, _, loaded_metrics = load_optimized_curve_and_metrics(str(output_npz))

    assert loaded_metrics["comfort_tav"] == pytest.approx(
        comfort_metrics["comfort_tav"]
    )
    assert loaded_metrics["comfort_rms"] == pytest.approx(
        comfort_metrics["comfort_rms"]
    )
    assert loaded_metrics["comfort_er_pct"] == pytest.approx(
        comfort_metrics["comfort_er_pct"]
    )
