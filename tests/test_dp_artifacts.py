import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dp.core import VariableSpacingDPOptimizer
from dp.experiment_utils import compute_dp_reference_curve, load_dp_curve_artifact
from model.ocs import TrainService
from utils.io_utils import save_curve_and_metrics
from utils.trajectory import OptimizedCurveArtifact


def test_dp_inner_result_includes_cumulative_time_from_policy() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)

    def _prepare_transition_graph_cache(*, max_speed: float, delta_speed: float):
        del max_speed, delta_speed
        return {
            "stages": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "speed_states": np.asarray([0.0, 1.0], dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([0, 1, 0], dtype=int),
            "transitions": [
                [
                    (
                        np.asarray([1], dtype=int),
                        np.asarray([5.0], dtype=np.float64),
                        np.asarray([2.0], dtype=np.float64),
                    ),
                    None,
                ],
                [
                    None,
                    (
                        np.asarray([0], dtype=int),
                        np.asarray([7.0], dtype=np.float64),
                        np.asarray([3.0], dtype=np.float64),
                    ),
                ],
            ],
        }

    optimizer._prepare_transition_graph_cache = _prepare_transition_graph_cache

    result = optimizer._solve_dp_inner(
        lambda_time=10.0,
        max_speed=1.0,
        delta_speed=1.0,
    )

    assert result is not None
    np.testing.assert_allclose(result["cum_time_s"], np.asarray([0.0, 2.0, 5.0]))
    assert result["cum_time_s"][0] == pytest.approx(0.0)
    assert result["cum_time_s"][-1] == pytest.approx(result["total_time"])
    assert np.all(np.diff(result["cum_time_s"]) >= 0.0)


def test_save_curve_and_metrics_writes_extra_cumulative_time_array(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "optimized_speed_curve.npz"
    cum_time = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)

    save_curve_and_metrics(
        pos_arr=[0.0, 1.0, 3.0],
        speed_arr=[1.0, 1.0, 1.0],
        output_path=str(curve_path),
        extra_arrays={"cum_time_s": cum_time},
        metrics={"total_time_s": 3.0},
    )

    with np.load(curve_path, allow_pickle=False) as npz_data:
        assert "cum_time_s" in npz_data.files
        np.testing.assert_allclose(npz_data["cum_time_s"], cum_time)

    artifact = OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(curve_path.with_name("optimized_speed_curve_metrics.json")),
    )
    pos_arr, speed_arr, cum_time_arr, metrics = load_dp_curve_artifact(artifact)

    np.testing.assert_allclose(pos_arr, np.asarray([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(speed_arr, np.asarray([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(cum_time_arr, cum_time)
    assert metrics["total_time_s"] == pytest.approx(3.0)


def test_load_dp_curve_artifact_recovers_cumulative_time_for_legacy_npz(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "optimized_speed_curve.npz"
    metrics_path = tmp_path / "optimized_speed_curve_metrics.json"
    np.savez_compressed(
        curve_path,
        pos_m=np.asarray([0.0, 1.0, 3.0], dtype=np.float32),
        speed_mps=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )
    metrics_path.write_text(json.dumps({"total_time_s": 3.0}), encoding="utf-8")

    artifact = OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(metrics_path),
    )

    _pos_arr, _speed_arr, cum_time_arr, metrics = load_dp_curve_artifact(artifact)

    np.testing.assert_allclose(cum_time_arr, np.asarray([0.0, 1.0, 3.0]))
    assert metrics["total_time_s"] == pytest.approx(3.0)


def test_compute_dp_reference_curve_saves_and_returns_cumulative_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeOptimizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def optimize(self, *, max_speed: float, delta_speed: float, max_iters: int):
            del max_speed, delta_speed, max_iters
            return {
                "pos": [0.0, 20.0],
                "speed": [1.5, 0.0],
                "cum_time_s": [0.0, 9.0],
                "total_time": 9.0,
                "total_energy": 12.0,
            }

    monkeypatch.setattr("dp.experiment_utils.VariableSpacingDPOptimizer", FakeOptimizer)
    monkeypatch.setattr(
        "dp.experiment_utils.compute_comfort_metrics_from_trajectory",
        lambda **kwargs: {},
    )
    train_service = TrainService(
        start_position=0.0,
        start_speed=0.0,
        target_position=20.0,
        schedule_time=10.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=0.01,
        max_stop_error=0.3,
    )

    pos_arr, speed_arr, cum_time_arr, metrics = compute_dp_reference_curve(
        vehicle=SimpleNamespace(max_speed=30.0),
        track=object(),
        safeguard_utility=object(),
        train_service=train_service,
        output_dir=tmp_path,
        start_position=0.0,
        start_speed=1.5,
        target_position=20.0,
        schedule_time_s=10.0,
        target_speed=0.0,
        precompute_mode="serial",
        precompute_workers=1,
        max_outer_iterations=1,
    )

    np.testing.assert_allclose(pos_arr, np.asarray([0.0, 20.0]))
    np.testing.assert_allclose(speed_arr, np.asarray([1.5, 0.0]))
    np.testing.assert_allclose(cum_time_arr, np.asarray([0.0, 9.0]))
    assert metrics["start_speed_mps"] == pytest.approx(1.5)
    assert metrics["target_speed_mps"] == pytest.approx(0.0)
    assert metrics["max_step_distance_m"] == pytest.approx(30.0)
    assert metrics["stage_division"] == "uniform"
    with np.load(tmp_path / "optimized_speed_curve.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["cum_time_s"], np.asarray([0.0, 9.0]))
