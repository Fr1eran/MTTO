import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import utils.io_utils as io_utils
from dp.core import DP_UPPER_SPEED_ENVELOPE_VERSION, VariableSpacingDPOptimizer
from dp.experiment_utils import compute_dp_reference_curve, load_dp_curve_artifact
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.io_utils import load_curve_with_cum_time_and_metrics, save_curve_and_metrics
from utils.trajectory import OptimizedCurveArtifact


def test_dp_optimizer_constructor_does_not_accept_max_speed() -> None:
    parameters = inspect.signature(VariableSpacingDPOptimizer).parameters

    assert "max_speed" not in parameters


def test_dp_inner_result_includes_cumulative_time_from_policy() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    cache = {
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
        "total_valid_edges": 2,
    }

    result = optimizer._solve_dp_inner(
        cache=cache,
        lambda_time=10.0,
        start_state_idx=0,
        target_state_idx=0,
    )

    assert result is not None
    np.testing.assert_allclose(result["cum_time_s"], np.asarray([0.0, 2.0, 5.0]))
    assert result["cum_time_s"][0] == pytest.approx(0.0)
    assert result["cum_time_s"][-1] == pytest.approx(result["total_time"])
    assert np.all(np.diff(result["cum_time_s"]) >= 0.0)


def test_stage_speed_upper_indices_include_task_upper_curve() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.speed_grid_upper_mps = 20.0
    optimizer.vehicle = cast(
        VehicleInfo, cast(object, SimpleNamespace(max_speed=20.0))
    )
    optimizer.safeguard_utility = cast(
        SafeGuardUtility,
        cast(
            object,
            SimpleNamespace(
                speed_limits=np.asarray([20.0], dtype=np.float64),
                speed_limit_intervals=np.asarray([0.0, 10.0], dtype=np.float64),
                gamma=1.0,
            ),
        ),
    )
    optimizer.upper_curve_pos = np.asarray([0.0, 5.0, 10.0], dtype=np.float64)
    optimizer.upper_curve_speed = np.asarray([0.0, 4.4, 10.0], dtype=np.float64)

    upper = optimizer._get_stage_speed_upper_indices(
        np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
        np.arange(21.0, dtype=np.float64),
    )

    np.testing.assert_array_equal(upper, np.asarray([0, 4, 10]))


def test_dp_upper_curve_uses_operational_stepper_task_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def _fake_min_operation_time_curve(**kwargs: object):
        observed.update(kwargs)
        return (
            np.asarray([135.0, 10000.0, 29276.0], dtype=np.float64),
            np.asarray([0.0, 35.0, 0.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        "dp.core.min_operation_time_curve", _fake_min_operation_time_curve
    )
    vehicle = cast(VehicleInfo, cast(object, SimpleNamespace(max_speed=40.0)))
    track = cast(TrackInfo, object())
    safeguard = cast(
        SafeGuardUtility,
        cast(
            object,
            SimpleNamespace(
                gamma=0.99,
                speed_limits=np.asarray([30.0], dtype=np.float64),
            ),
        ),
    )
    service = TrainService(
        start_position=135.0,
        target_position=29270.0,
        schedule_time=465.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
    )

    optimizer = VariableSpacingDPOptimizer(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard,
        train_service=service,
        precompute_mode="serial",
    )

    assert observed["vehicle"] is vehicle
    assert observed["track"] is track
    assert observed["factor"] == pytest.approx(0.99)
    assert observed["begin_pos"] == pytest.approx(135.0)
    assert observed["begin_speed"] == pytest.approx(0.0)
    assert observed["end_pos"] == pytest.approx(29276.0)
    assert observed["end_speed"] == pytest.approx(0.0)
    assert optimizer.speed_grid_upper_mps == pytest.approx(29.7)


def test_dp_optimize_uses_train_service_absolute_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TrainService(
        start_position=0.0,
        target_position=20.0,
        schedule_time=20.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
        max_arr_time_error_s=10.0,
    )
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.train_service = service
    optimizer.speed_grid_upper_mps = 30.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 2

    calls = 0

    def _fake_prepare(
        self: VariableSpacingDPOptimizer,
        *,
        start_position: float,
        target_position: float,
    ) -> dict[str, object]:
        del self, start_position, target_position
        return {
            "stages": np.asarray([0.0, 20.0], dtype=np.float64),
            "speed_states": np.arange(31.0, dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([30, 30], dtype=int),
            "transitions": [[]],
            "total_valid_edges": 0,
        }

    def _fake_solve_dp_inner(
        self: VariableSpacingDPOptimizer,
        *,
        cache: dict[str, object],
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> dict[str, object]:
        del self, cache, lambda_time, start_state_idx, target_state_idx
        nonlocal calls
        calls += 1
        return {
            "pos": [0.0, 20.0],
            "speed": [1.5, 0.0],
            "cum_time_s": [0.0, 19.0],
            "total_time": 19.0,
            "total_energy": 12.0,
        }

    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_solve_dp_inner",
        _fake_solve_dp_inner,
    )
    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_prepare_transition_graph_cache",
        _fake_prepare,
    )

    result = optimizer.optimize(
        start_pos=0.0,
        start_speed=0.0,
        target_pos=20.0,
        target_speed=0.0,
        schedule_time=20.0,
    )

    assert calls == 1
    assert result is not None
    assert result["total_time"] == pytest.approx(19.0)


def test_dp_optimize_expands_lambda_and_returns_closest_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TrainService(
        start_position=0.0,
        target_position=20.0,
        schedule_time=20.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
        max_arr_time_error_s=0.1,
    )
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.train_service = service
    optimizer.speed_grid_upper_mps = 1.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 2

    def _fake_prepare(
        self: VariableSpacingDPOptimizer,
        *,
        start_position: float,
        target_position: float,
    ) -> dict[str, object]:
        del self, start_position, target_position
        return {
            "stages": np.asarray([0.0, 20.0], dtype=np.float64),
            "speed_states": np.asarray([0.0, 1.0], dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([1, 1], dtype=int),
            "transitions": [[]],
            "total_valid_edges": 0,
        }

    calls: list[float] = []

    def _fake_solve(
        self: VariableSpacingDPOptimizer,
        *,
        cache: dict[str, object],
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> dict[str, object]:
        del self, cache, start_state_idx, target_state_idx
        calls.append(lambda_time)
        if lambda_time < 1_000.0:
            total_time = 30.0
        elif lambda_time < 2_000.0:
            total_time = 25.0
        elif lambda_time < 4_000.0:
            total_time = 21.0
        else:
            total_time = 18.0
        return {
            "pos": [0.0, 20.0],
            "speed": [0.0, 0.0],
            "cum_time_s": [0.0, total_time],
            "total_time": total_time,
            "total_energy": 1.0,
        }

    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_prepare_transition_graph_cache",
        _fake_prepare,
    )
    monkeypatch.setattr(VariableSpacingDPOptimizer, "_solve_dp_inner", _fake_solve)

    result = optimizer.optimize(
        start_pos=0.0,
        start_speed=0.0,
        target_pos=20.0,
        target_speed=0.0,
        schedule_time=20.0,
    )

    assert result is not None
    assert result["total_time"] == pytest.approx(21.0)
    assert calls[:4] == pytest.approx([0.0, 1_000.0, 2_000.0, 4_000.0])
    assert len(calls) > 4


def test_dp_optimize_passes_nonzero_endpoint_states_to_inner_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TrainService(
        start_position=0.0,
        target_position=20.0,
        schedule_time=10.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
    )
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.train_service = service
    optimizer.speed_grid_upper_mps = 2.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 1

    def _fake_prepare(
        self: VariableSpacingDPOptimizer,
        *,
        start_position: float,
        target_position: float,
    ) -> dict[str, object]:
        del self, start_position, target_position
        return {
            "stages": np.asarray([0.0, 20.0], dtype=np.float64),
            "speed_states": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            "stage_speed_upper_idx": np.asarray([2, 2], dtype=int),
            "transitions": [[]],
            "total_valid_edges": 0,
        }

    observed_indices: list[tuple[int, int]] = []

    def _fake_solve(
        self: VariableSpacingDPOptimizer,
        *,
        cache: dict[str, object],
        lambda_time: float,
        start_state_idx: int,
        target_state_idx: int,
    ) -> dict[str, object]:
        del self, cache, lambda_time
        observed_indices.append((start_state_idx, target_state_idx))
        return {
            "pos": [0.0, 20.0],
            "speed": [1.0, 2.0],
            "cum_time_s": [0.0, 10.0],
            "total_time": 10.0,
            "total_energy": 1.0,
        }

    monkeypatch.setattr(
        VariableSpacingDPOptimizer,
        "_prepare_transition_graph_cache",
        _fake_prepare,
    )
    monkeypatch.setattr(VariableSpacingDPOptimizer, "_solve_dp_inner", _fake_solve)

    result = optimizer.optimize(
        start_pos=0.0,
        start_speed=1.0,
        target_pos=20.0,
        target_speed=2.0,
        schedule_time=10.0,
    )

    assert result is not None
    assert observed_indices == [(1, 2)]


def test_dp_optimize_rejects_endpoint_speed_off_grid() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.train_service = TrainService(
        start_position=0.0,
        target_position=20.0,
        schedule_time=10.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
    )
    optimizer.speed_grid_upper_mps = 2.0
    optimizer.delta_speed = 1.0
    optimizer.max_outer_iterations = 1

    with pytest.raises(ValueError, match="not representable"):
        _ = optimizer.optimize(
            start_pos=0.0,
            start_speed=0.5,
            target_pos=20.0,
            target_speed=0.0,
            schedule_time=10.0,
        )


def test_uniform_stage_generation_handles_reverse_direction() -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer.uniform_step_size = 30.0

    stages = optimizer._generate_uniform_spacing_stages(100.0, 0.0)

    assert len(stages) == 5
    assert stages[0] == pytest.approx(100.0)
    assert stages[-1] == pytest.approx(0.0)
    assert np.all(np.diff(stages) < 0.0)


def test_transition_graph_validation_rejects_malformed_payload() -> None:
    stages = np.asarray([0.0, 10.0], dtype=np.float64)
    speed_states = np.asarray([0.0, 1.0], dtype=np.float64)
    upper_idx = np.asarray([1, 1], dtype=int)
    graph = {
        "stages": stages,
        "speed_states": speed_states,
        "stage_speed_upper_idx": upper_idx,
        "transitions": [
            [
                (
                    np.asarray([1], dtype=int),
                    np.asarray([1.0], dtype=np.float64),
                    np.asarray([2.0], dtype=np.float64),
                ),
                None,
            ]
        ],
        "total_valid_edges": 1,
    }

    valid, reason = VariableSpacingDPOptimizer._validate_transition_graph(
        graph,
        expected_stages=stages,
        expected_speed_states=speed_states,
        expected_stage_speed_upper_idx=upper_idx,
    )
    assert valid is True
    assert reason == ""

    graph["transitions"][0][0][2][0] = 0.0
    valid, reason = VariableSpacingDPOptimizer._validate_transition_graph(
        graph,
        expected_stages=stages,
        expected_speed_states=speed_states,
        expected_stage_speed_upper_idx=upper_idx,
    )
    assert valid is False
    assert "time" in reason


def test_transition_graph_cache_requires_current_schema_and_structure(
    tmp_path: Path,
) -> None:
    optimizer = VariableSpacingDPOptimizer.__new__(VariableSpacingDPOptimizer)
    optimizer._CACHE_BASE_DIR = str(tmp_path)
    optimizer.stage_division = "uniform"
    optimizer.sub_stage_count = 1
    optimizer.uniform_step_size = 10.0
    optimizer.speed_grid_upper_mps = 1.0
    optimizer.delta_speed = 0.5

    stages = np.asarray([0.0, 10.0], dtype=np.float64)
    speed_states = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    upper_idx = np.asarray([2, 2], dtype=int)
    graph = {
        "stages": stages,
        "speed_states": speed_states,
        "stage_speed_upper_idx": upper_idx,
        "transitions": [
            [
                (
                    np.asarray([1], dtype=int),
                    np.asarray([1.0], dtype=np.float64),
                    np.asarray([2.0], dtype=np.float64),
                ),
                None,
                None,
            ]
        ],
        "total_valid_edges": 1,
    }
    content_hash = "a" * 64
    optimizer._save_transition_graph_to_disk(
        graph_cache=graph,
        start_position=0.0,
        target_position=10.0,
        content_hash=content_hash,
    )

    loaded = optimizer._load_transition_graph_from_disk(
        content_hash=content_hash,
        expected_stages=stages,
        expected_speed_states=speed_states,
        expected_stage_speed_upper_idx=upper_idx,
        start_position=0.0,
        target_position=10.0,
    )
    assert loaded is not None

    metadata_path = optimizer._get_disk_cache_dir(content_hash) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cache_schema_version"] = -1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert (
        optimizer._load_transition_graph_from_disk(
            content_hash=content_hash,
            expected_stages=stages,
            expected_speed_states=speed_states,
            expected_stage_speed_upper_idx=upper_idx,
            start_position=0.0,
            target_position=10.0,
        )
        is None
    )


def test_save_curve_and_metrics_writes_extra_cumulative_time_array(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "optimized_speed_curve.npz"
    cum_time = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)

    _ = save_curve_and_metrics(
        pos_arr=[0.0, 1.0, 3.0],
        speed_arr=[1.0, 1.0, 1.0],
        output_path=str(curve_path),
        extra_arrays={"cum_time_s": cum_time},
        metrics={"total_time_s": 3.0},
    )

    with np.load(curve_path, allow_pickle=False) as npz_data:
        assert "cum_time_s" in npz_data.files
        assert "allow_pickle" not in npz_data.files
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


def test_load_curve_with_cum_time_reads_npz_once_when_time_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve_path = tmp_path / "optimized_speed_curve.npz"
    _ = save_curve_and_metrics(
        pos_arr=[0.0, 1.0, 3.0],
        speed_arr=[1.0, 1.0, 1.0],
        output_path=str(curve_path),
        extra_arrays={"cum_time_s": [0.0, 1.0, 3.0]},
        metrics={"total_time_s": 3.0},
    )

    real_load = io_utils.np.load
    load_count = 0

    def counting_load(*args: object, **kwargs: object):
        nonlocal load_count
        load_count += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(io_utils.np, "load", counting_load)
    pos_arr, speed_arr, cum_time_arr, metrics = load_curve_with_cum_time_and_metrics(
        str(curve_path)
    )

    assert load_count == 1
    np.testing.assert_allclose(pos_arr, [0.0, 1.0, 3.0])
    np.testing.assert_allclose(speed_arr, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cum_time_arr, [0.0, 1.0, 3.0])
    assert metrics["total_time_s"] == pytest.approx(3.0)


def test_load_curve_with_cum_time_reads_legacy_npz_once_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve_path = tmp_path / "legacy_curve.npz"
    metrics_path = tmp_path / "legacy_curve_metrics.json"
    np.savez_compressed(
        curve_path,
        pos_m=np.asarray([0.0, 1.0, 3.0], dtype=np.float32),
        speed_mps=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )
    _ = metrics_path.write_text(json.dumps({"total_time_s": 3.0}), encoding="utf-8")

    real_load = io_utils.np.load
    load_count = 0

    def counting_load(*args: object, **kwargs: object):
        nonlocal load_count
        load_count += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(io_utils.np, "load", counting_load)
    _pos_arr, _speed_arr, cum_time_arr, metrics = load_curve_with_cum_time_and_metrics(
        str(curve_path),
        str(metrics_path),
    )

    assert load_count == 1
    np.testing.assert_allclose(cum_time_arr, [0.0, 1.0, 3.0])
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
    _ = metrics_path.write_text(json.dumps({"total_time_s": 3.0}), encoding="utf-8")

    artifact = OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(metrics_path),
    )

    _pos_arr, _speed_arr, cum_time_arr, metrics = load_dp_curve_artifact(artifact)

    np.testing.assert_allclose(cum_time_arr, np.asarray([0.0, 1.0, 3.0]))
    assert metrics["total_time_s"] == pytest.approx(3.0)


def test_compute_dp_reference_curve_saves_and_returns_cumulative_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOptimizer:
        def __init__(self, **kwargs: object):
            self.kwargs: dict[str, object] = kwargs

        def optimize(
            self,
            *,
            start_pos: float,
            start_speed: float,
            target_pos: float,
            target_speed: float,
            schedule_time: float,
        ):
            del start_pos, start_speed, target_pos, target_speed, schedule_time
            return {
                "pos": [0.0, 20.0],
                "speed": [1.5, 0.0],
                "cum_time_s": [0.0, 9.0],
                "total_time": 9.0,
                "total_energy": 12.0,
            }

    monkeypatch.setattr("dp.experiment_utils.VariableSpacingDPOptimizer", FakeOptimizer)

    def _fake_compute_comfort_metrics(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        "dp.experiment_utils.compute_comfort_metrics_from_trajectory",
        _fake_compute_comfort_metrics,
    )
    train_service = TrainService(
        start_position=0.0,
        target_position=20.0,
        schedule_time=10.0,
        max_acc_change=0.75,
        max_stop_error=0.3,
        max_arr_time_error_s=10.0,
    )

    pos_arr, speed_arr, cum_time_arr, metrics = compute_dp_reference_curve(
        vehicle=cast(VehicleInfo, cast(object, SimpleNamespace(max_speed=30.0))),
        track=cast(TrackInfo, object()),
        safeguard_utility=cast(SafeGuardUtility, object()),
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
    assert metrics["dp_upper_speed_envelope_version"] == (
        DP_UPPER_SPEED_ENVELOPE_VERSION
    )
    with np.load(tmp_path / "optimized_speed_curve.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["cum_time_s"], np.asarray([0.0, 9.0]))
