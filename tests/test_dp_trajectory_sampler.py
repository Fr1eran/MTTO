import os
from pathlib import Path

import numpy as np
import pytest

from model.ocs import TrainService
from rl.dp_trajectory_sampler import DPTrajectorySampler
from utils.io_utils import save_curve_and_metrics
from utils.trajectory import OptimizedCurveArtifact


@pytest.fixture
def train_service() -> TrainService:
    return TrainService(
        start_position=0.0,
        start_speed=0.0,
        target_position=20.0,
        schedule_time=10.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=0.1,
        max_stop_error=1.0,
    )


def _write_artifact(
    directory: Path,
    *,
    pos: list[float] | None = None,
    speed: list[float] | None = None,
    cum_time: list[float] | None = None,
    metrics_updates: dict[str, object] | None = None,
) -> OptimizedCurveArtifact:
    directory.mkdir(parents=True, exist_ok=True)
    curve_path = directory / "optimized_speed_curve.npz"
    metrics = {
        "target_time_s": 10.0,
        "start_position_m": 0.0,
        "start_speed_mps": 0.0,
        "target_position_m": 20.0,
        "target_speed_mps": 0.0,
        "max_step_distance_m": 30.0,
        "stage_division": "uniform",
    }
    if metrics_updates is not None:
        metrics.update(metrics_updates)
    save_curve_and_metrics(
        pos_arr=[0.0, 10.0, 20.0] if pos is None else pos,
        speed_arr=[0.0, 10.0, 0.0] if speed is None else speed,
        output_path=str(curve_path),
        extra_arrays={"cum_time_s": [0.0, 2.0, 5.0] if cum_time is None else cum_time},
        metrics=metrics,
    )
    return OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(curve_path.with_name("optimized_speed_curve_metrics.json")),
    )


def test_from_artifact_returns_discrete_state_and_segment_acceleration(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    artifact = _write_artifact(tmp_path / "run")

    sampler = DPTrajectorySampler.from_artifact(
        artifact=artifact,
        train_service=train_service,
        max_step_distance_m=30.0,
    )

    assert sampler.node_count == 3
    assert sampler.state_at(0).acceleration_mps2 == pytest.approx(0.0)
    state = sampler.state_at(1)
    assert state.reference_index == 1
    assert state.position_m == pytest.approx(10.0)
    assert state.speed_mps == pytest.approx(10.0)
    assert state.operation_time_s == pytest.approx(2.0)
    assert sampler.state_at(2).acceleration_mps2 == pytest.approx(-5.0)


def test_from_curve_dir_selects_latest_matching_artifact(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    old_artifact = _write_artifact(tmp_path / "old")
    selected_artifact = _write_artifact(tmp_path / "selected")
    _write_artifact(
        tmp_path / "mismatch",
        metrics_updates={"target_time_s": 11.0},
    )
    incompatible_artifact = _write_artifact(
        tmp_path / "incompatible",
        metrics_updates={"stage_division": "variable", "max_step_distance_m": None},
    )
    os.utime(old_artifact.npz_path, (1, 1))
    os.utime(selected_artifact.npz_path, (2, 2))
    os.utime(incompatible_artifact.npz_path, (3, 3))

    sampler = DPTrajectorySampler.from_curve_dir(
        curve_dir=tmp_path,
        train_service=train_service,
        max_step_distance_m=30.0,
    )

    assert sampler.artifact.npz_path == selected_artifact.npz_path
    assert sampler.max_step_distance_m == pytest.approx(30.0)


def test_sample_supports_inclusive_index_and_remaining_distance_ranges(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    sampler = DPTrajectorySampler.from_artifact(
        artifact=_write_artifact(tmp_path / "run"),
        train_service=train_service,
        max_step_distance_m=30.0,
    )
    rng = np.random.default_rng(7)

    assert sampler.sample(rng, index_range=(1, 1)).reference_index == 1
    assert (
        sampler.sample(rng, remaining_distance_range_m=(9.0, 11.0)).reference_index
        == 1
    )


def test_sample_rejects_ambiguous_or_empty_ranges(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    sampler = DPTrajectorySampler.from_artifact(
        artifact=_write_artifact(tmp_path / "run"),
        train_service=train_service,
        max_step_distance_m=30.0,
    )
    rng = np.random.default_rng(7)

    with pytest.raises(ValueError, match="at most one"):
        sampler.sample(
            rng,
            index_range=(0, 1),
            remaining_distance_range_m=(0.0, 10.0),
        )
    with pytest.raises(ValueError, match="contains no DP nodes"):
        sampler.sample(rng, remaining_distance_range_m=(1.0, 2.0))
    with pytest.raises(ValueError, match="within"):
        sampler.sample(rng, index_range=(-1, 1))


@pytest.mark.parametrize(
    "metrics_updates",
    [
        {"target_time_s": 9.0},
        {"start_speed_mps": 1.0},
        {"target_speed_mps": 1.0},
        {"start_speed_mps": None},
    ],
)
def test_from_artifact_rejects_incompatible_or_missing_metadata(
    tmp_path: Path,
    train_service: TrainService,
    metrics_updates: dict[str, object],
) -> None:
    artifact = _write_artifact(tmp_path / "run", metrics_updates=metrics_updates)

    with pytest.raises(ValueError, match="metadata does not match"):
        DPTrajectorySampler.from_artifact(
            artifact=artifact,
            train_service=train_service,
            max_step_distance_m=30.0,
        )


@pytest.mark.parametrize(
    ("metrics_updates", "max_step_distance_m", "message"),
    [
        (
            {"stage_division": "variable", "max_step_distance_m": None},
            30.0,
            "stage_division",
        ),
        (
            {"max_step_distance_m": None},
            30.0,
            "max_step_distance_m metadata is missing",
        ),
        ({"max_step_distance_m": 20.0}, 30.0, "does not match the environment"),
    ],
)
def test_from_artifact_rejects_incompatible_discretization_metadata(
    tmp_path: Path,
    train_service: TrainService,
    metrics_updates: dict[str, object],
    max_step_distance_m: float,
    message: str,
) -> None:
    artifact = _write_artifact(tmp_path / "run", metrics_updates=metrics_updates)

    with pytest.raises(ValueError, match=message):
        DPTrajectorySampler.from_artifact(
            artifact=artifact,
            train_service=train_service,
            max_step_distance_m=max_step_distance_m,
        )


@pytest.mark.parametrize(
    ("pos", "speed", "cum_time", "message"),
    [
        (
            [0.0, 10.0, 10.0, 20.0],
            [0.0, 10.0, 10.0, 0.0],
            [0.0, 2.0, 3.0, 5.0],
            "monotonic",
        ),
        ([0.0, 10.0, 20.0], [0.0, -1.0, 0.0], [0.0, 2.0, 5.0], "non-negative"),
        ([0.0, 10.0, 20.0], [0.0, 10.0, 0.0], [0.0, 2.0, 2.0], "increasing"),
    ],
)
def test_from_artifact_rejects_invalid_arrays(
    tmp_path: Path,
    train_service: TrainService,
    pos: list[float],
    speed: list[float],
    cum_time: list[float],
    message: str,
) -> None:
    artifact = _write_artifact(
        tmp_path / "run",
        pos=pos,
        speed=speed,
        cum_time=cum_time,
    )

    with pytest.raises(ValueError, match=message):
        DPTrajectorySampler.from_artifact(
            artifact=artifact,
            train_service=train_service,
            max_step_distance_m=30.0,
        )
