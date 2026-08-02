import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model.common import calc_transition_from_acc_scalar_numba
from model.ocs import TrainService
from model.ocs.stopping_points_stepping import SPSState
from rl.dp_trajectory_reader import DPTrajectoryReader
from rl.operational_state import OperationalState, OperationalTransition, ViolationCode
from rl.reference_trajectory_sampler import ReferenceTrajectorySampler
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
    metrics_updates: dict[str, object] | None = None,
) -> OptimizedCurveArtifact:
    directory.mkdir(parents=True, exist_ok=True)
    curve_path = directory / "optimized_speed_curve.npz"
    metrics: dict[str, object] = {
        "target_time_s": 10.0,
        "start_position_m": 0.0,
        "start_speed_mps": 0.0,
        "target_position_m": 20.0,
        "target_speed_mps": 0.0,
    }
    if metrics_updates is not None:
        metrics.update(metrics_updates)
    save_curve_and_metrics(
        pos_arr=[0.0, 10.0, 20.0],
        speed_arr=[0.0, 10.0, 0.0],
        output_path=str(curve_path),
        extra_arrays={"cum_time_s": [0.0, 2.0, 4.0]},
        metrics=metrics,
    )
    return OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(curve_path.with_name("optimized_speed_curve_metrics.json")),
    )


def test_dp_adapter_loads_task_matching_reference_without_grid_metadata(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    trajectory = DPTrajectoryReader.from_artifact(
        artifact=_write_artifact(
            tmp_path / "run",
            metrics_updates={"stage_division": "variable", "max_step_distance_m": None},
        ),
        train_service=train_service,
    )

    np.testing.assert_allclose(trajectory.position_m, [0.0, 10.0, 20.0])
    assert trajectory.metadata["stage_division"] == "variable"


def test_dp_adapter_selects_newest_matching_artifact(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    old_artifact = _write_artifact(tmp_path / "old")
    selected_artifact = _write_artifact(tmp_path / "selected")
    _write_artifact(tmp_path / "mismatch", metrics_updates={"target_time_s": 11.0})
    os.utime(old_artifact.npz_path, (1, 1))
    os.utime(selected_artifact.npz_path, (2, 2))

    trajectory = DPTrajectoryReader.from_curve_dir(
        curve_dir=tmp_path,
        train_service=train_service,
    )

    assert trajectory.metadata["target_time_s"] == pytest.approx(10.0)
    np.testing.assert_allclose(trajectory.position_m, [0.0, 10.0, 20.0])


class _FakeStepper:
    def __init__(
        self, *, max_step_distance_m: float = 5.0, truncate_step: int | None = None
    ):
        self.train_service = TrainService(
            start_position=0.0,
            start_speed=0.0,
            target_position=9.0,
            schedule_time=8.0,
            max_acc_change=0.75,
            max_arr_time_error_ratio=0.1,
            max_stop_error=1.0,
        )
        self.direction = 1
        self.whole_distance_m = 9.0
        self.max_step_distance_m = max_step_distance_m
        self.vehicle = SimpleNamespace(max_dec=-2.5, max_acc=2.5)
        self._truncate_step = truncate_step
        self.requested_distances: list[float] = []

    def reset(self) -> OperationalState:
        return OperationalState(
            position_m=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            operation_time_s=0.0,
            redundant_operation_time_s=0.0,
            energy_consumption_kj=0.0,
            slope_permille=0.0,
            min_speed_mps=0.0,
            max_speed_mps=100.0,
            stop_error_m=9.0,
            sps_state=SPSState(),
            step_count=0,
        )

    def advance(
        self,
        state: OperationalState,
        acceleration_mps2: float,
        *,
        requested_distance_m: float | None = None,
    ) -> OperationalTransition:
        assert requested_distance_m is not None
        self.requested_distances.append(requested_distance_m)
        next_speed, distance, duration = calc_transition_from_acc_scalar_numba(
            state.speed_mps, acceleration_mps2, requested_distance_m
        )
        position = state.position_m + distance
        step_count = state.step_count + 1
        terminated = bool(
            np.isclose(position, self.train_service.target_position)
            and np.isclose(next_speed, 0.0)
        )
        truncated = step_count == self._truncate_step
        next_state = replace(
            state,
            position_m=position,
            speed_mps=next_speed,
            acceleration_mps2=acceleration_mps2,
            operation_time_s=state.operation_time_s + duration,
            energy_consumption_kj=state.energy_consumption_kj + distance,
            stop_error_m=abs(self.train_service.target_position - position),
            sps_state=SPSState(target_stopping_point_index=step_count),
            step_count=step_count,
        )
        return OperationalTransition(
            previous_state=state,
            next_state=next_state,
            acceleration_mps2=acceleration_mps2,
            distance_m=distance,
            duration_s=duration,
            energy_delta_kj=distance,
            terminated=terminated,
            truncated=truncated,
            violation_code=(
                ViolationCode.SPEED_HIGH if truncated else ViolationCode.ONGOING
            ),
        )


def _build_reference_sampler(
    *,
    stepper: _FakeStepper | None = None,
) -> tuple[ReferenceTrajectorySampler, _FakeStepper]:
    resolved_stepper = stepper or _FakeStepper()
    sampler = ReferenceTrajectorySampler.from_arrays(
        position_m=[0.0, 4.0, 9.0],
        speed_mps=[0.0, 4.0, 0.0],
        cumulative_time_s=[0.0, 2.0, 4.5],
        stepper=resolved_stepper,  # type: ignore[arg-type]
    )
    return sampler, resolved_stepper


def test_reference_sampler_resamples_and_replays_complete_runtime_states() -> None:
    sampler, stepper = _build_reference_sampler()

    assert sampler.node_count == 3
    assert sampler.eligible_node_count == 2
    np.testing.assert_allclose(stepper.requested_distances, [5.0, 4.0])
    middle = sampler.state_at(1)
    assert middle.position_m == pytest.approx(5.0)
    assert middle.runtime_state.step_count == 1
    assert middle.runtime_state.sps_state.target_stopping_point_index == 1
    assert middle.runtime_state.energy_consumption_kj == pytest.approx(5.0)
    assert sampler.state_at(2).runtime_state.step_count == 2


def test_reference_sampler_weighted_sampling_excludes_terminal_node() -> None:
    sampler, _ = _build_reference_sampler()

    sampled = sampler.sample(
        np.random.default_rng(42), weights=np.asarray([0.0, 1.0, 100.0])
    )

    assert sampled.reference_index == 1
    with pytest.raises(ValueError, match="non-terminal"):
        sampler.sample(np.random.default_rng(1), index_range=(2, 2))


def test_reference_sampler_rejects_inconsistent_source_time() -> None:
    with pytest.raises(ValueError, match="cumulative time"):
        ReferenceTrajectorySampler.from_arrays(
            position_m=[0.0, 4.0, 9.0],
            speed_mps=[0.0, 4.0, 0.0],
            cumulative_time_s=[0.0, 3.0, 4.5],
            stepper=_FakeStepper(),  # type: ignore[arg-type]
        )


def test_reference_sampler_rejects_early_truncation() -> None:
    with pytest.raises(ValueError, match="becomes done"):
        _build_reference_sampler(stepper=_FakeStepper(truncate_step=1))
