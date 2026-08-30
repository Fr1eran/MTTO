import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from dp.core import DP_UPPER_SPEED_ENVELOPE_VERSION
from model.common import calc_transition_from_acc_scalar_numba
from model.ocs import TrainService
from model.ocs.stopping_points_stepping import SPSState
from rl.context_pool import ContextPool, ContextPoolBuilder
from rl.dp_trajectory_reader import DPTrajectoryReader
from rl.operational_state import OperationalState, OperationalTransition, ViolationCode
from rl.operational_stepper import OperationalStepper
from utils.io_utils import save_curve_and_metrics
from utils.trajectory import OptimizedCurveArtifact


@pytest.fixture
def train_service() -> TrainService:
    return TrainService(
        start_position=0.0,
        target_position=20.0,
        schedule_time=10.0,
        max_acc_change=0.75,
        max_stop_error=1.0,
        max_arr_time_error_s=10.0,
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
        "dp_upper_speed_envelope_version": DP_UPPER_SPEED_ENVELOPE_VERSION,
    }
    if metrics_updates is not None:
        metrics.update(metrics_updates)
    _ = save_curve_and_metrics(
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
    _ = _write_artifact(tmp_path / "mismatch", metrics_updates={"target_time_s": 11.0})
    os.utime(old_artifact.npz_path, (1, 1))
    os.utime(selected_artifact.npz_path, (2, 2))

    trajectory = DPTrajectoryReader.from_curve_dir(
        curve_dir=tmp_path,
        train_service=train_service,
    )

    assert trajectory.metadata["target_time_s"] == pytest.approx(10.0)
    np.testing.assert_allclose(trajectory.position_m, [0.0, 10.0, 20.0])


def test_dp_adapter_rejects_legacy_upper_envelope_artifact(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    artifact = _write_artifact(
        tmp_path / "legacy",
        metrics_updates={"dp_upper_speed_envelope_version": 0},
    )

    with pytest.raises(ValueError, match="incompatible upper-speed-envelope"):
        _ = DPTrajectoryReader.from_artifact(
            artifact=artifact,
            train_service=train_service,
        )


class _FakeStepper:
    def __init__(
        self, *, step_distance_m: float = 5.0, truncate_step: int | None = None
    ):
        self.train_service: TrainService = TrainService(
            start_position=0.0,
            target_position=9.0,
            schedule_time=8.0,
            max_acc_change=0.75,
            max_stop_error=1.0,
            max_arr_time_error_s=10.0,
        )
        self.direction: int = 1
        self.whole_distance_m: float = 9.0
        self.step_distance_m: float = step_distance_m
        self.vehicle: object = SimpleNamespace(max_dec=-2.5, max_acc=2.5)
        self._truncate_step: int | None = truncate_step
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


def _build_context_pool(
    *,
    stepper: _FakeStepper | None = None,
) -> tuple[ContextPool, _FakeStepper]:
    resolved_stepper = stepper or _FakeStepper()
    context_pool = ContextPoolBuilder.from_arrays(
        position_m=[0.0, 4.0, 9.0],
        speed_mps=[0.0, 4.0, 0.0],
        cumulative_time_s=[0.0, 2.0, 4.5],
        stepper=cast(OperationalStepper, cast(object, resolved_stepper)),
    )
    return context_pool, resolved_stepper


def test_context_pool_builder_resamples_and_replays_complete_runtime_states() -> None:
    context_pool, stepper = _build_context_pool()

    assert context_pool.context_count == 2
    np.testing.assert_allclose(stepper.requested_distances, [5.0, 4.0])
    middle = context_pool.context_at(1)
    assert middle.initial_state.position_m == pytest.approx(5.0)
    assert middle.initial_state.step_count == 1
    assert middle.initial_state.sps_state.target_stopping_point_index == 1
    assert middle.initial_state.energy_consumption_kj == pytest.approx(5.0)


def test_context_pool_excludes_terminal_node() -> None:
    context_pool, _ = _build_context_pool()
    assert [context.context_index for context in context_pool.contexts] == [0, 1]


def test_reference_sampler_rejects_inconsistent_source_time() -> None:
    with pytest.raises(ValueError, match="cumulative time"):
        _ = ContextPoolBuilder.from_arrays(
            position_m=[0.0, 4.0, 9.0],
            speed_mps=[0.0, 4.0, 0.0],
            cumulative_time_s=[0.0, 3.0, 4.5],
            stepper=cast(OperationalStepper, cast(object, _FakeStepper())),
        )


def test_reference_sampler_rejects_early_truncation() -> None:
    with pytest.raises(ValueError, match="becomes done"):
        _ = _build_context_pool(stepper=_FakeStepper(truncate_step=1))
