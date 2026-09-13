import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from dp.core import DP_UPPER_SPEED_ENVELOPE_VERSION
from model.ocs import TrainService
from model.ocs.stopping_points_stepping import SPSState
from rl.context_pool import ContextPool, ContextPoolBuilder
from rl.dp_trajectory_reader import DPTrajectoryReader
from rl.operational_state import OperationalState
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
    position_m: list[float] | None = None,
    speed_mps: list[float] | None = None,
    cumulative_time_s: list[float] | None = None,
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
        pos_arr=position_m or [0.0, 10.0, 20.0],
        speed_arr=speed_mps or [0.0, 10.0, 0.0],
        output_path=str(curve_path),
        extra_arrays={"cum_time_s": cumulative_time_s or [0.0, 2.0, 4.0]},
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


def test_dp_adapter_keeps_latest_sample_at_consecutive_duplicate_position(
    tmp_path: Path,
    train_service: TrainService,
) -> None:
    trajectory = DPTrajectoryReader.from_artifact(
        artifact=_write_artifact(
            tmp_path / "duplicates",
            position_m=[0.0, 10.0, 10.0, 20.0],
            speed_mps=[0.0, 8.0, 9.0, 0.0],
            cumulative_time_s=[0.0, 2.0, 2.5, 4.0],
        ),
        train_service=train_service,
    )

    np.testing.assert_allclose(trajectory.position_m, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(trajectory.speed_mps, [0.0, 9.0, 0.0])
    np.testing.assert_allclose(trajectory.cumulative_time_s, [0.0, 2.5, 4.0])


class _FakeStepper:
    def __init__(
        self,
        *,
        step_distance_m: float = 5.0,
        target_position_m: float = 9.0,
        schedule_time_s: float = 4.5,
        max_safe_speed_mps: float = 100.0,
    ):
        self.train_service: TrainService = TrainService(
            start_position=0.0,
            target_position=target_position_m,
            schedule_time=schedule_time_s,
            max_acc_change=0.75,
            max_stop_error=1.0,
            max_arr_time_error_s=10.0,
        )
        self.direction: int = 1 if target_position_m > 0.0 else -1
        self.whole_distance_m: float = abs(target_position_m)
        self.step_distance_m: float = step_distance_m
        self.vehicle: object = SimpleNamespace(max_dec=-2.5, max_acc=2.5)
        self.track = object()
        self.sps = _FakeSPS()
        self.ecc = _FakeECC()
        self._max_safe_speed_mps = max_safe_speed_mps

    def build_state(
        self,
        *,
        position_m: float,
        speed_mps: float,
        acceleration_mps2: float,
        operation_time_s: float,
        energy_consumption_kj: float,
        step_count: int,
        sps_state: SPSState,
    ) -> OperationalState:
        return OperationalState(
            position_m=position_m,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            operation_time_s=operation_time_s,
            redundant_operation_time_s=0.0,
            energy_consumption_kj=energy_consumption_kj,
            slope_permille=0.0,
            min_speed_mps=0.0,
            max_speed_mps=self._max_safe_speed_mps,
            stop_error_m=abs(self.train_service.target_position - position_m),
            sps_state=sps_state,
            step_count=step_count,
        )

    def advance(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("context construction must not call stepper.advance()")


class _FakeSPS:
    @staticmethod
    def initial_state() -> SPSState:
        return SPSState()

    @staticmethod
    def advance(
        state: SPSState,
        *,
        position_m: float,
        speed_mps: float,
        time_s: float,
    ) -> SPSState:
        del position_m, speed_mps, time_s
        return SPSState(
            target_stopping_point_index=state.target_stopping_point_index + 1
        )


class _FakeECC:
    @staticmethod
    def calc_energy(**kwargs: object) -> tuple[float, float]:
        return float(kwargs["distance"]), 0.0


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
        context_count=2,
    )
    return context_pool, resolved_stepper


def test_context_pool_builder_uniformly_partitions_route_and_replays_state() -> None:
    context_pool, stepper = _build_context_pool()

    assert context_pool.context_count == 2
    middle = context_pool.context_at(1)
    assert middle.initial_state.position_m == pytest.approx(4.5)
    assert middle.initial_state.speed_mps == pytest.approx(3.6)
    assert middle.initial_state.operation_time_s == pytest.approx(2.25)
    assert middle.initial_state.acceleration_mps2 == pytest.approx(-1.6)
    assert middle.initial_state.step_count == 0
    assert middle.initial_state.sps_state.target_stopping_point_index == 1
    assert middle.initial_state.energy_consumption_kj == pytest.approx(4.5)


def test_context_pool_excludes_terminal_node() -> None:
    context_pool, _ = _build_context_pool()
    assert [context.context_index for context in context_pool.contexts] == [0, 1]


def test_context_pool_positions_do_not_depend_on_rl_step_grid() -> None:
    context_pool = ContextPoolBuilder.from_arrays(
        position_m=[0.0, 5.0, 10.0, 15.0, 20.0],
        speed_mps=[0.0, 5.0, 5.0, 5.0, 0.0],
        cumulative_time_s=[0.0, 2.0, 3.0, 4.0, 6.0],
        stepper=cast(
            OperationalStepper,
            cast(
                object,
                _FakeStepper(target_position_m=20.0, schedule_time_s=6.0),
            ),
        ),
        context_count=2,
    )

    assert [context.context_index for context in context_pool.contexts] == [0, 1]
    assert [context.initial_state.position_m for context in context_pool.contexts] == [
        0.0,
        10.0,
    ]


def test_single_context_pool_keeps_complete_task_start() -> None:
    context_pool, _ = _build_context_pool()
    reduced = ContextPoolBuilder.from_arrays(
        position_m=[0.0, 4.0, 9.0],
        speed_mps=[0.0, 4.0, 0.0],
        cumulative_time_s=[0.0, 2.0, 4.5],
        stepper=cast(OperationalStepper, cast(object, _FakeStepper())),
        context_count=1,
    )

    assert context_pool.context_count == 2
    assert reduced.context_count == 1
    assert reduced.context_at(0).initial_state.position_m == 0.0


def test_context_pool_uniform_partition_supports_reverse_task() -> None:
    context_pool = ContextPoolBuilder.from_arrays(
        position_m=[0.0, -4.0, -9.0],
        speed_mps=[0.0, 4.0, 0.0],
        cumulative_time_s=[0.0, 2.0, 4.5],
        stepper=cast(
            OperationalStepper,
            cast(object, _FakeStepper(target_position_m=-9.0)),
        ),
        context_count=3,
    )

    np.testing.assert_allclose(
        [context.initial_state.position_m for context in context_pool.contexts],
        [0.0, -3.0, -6.0],
    )


@pytest.mark.parametrize(
    ("position_m", "speed_mps", "time_s", "message"),
    [
        ([0.01, 4.0, 9.0], [0.0, 4.0, 0.0], [0.0, 2.0, 4.5], "start"),
        ([0.0, 4.0, 9.3], [0.0, 4.0, 0.0], [0.0, 2.0, 4.5], "target"),
        ([0.0, 4.0, 9.0], [0.02, 4.0, 0.0], [0.0, 2.0, 4.5], "start"),
        ([0.0, 4.0, 9.0], [0.0, 4.0, 0.02], [0.0, 2.0, 4.5], "end"),
        ([0.0, 4.0, 9.0], [0.0, 4.0, 0.0], [0.0, 2.0, 14.5], "time"),
    ],
)
def test_reference_sampler_rejects_invalid_task_source(
    position_m: list[float],
    speed_mps: list[float],
    time_s: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = ContextPoolBuilder.from_arrays(
            position_m=position_m,
            speed_mps=speed_mps,
            cumulative_time_s=time_s,
            stepper=cast(OperationalStepper, cast(object, _FakeStepper())),
            context_count=2,
        )


def test_reference_sampler_uses_source_timing_without_reconstruction() -> None:
    context_pool = ContextPoolBuilder.from_arrays(
        position_m=[0.0, 4.0, 9.0],
        speed_mps=[0.0, 2.0, 0.0],
        cumulative_time_s=[0.0, 3.0, 4.5],
        stepper=cast(OperationalStepper, cast(object, _FakeStepper())),
        context_count=2,
    )

    assert context_pool.context_at(1).initial_state.operation_time_s == pytest.approx(
        3.15
    )


def test_reference_sampler_rejects_schedule_time_mismatch() -> None:
    with pytest.raises(ValueError, match="total time error"):
        _ = ContextPoolBuilder.from_arrays(
            position_m=[0.0, 4.0, 9.0],
            speed_mps=[0.0, 4.0, 0.0],
            cumulative_time_s=[0.0, 3.0, 4.5],
            stepper=cast(
                OperationalStepper,
                cast(object, _FakeStepper(schedule_time_s=20.0)),
            ),
            context_count=2,
        )


def test_reference_sampler_rejects_unsafe_source_speed() -> None:
    with pytest.raises(ValueError, match="violates safety bounds"):
        _ = _build_context_pool(stepper=_FakeStepper(max_safe_speed_mps=3.0))
