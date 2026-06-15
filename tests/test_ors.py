import json
from pathlib import Path

import numpy as np
import pytest

from model.vehicle import VehicleInfo
from model.track import TrackInfo
from model.ocs import TrainService
from model.common import ORS
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)
from utils.indexing_utils import find_speed_rise_entry_and_fall
from utils.io_utils import save_curve_and_metrics


@pytest.fixture(scope="module")
def operation_reference_system():
    # 坡度，百分位
    slopes, slope_intervals = load_slopes()

    # 区间限速
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

    aps, dps = load_auxiliary_stopping_areas_ap_and_dp()

    # 车站
    ly_zp, pa_zp = load_stations_goal_positions()

    track = TrackInfo(
        slopes,
        slope_intervals,
        speed_limits.tolist(),
        speed_limit_intervals,
        ASA_aps=aps,
        ASA_dps=dps,
    )
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    train_service = TrainService(
        start_position=ly_zp,
        start_speed=0.0,
        target_position=pa_zp,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=120.0,
        max_stop_error=0.3,
    )
    return ORS(vehicle=vehicle, track=track, factor=0.95)


def _sum_operation_time(operations) -> float:
    return float(sum(float(time) for _, time in operations))


def _build_fake_reference_ors() -> ORS:
    ors = ORS.__new__(ORS)

    def _fake_calc_min_operation_time(
        *,
        begin_pos: float,
        begin_speed: float,
        end_pos: float,
        end_speed: float,
    ) -> float:
        del end_speed
        return (float(end_pos) - float(begin_pos)) / 10.0 + 0.1 * float(begin_speed)

    ors.calc_min_operation_time = _fake_calc_min_operation_time
    return ors


def _write_dp_reference_artifact(
    run_dir: Path,
    *,
    include_cum_time: bool = True,
    schedule_time_s: float = 10.0,
    start_position: float = 0.0,
    start_speed: float = 0.0,
    target_position: float = 20.0,
    target_speed: float = 0.0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "target_time_s": schedule_time_s,
        "start_position_m": start_position,
        "start_speed_mps": start_speed,
        "target_position_m": target_position,
        "target_speed_mps": target_speed,
    }
    if include_cum_time:
        save_curve_and_metrics(
            pos_arr=[0.0, 10.0, 20.0],
            speed_arr=[0.0, 2.0, 0.0],
            output_path=str(run_dir / "optimized_speed_curve.npz"),
            extra_arrays={"cum_time_s": [0.0, 2.0, 5.0]},
            metrics=metrics,
        )
    else:
        np.savez_compressed(
            run_dir / "optimized_speed_curve.npz",
            pos_m=np.asarray([0.0, 1.0, 3.0], dtype=np.float32),
            speed_mps=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )
        (run_dir / "optimized_speed_curve_metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )


def test_find_speed_rise_entry_and_fall_is_stable_under_repeated_calls(
    operation_reference_system: ORS,
):
    ors = operation_reference_system
    n_limits = len(ors.track.speed_limits)
    start_idx = max(0, n_limits // 8)
    end_idx = min(n_limits - 1, start_idx + max(2, n_limits // 5))

    first = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )
    second = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Interleave another range query; same-range result should remain unchanged.
    _ = ors._find_speed_rise_entry_and_fall(
        start_idx=max(0, start_idx - 2),
        end_idx=min(n_limits - 1, end_idx + 2),
    )
    third = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )

    assert first == second
    assert second == third


def test_find_speed_rise_entry_and_fall_matches_indexing_utils(
    operation_reference_system: ORS,
):
    ors = operation_reference_system
    n_limits = len(ors.track.speed_limits)
    start_idx = max(0, n_limits // 10)
    end_idx = min(n_limits - 1, start_idx + max(3, n_limits // 4))

    rise_entries, fall_exits = find_speed_rise_entry_and_fall(
        speed_limits=ors.track.speed_limits,
        interval_points=ors.track.speed_limit_intervals,
        start_idx=start_idx,
        end_idx=end_idx,
        speed_factor=ors.gamma,
    )
    expected_rise = [
        (entry.boundary_pos, entry.left_speed_scaled, entry.next_interval)
        for entry in rise_entries
    ]
    expected_fall = [
        (entry.boundary_pos, entry.right_speed_scaled, entry.prev_interval)
        for entry in fall_exits
    ]

    actual_rise_points, actual_fall_points = ors._find_speed_rise_entry_and_fall(
        start_idx=start_idx,
        end_idx=end_idx,
    )
    actual_rise = [
        (point.begin_pos, point.begin_speed, point.begin_interval)
        for point in actual_rise_points
    ]
    actual_fall = [
        (point.end_pos, point.end_speed, point.end_interval)
        for point in actual_fall_points
    ]

    assert actual_rise == expected_rise
    assert actual_fall == expected_fall


def test_calc_min_operation_time_matches_runtime_operation_sum(
    operation_reference_system: ORS,
):
    ors = operation_reference_system
    train_start = float(ors.track.speed_limit_intervals[0])
    train_end = float(ors.track.speed_limit_intervals[-1])
    target_end = float(
        max(train_start, min(train_end, ors.track.speed_limit_intervals[-1] - 1.0))
    )

    rng = np.random.default_rng(123)
    for _ in range(16):
        begin_pos = float(rng.uniform(train_start, target_end))
        begin_speed = float(rng.uniform(0.0, 80.0 / 3.6))
        end_pos = float(rng.uniform(begin_pos, train_end))
        end_speed = 0.0

        runtime_time = ors.calc_min_operation_time(
            begin_pos=begin_pos,
            begin_speed=begin_speed,
            end_pos=end_pos,
            end_speed=end_speed,
        )
        operations = ors._calc_min_runtime_operation(
            current_pos=begin_pos,
            current_speed=begin_speed,
            end_pos=end_pos,
            end_speed=end_speed,
        )
        operations_time = _sum_operation_time(operations)
        np.testing.assert_allclose(runtime_time, operations_time, rtol=0.0, atol=1e-9)


def test_load_or_build_ref_redundant_operation_time_loads_matching_dp_artifact(
    tmp_path: Path,
) -> None:
    _write_dp_reference_artifact(tmp_path / "run")
    ors = _build_fake_reference_ors()

    pos, speed, cum_time, ref_redundant = (
        ors.load_or_build_ref_redundant_operation_time_from_dp(
            start_position=0.0,
            start_speed=0.0,
            target_position=20.0,
            schedule_time_s=10.0,
            target_speed=0.0,
            dp_curve_dir=tmp_path,
        )
    )

    np.testing.assert_allclose(pos, np.asarray([0.0, 10.0, 20.0]))
    np.testing.assert_allclose(speed, np.asarray([0.0, 2.0, 0.0]))
    np.testing.assert_allclose(cum_time, np.asarray([0.0, 2.0, 5.0]))
    np.testing.assert_allclose(ref_redundant, np.asarray([8.0, 6.8, 5.0]))


def test_load_or_build_ref_redundant_operation_time_recovers_legacy_cum_time(
    tmp_path: Path,
) -> None:
    _write_dp_reference_artifact(tmp_path / "legacy", include_cum_time=False)
    ors = _build_fake_reference_ors()

    pos, speed, cum_time, ref_redundant = (
        ors.load_or_build_ref_redundant_operation_time_from_dp(
            start_position=0.0,
            start_speed=0.0,
            target_position=20.0,
            schedule_time_s=10.0,
            target_speed=0.0,
            dp_curve_dir=tmp_path,
        )
    )

    np.testing.assert_allclose(pos, np.asarray([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(speed, np.asarray([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(cum_time, np.asarray([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(ref_redundant, np.asarray([7.9, 7.0, 5.2]))


def test_load_or_build_ref_redundant_operation_time_raises_without_dp_source(
    tmp_path: Path,
) -> None:
    ors = _build_fake_reference_ors()

    with pytest.raises(FileNotFoundError, match="matching DP trajectory"):
        ors.load_or_build_ref_redundant_operation_time_from_dp(
            start_position=0.0,
            start_speed=0.0,
            target_position=20.0,
            schedule_time_s=10.0,
            dp_curve_dir=tmp_path,
        )


def test_load_or_build_ref_redundant_operation_time_uses_injected_compute_callback(
    tmp_path: Path,
) -> None:
    ors = _build_fake_reference_ors()
    called_kwargs: dict[str, float] = {}

    def _compute_dp_curve(**kwargs):
        called_kwargs.update(kwargs)
        return (
            np.asarray([0.0, 20.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 6.0], dtype=np.float32),
            {"target_time_s": 10.0},
        )

    pos, speed, cum_time, ref_redundant = (
        ors.load_or_build_ref_redundant_operation_time_from_dp(
            start_position=0.0,
            start_speed=0.0,
            target_position=20.0,
            schedule_time_s=10.0,
            target_speed=0.0,
            dp_curve_dir=tmp_path,
            compute_dp_curve=_compute_dp_curve,
        )
    )

    assert called_kwargs["schedule_time_s"] == pytest.approx(10.0)
    np.testing.assert_allclose(pos, np.asarray([0.0, 20.0]))
    np.testing.assert_allclose(speed, np.asarray([0.0, 0.0]))
    np.testing.assert_allclose(cum_time, np.asarray([0.0, 6.0]))
    np.testing.assert_allclose(ref_redundant, np.asarray([8.0, 4.0]))
