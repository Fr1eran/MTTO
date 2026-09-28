"""Domain safeguard tests: dynamic limits, SPS stepping, curves and geometry helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from mtto.domain._numerics import (
    get_interval_index_array,
    get_interval_index_scalar_numba,
)
from mtto.domain.dynamics import Vehicle
from mtto.domain.line import Line
from mtto.domain.safeguard import (
    SPS,
    Safeguard,
    SafeGuardCurves,
    SafeguardParams,
    StaticRegion,
    ViolationKind,
    build_safeguard,
    current_stopping_point,
    detect_danger,
    dynamic_limit_violation,
    dynamic_limits,
    latest_intervention_points,
    max_speed,
    max_speed as get_max_speed,
    min_speed as get_min_speed,
)
from mtto.domain.safeguard.curves import _sanitize_curve
from mtto.domain.safeguard.geometry import (
    cal_regions,
    concatenate_curves_list,
    cut_curve_by_crosspoint,
    find_2curves_crosspoint,
    pad_2curve_lists,
    pad_2curves,
)
from mtto.io.scenario import load_tasks
from paper.figures import load_paper_scenario
from paper.real_operation import (
    DEFAULT_INPUT_FILE,
    DEFAULT_SHEET_NAME,
    load_real_operation_curve,
)


@pytest.fixture(scope="module")
def safeguard() -> Safeguard:
    scenario = load_paper_scenario()
    return scenario.safeguard


def test_detect_danger(safeguard: Safeguard):
    pos = np.array(
        [
            725,
            1754,
            2116,
            2762,
            4113,
            5484,
            6794,
            7800,
            11109,
            13125,
            17419,
            20060,
            6189.5,
            8548,
        ],
        dtype=np.float64,
    )
    speed = (
        np.array(
            [42, 8, 16, 9.5, 10, 15, 61, 45, 66, 74, 92, 90, 378, 372], dtype=np.float64
        )
        / 3.6
    )
    expected_result = np.array(
        [
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
        ]
    )
    result = detect_danger(safeguard, pos, speed)
    np.testing.assert_array_equal(result, expected_result)


def test_dynamic_limits_with_current_stopping_point_is_none(
    safeguard: Safeguard,
):
    pos: list[float] = [
        200.0,
        530.0,
        800.0,
        300.0,
        1250.0,
        1200.0,
        1800.0,
        3500.0,
        4500.0,
        8800.0,
        17000.0,
        22000.0,
        26500.0,
        29000.0,
    ]
    speed: list[float] = [
        8.8 / 3.6,
        9.0 / 3.6,
        5.0 / 3.6,
        15.0 / 3.6,
        20.0 / 3.6,
        45.0 / 3.6,
        60.0 / 3.6,
        60.0 / 3.6,
        60.0 / 3.6,
        25.0 / 3.6,
        50.0 / 3.6,
        130.0 / 3.6,
        60.0 / 3.6,
        60.0 / 3.6,
    ]
    input_sp: list[int] = [-1, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8]
    expected_IsCurrentMinSpeedEqualToZero: list[bool] = [
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    expected_IsCurrentSpeedBiggerThanCurrentMaxSpeed: list[bool] = [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    result_CurrentMinSpeed: list[float] = []
    result_CurrentMaxSpeed: list[float] = []
    for i in range(len(pos)):
        current_min_speed, current_max_speed = dynamic_limits(
            safeguard,
            pos[i],
            input_sp[i],
        )
        result_CurrentMinSpeed.append(current_min_speed)
        result_CurrentMaxSpeed.append(current_max_speed)
    result_IsCurrentMinSpeedEqualToZero = np.isclose(result_CurrentMinSpeed, 0.0)
    result_IsCurrentSpeedBigger = np.asarray(speed) > np.asarray(result_CurrentMaxSpeed)
    np.testing.assert_array_equal(
        result_IsCurrentMinSpeedEqualToZero, expected_IsCurrentMinSpeedEqualToZero
    )
    np.testing.assert_array_equal(
        result_IsCurrentSpeedBigger,
        expected_IsCurrentSpeedBiggerThanCurrentMaxSpeed,
    )


def test_dynamic_limits_with_current_stopping_point_is_not_none(
    safeguard: Safeguard,
):
    pos: list[float] = [
        200.0,
        530.0,
        800.0,
        300.0,
        1300.0,
        800.0,
        2000.0,
        2600.0,
        5230.0,
        28600.0,
        29312.0,
    ]
    speed: list[float] = [
        8.8 / 3.6,
        9.0 / 3.6,
        5.0 / 3.6,
        15.0 / 3.6,
        10.0 / 3.6,
        65.0 / 3.6,
        50.0 / 3.6,
        50.0 / 3.6,
        8.0 / 3.6,
        70.0 / 3.6,
        40.0 / 3.6,
    ]
    sp: list[int] = [
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        1,
        2,
        3,
        8,
        8,
    ]
    expected_IsCurrentMinSpeedEqualToZero: list[bool] = [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    expected_IsCurrentMaxSpeedBigger: list[bool] = [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]
    result_CurrentMinSpeed: list[float] = []
    result_CurrentMaxSpeed: list[float] = []
    for i in range(len(pos)):
        current_sp = sp[i]
        current_min_speed, current_max_speed = dynamic_limits(
            safeguard,
            pos[i],
            current_sp,
        )
        result_CurrentMinSpeed.append(current_min_speed)
        result_CurrentMaxSpeed.append(current_max_speed)
    result_IsCurrentMinSpeedEqualToZero = np.isclose(result_CurrentMinSpeed, 0.0)
    result_IsCurrentMaxSpeedBiggerThanCurrentSpeed = np.asarray(
        result_CurrentMaxSpeed
    ) > np.asarray(speed)
    np.testing.assert_array_equal(
        result_IsCurrentMinSpeedEqualToZero, expected_IsCurrentMinSpeedEqualToZero
    )
    np.testing.assert_array_equal(
        result_IsCurrentMaxSpeedBiggerThanCurrentSpeed, expected_IsCurrentMaxSpeedBigger
    )


@pytest.fixture(scope="module")
def position_safeguard() -> Safeguard:
    dummy_params = SafeguardParams(
        factor=1.0,
        step_delay_s=2.0,
        distance_step_m=1.0,
        position_error_m=0.0,
        speed_error_mps=0.0,
        traction_cutoff_delay_s=0.6,
        vortex_brake_delay_s=0.6,
        min_curve_position_offset_m=0.0,
        generation_danger_points_m=(100.0,),
    )
    dummy_static = StaticRegion(
        idp_points_x=np.empty(0, dtype=np.float64),
        min_curves_part_x_padded=(),
        min_curves_part_y_padded=(),
        max_curves_part_x_padded=(),
        max_curves_part_y_padded=(),
        num_regions=0,
    )
    return Safeguard(
        params=dummy_params,
        static_region=dummy_static,
        speed_limits=np.array([10.0], dtype=np.float64),
        speed_limit_intervals=np.array([0.0, 100.0], dtype=np.float64),
        levi_curves=(),
        brake_curves=(),
        min_curves=(
            np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 0.0]], dtype=np.float64),
        ),
        max_curves=(
            np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 4.0, 2.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0, 2.0, 3.0], [6.0, 5.0, 3.0, 0.0]], dtype=np.float64),
        ),
        min_pos_packed=np.empty(0, dtype=np.float64),
        min_speed_packed=np.empty(0, dtype=np.float64),
        min_lengths=np.empty(0, dtype=np.int32),
        max_pos_packed=np.empty(0, dtype=np.float64),
        max_speed_packed=np.empty(0, dtype=np.float64),
        max_lengths=np.empty(0, dtype=np.int32),
    )


def test_get_min_and_max_position_with_currentsp(
    position_safeguard: Safeguard,
):
    min_pos, max_pos = latest_intervention_points(
        position_safeguard,
        1.5,
        0,
    )
    np.testing.assert_allclose(min_pos, 1.5)
    np.testing.assert_allclose(max_pos, 1.75)


def test_get_min_and_max_position_with_currentsp_extrapolation(
    position_safeguard: Safeguard,
):
    min_pos, max_pos = latest_intervention_points(
        position_safeguard,
        4.5,
        0,
    )
    np.testing.assert_allclose(min_pos, -1.5)
    np.testing.assert_allclose(max_pos, -0.5)


def test_get_min_and_max_position_with_currentsp_negative_one(
    position_safeguard: Safeguard,
):
    min_pos, max_pos = latest_intervention_points(
        position_safeguard,
        1.5,
        -1,
    )
    np.testing.assert_allclose(min_pos, 0.0)
    np.testing.assert_allclose(max_pos, 2.25)


def test_get_min_and_max_position_real_data_smoke(
    safeguard: Safeguard,
):
    min_pos, max_pos = latest_intervention_points(
        safeguard,
        2.0,
        0,
    )
    assert np.isfinite(min_pos)
    assert np.isfinite(max_pos)
    assert min_pos <= max_pos


def test_loaded_max_curves_are_monotone_after_init(
    safeguard: Safeguard,
):
    for curve in safeguard.max_curves:
        assert np.all(np.diff(curve[1, :]) <= 1e-9)


def test_sanitize_curve_projects_speed_to_monotone() -> None:
    curve = np.array([[0.0, 10.0, 20.0, 30.0], [12.0, 8.0, 11.0, 5.0]])
    sanitized = _sanitize_curve(curve)

    np.testing.assert_array_equal(sanitized[0], curve[0])
    np.testing.assert_array_equal(sanitized[1], np.minimum.accumulate(curve[1]))
    assert np.all(np.diff(sanitized[1]) <= 0.0)


def test_get_min_and_max_position_real_data_sp7_regression(
    safeguard: Safeguard,
):
    if len(safeguard.max_curves) < 9:
        pytest.skip("requires full rail data with at least 9 max curves")

    min_pos, max_pos = latest_intervention_points(
        safeguard,
        10.0,
        7,
    )
    assert np.isfinite(min_pos)
    assert np.isfinite(max_pos)
    assert min_pos <= max_pos


def test_get_interval_index_scalar_numba_matches_reference():
    interval_points = np.array([0.0, 100.0, 230.0, 500.0], dtype=np.float64)
    points = np.array(
        [-50.0, 0.0, 10.0, 100.0, 229.9, 230.0, 499.9, 500.0, 900.0],
        dtype=np.float64,
    )

    expected = np.asarray(
        get_interval_index_array(points, interval_points), dtype=np.int64
    )
    actual = np.asarray(
        [
            get_interval_index_scalar_numba(float(pos), interval_points)
            for pos in points
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_get_interval_index_scalar_numba_supports_left_and_right_side():
    interval_points = np.array([0.0, 100.0, 230.0, 500.0], dtype=np.float64)
    points = np.array(
        [-50.0, 0.0, 10.0, 100.0, 229.9, 230.0, 499.9, 500.0, 900.0],
        dtype=np.float64,
    )

    expected_right = np.searchsorted(interval_points, points, side="right") - 1
    actual_right = np.asarray(
        [
            get_interval_index_scalar_numba(
                float(pos),
                interval_points,
                True,
            )
            for pos in points
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual_right, expected_right)

    expected_left = np.searchsorted(interval_points, points, side="left") - 1
    actual_left = np.asarray(
        [
            get_interval_index_scalar_numba(
                float(pos),
                interval_points,
                False,
            )
            for pos in points
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual_left, expected_left)


def test_dynamic_limits_valid_bounds(
    safeguard: Safeguard,
):
    rng = np.random.default_rng(0)
    pos_random = rng.uniform(
        low=float(safeguard.speed_limit_intervals[0]),
        high=float(safeguard.speed_limit_intervals[-1]),
        size=128,
    )
    pos_boundaries = safeguard.speed_limit_intervals
    positions = np.unique(np.concatenate([pos_random, pos_boundaries]))
    stopping_points = list(range(-1, len(safeguard.min_curves)))

    for sp in stopping_points:
        for pos in positions:
            min_speed, max_speed = dynamic_limits(safeguard, float(pos), int(sp))
            assert min_speed >= 0.0
            assert max_speed >= 0.0


def test_get_current_stopping_point_numba(
    safeguard: Safeguard,
):
    # 起始位置与0速应处于 -1 停车点
    sp_start = current_stopping_point(safeguard, 0.0, 0.0)
    assert sp_start == -1

    # 超过最后一条最小速度曲线右端点时应达到最大停车点编号
    last_curve_end = float(safeguard.min_curves[-1][0, -1])
    sp_end = current_stopping_point(safeguard, last_curve_end + 100.0, 0.0)
    assert sp_end == len(safeguard.min_curves) - 1


def test_min_speed_matches_dynamic_limits(
    safeguard: Safeguard,
):
    rng = np.random.default_rng(1)
    positions = rng.uniform(
        low=float(safeguard.speed_limit_intervals[0]),
        high=float(safeguard.speed_limit_intervals[-1]),
        size=64,
    )
    stopping_points = list(range(-1, len(safeguard.min_curves)))

    for sp in stopping_points:
        for pos in positions:
            min_val, _ = dynamic_limits(safeguard, float(pos), int(sp))
            min_only = get_min_speed(safeguard, float(pos), int(sp))
            np.testing.assert_allclose(min_only, min_val, rtol=0.0, atol=1e-12)


def test_max_speed_matches_dynamic_limits(
    safeguard: Safeguard,
):
    rng = np.random.default_rng(2)
    positions = rng.uniform(
        low=float(safeguard.speed_limit_intervals[0]),
        high=float(safeguard.speed_limit_intervals[-1]),
        size=64,
    )
    stopping_points = list(range(-1, len(safeguard.min_curves)))

    for sp in stopping_points:
        for pos in positions:
            _, max_speed = dynamic_limits(safeguard, float(pos), int(sp))
            max_only = get_max_speed(safeguard, float(pos), int(sp))
            np.testing.assert_allclose(max_only, max_speed, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    (
        "pos",
        "speed",
        "min_speed_val",
        "max_speed_val",
        "expected_kind",
        "expected_limit",
        "expected_margin",
    ),
    [
        # 1. 低于下界
        (100.0, 5.0, 10.0, 50.0, ViolationKind.UNDER_LOWER_LIMIT, 10.0, 5.0),
        # 2. 高于上界
        (200.0, 55.0, 10.0, 50.0, ViolationKind.OVER_UPPER_LIMIT, 50.0, 5.0),
        # 3. 恰好在下界 (不判定为违规)
        (300.0, 10.0, 10.0, 50.0, None, None, None),
        # 4. 恰好在上界 (不判定为违规)
        (400.0, 50.0, 10.0, 50.0, None, None, None),
        # 5. 在界内 (无违规)
        (500.0, 30.0, 10.0, 50.0, None, None, None),
    ],
)
def test_dynamic_limit_violation(
    pos: float,
    speed: float,
    min_speed_val: float,
    max_speed_val: float,
    expected_kind: ViolationKind | None,
    expected_limit: float | None,
    expected_margin: float | None,
):
    violation = dynamic_limit_violation(pos, speed, min_speed_val, max_speed_val)
    if expected_kind is None:
        assert violation is None
    else:
        assert violation is not None
        assert violation.kind == expected_kind
        assert violation.position_m == pos
        assert np.isclose(violation.margin_mps, expected_margin)


def test_safeguard_arrays_are_read_only(safeguard: Safeguard) -> None:
    assert not safeguard.speed_limits.flags.writeable
    assert not safeguard.speed_limit_intervals.flags.writeable
    assert not safeguard.min_pos_packed.flags.writeable
    assert not safeguard.min_speed_packed.flags.writeable
    assert not safeguard.min_lengths.flags.writeable
    assert not safeguard.max_pos_packed.flags.writeable
    assert not safeguard.max_speed_packed.flags.writeable
    assert not safeguard.max_lengths.flags.writeable

    for c in safeguard.levi_curves:
        assert not c.flags.writeable
    for c in safeguard.brake_curves:
        assert not c.flags.writeable
    for c in safeguard.min_curves:
        assert not c.flags.writeable
    for c in safeguard.max_curves:
        assert not c.flags.writeable

    with pytest.raises(ValueError):
        safeguard.min_curves[0][0, 0] = 999.0
    with pytest.raises(ValueError):
        safeguard.max_curves[0][0, 0] = 999.0
    with pytest.raises(ValueError):
        safeguard.speed_limits[0] = 999.0


@pytest.fixture(scope="module")
def safeguard_curves_and_vehicle(
    paper_scenario,
) -> tuple[SafeGuardCurves, Vehicle]:
    cal_SGC = SafeGuardCurves(track=paper_scenario.line)
    return cal_SGC, paper_scenario.vehicle


def test_cal_levi_curves(
    safeguard_curves_and_vehicle: tuple[SafeGuardCurves, Vehicle],
    paper_scenario,
):
    cal_SGC, vehicle = safeguard_curves_and_vehicle
    aps = paper_scenario.line.accessible_points_m
    curves = cal_SGC.calc_levi_curves(np.asarray(aps, dtype=np.float64), vehicle, ds=1)
    # 检查返回类型和内容
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    # 检查每个曲线的横纵坐标长度一致
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)  # 每条曲线横坐标数组
        assert isinstance(item[1, :], np.ndarray)  # 每条曲线纵坐标数组


def test_cal_brake_curves(
    safeguard_curves_and_vehicle: tuple[SafeGuardCurves, Vehicle],
    paper_scenario,
):
    cal_SGC, vehicle = safeguard_curves_and_vehicle
    dps = paper_scenario.line.danger_points_m
    curves = cal_SGC.calc_brake_curves(np.asarray(dps, dtype=np.float64), vehicle, ds=1)
    assert isinstance(curves, list)
    assert all(isinstance(item, np.ndarray) and item.shape[0] == 2 for item in curves)
    for item in curves:
        assert isinstance(item[0, :], np.ndarray)
        assert isinstance(item[1, :], np.ndarray)


def test_calc_brake_and_max_curves_max_speed_is_monotone(
    safeguard_curves_and_vehicle: tuple[SafeGuardCurves, Vehicle],
    paper_scenario,
):
    cal_SGC, vehicle = safeguard_curves_and_vehicle
    sp = paper_scenario.safeguard.params
    dps = paper_scenario.line.danger_points_m
    _, max_curves = cal_SGC.calc_brake_and_max_curves(
        dpoffsets=np.asarray(dps, dtype=np.float64),
        vehicle=vehicle,
        ds=1.0,
        pos_error=sp.position_error_m,
        speed_error=sp.speed_error_mps,
        delay_time_until_DPS_done=sp.traction_cutoff_delay_s,
        delay_time_until_VB_begin=sp.vortex_brake_delay_s,
    )

    for curve in max_curves:
        speed_diff = np.diff(curve[1, :])
        assert np.all(speed_diff <= 1e-9)


def test_safeguard_curves_reproducibility_and_counts() -> None:
    first_sg = load_paper_scenario().safeguard
    second_sg = load_paper_scenario().safeguard
    first = (
        first_sg.levi_curves,
        first_sg.brake_curves,
        first_sg.min_curves,
        first_sg.max_curves,
    )
    second = (
        second_sg.levi_curves,
        second_sg.brake_curves,
        second_sg.min_curves,
        second_sg.max_curves,
    )

    levi_1, brake_1, min_1, max_1 = first
    assert len(levi_1) == 9
    assert len(min_1) == 9
    assert len(brake_1) == 10
    assert len(max_1) == 10

    for curves_a, curves_b in zip(first, second, strict=True):
        assert len(curves_a) == len(curves_b)
        for arr_a, arr_b in zip(curves_a, curves_b, strict=True):
            assert arr_a.dtype == arr_b.dtype
            assert np.array_equal(arr_a, arr_b)


def _make_dummy_curve(
    x0: float, x1: float, y0: float = 0.0, y1: float = 50.0
) -> np.ndarray:
    return np.array([[x0, x1], [y0, y1]], dtype=np.float64)


def _make_dummy_params() -> SafeguardParams:
    return SafeguardParams(
        factor=0.99,
        step_delay_s=2.0,
        distance_step_m=1.0,
        position_error_m=10.0,
        speed_error_mps=1.0,
        traction_cutoff_delay_s=0.6,
        vortex_brake_delay_s=0.6,
        min_curve_position_offset_m=0.0,
        generation_danger_points_m=(100.0,),
    )


def _make_dummy_line() -> Line:
    return Line(
        slopes=np.array([0.0], dtype=np.float64),
        slope_intervals=np.array([0.0, 500.0], dtype=np.float64),
        speed_limits=np.array([100.0], dtype=np.float64),
        speed_limit_intervals=np.array([0.0, 500.0], dtype=np.float64),
        accessible_points_m=(100.0,),
        danger_points_m=(110.0,),
    )


def _make_safeguard(accessible_count: int) -> Safeguard:
    return build_safeguard(
        params=_make_dummy_params(),
        line=_make_dummy_line(),
        levi_curves=[_make_dummy_curve(0.0, 100.0)] * accessible_count,
        brake_curves=[_make_dummy_curve(0.0, 100.0)] * (accessible_count + 1),
        min_curves=[_make_dummy_curve(0.0, 100.0, 0.0, 50.0)] * accessible_count,
        max_curves=[_make_dummy_curve(0.0, 100.0, 50.0, 0.0)] * (accessible_count + 1),
    )


def test_sps_advances_explicit_state_after_delay() -> None:
    sps = SPS(
        safeguard=_make_safeguard(accessible_count=1),
        accessible_positions_m=[100.0],
        danger_positions_m=[110.0],
        step_delay_s=2.0,
    )
    state = sps.initial_state()
    requested = sps.advance(state, position_m=0.0, speed_mps=2.0, time_s=5.0)
    assert requested.target_stopping_point_index == -1
    assert requested.request_pending is True
    assert requested.request_started_at_s == 5.0

    completed = sps.advance(requested, position_m=0.0, speed_mps=2.0, time_s=7.0)
    assert completed.target_stopping_point_index == 0
    assert completed.request_pending is False


def test_sps_keeps_current_target_when_step_window_is_missed() -> None:
    sps = SPS(
        safeguard=_make_safeguard(accessible_count=1),
        accessible_positions_m=[100.0],
        danger_positions_m=[110.0],
        step_delay_s=2.0,
    )
    requested = sps.advance(
        sps.initial_state(), position_m=0.0, speed_mps=2.0, time_s=0.0
    )

    missed = sps.advance(requested, position_m=1.0, speed_mps=150.0, time_s=3.0)

    assert missed == requested


def test_sps_validates_stopping_points_and_returns_target_midpoint() -> None:
    guard = _make_safeguard(accessible_count=2)
    sps = SPS(
        safeguard=guard,
        accessible_positions_m=[100.0, 200.0],
        danger_positions_m=[110.0, 220.0],
        step_delay_s=2.0,
    )

    assert sps.target_position_m(1) == 210.0
    with pytest.raises(ValueError, match="counts must match"):
        _ = SPS(
            safeguard=guard,
            accessible_positions_m=[100.0],
            danger_positions_m=[110.0, 220.0],
            step_delay_s=2.0,
        )


class _SetupData(TypedDict):
    distance: NDArray[np.float64]
    speed: NDArray[np.float64]
    time: NDArray[np.float64]
    accessible_points: list[float]
    dangerous_points: list[float]
    safeguard: Safeguard


class TestSPSIntegration:
    @pytest.fixture
    def setup_data(self, paper_scenario) -> _SetupData:
        task = load_tasks(Path("paper/specs/tasks.toml"))["longyang_to_airport"]
        curve = load_real_operation_curve(
            input_file=DEFAULT_INPUT_FILE,
            sheet_name=DEFAULT_SHEET_NAME,
            start_position_m=task.start_position_m,
            target_position_m=task.target_position_m,
        )
        distance = np.asarray(curve["source_position_m"][1:], dtype=np.float64)
        speed_mps = np.asarray(curve["speed_mps"][1:], dtype=np.float64)
        travel_time = np.asarray(curve["source_time_s"][1:], dtype=np.float64)

        safeguard = build_safeguard(
            params=replace(paper_scenario.safeguard.params, factor=0.9),
            line=paper_scenario.line,
            levi_curves=paper_scenario.safeguard.levi_curves,
            brake_curves=paper_scenario.safeguard.brake_curves,
            min_curves=paper_scenario.safeguard.min_curves,
            max_curves=paper_scenario.safeguard.max_curves,
        )

        return {
            "distance": distance,
            "speed": speed_mps,
            "time": travel_time,
            "accessible_points": list(paper_scenario.line.accessible_points_m),
            "dangerous_points": list(paper_scenario.line.danger_points_m),
            "safeguard": safeguard,
        }

    @pytest.fixture
    def setup_system(self, setup_data: _SetupData) -> tuple[Safeguard, int]:
        return setup_data["safeguard"], len(setup_data["accessible_points"])

    def test_sps_real_data_stops_at_the_first_missed_step_window(
        self,
        setup_data: _SetupData,
        setup_system: tuple[Safeguard, int],
    ):
        """使用真实运行数据测试停车点步进机制实现"""
        safeguard, num_sp = setup_system

        T_r = 2.0
        sps = SPS(
            safeguard=safeguard,
            accessible_positions_m=setup_data["accessible_points"],
            danger_positions_m=setup_data["dangerous_points"],
            step_delay_s=T_r,
        )

        sps_state = sps.initial_state()
        current_sp = sps_state.target_stopping_point_index
        request_seen = False
        window_missed = False

        distances = setup_data["distance"]
        speeds = setup_data["speed"]
        times = setup_data["time"]

        for i in range(len(times)):
            t = times[i]
            x = distances[i]
            v = speeds[i]

            previous_state = sps_state
            sps_state = sps.advance(
                previous_state,
                position_m=x,
                speed_mps=v,
                time_s=t,
            )
            request_seen = request_seen or (
                not previous_state.request_pending and sps_state.request_pending
            )
            old_max_speed = max_speed(
                safeguard,
                position_m=x,
                stopping_point=previous_state.target_stopping_point_index,
            )
            if (
                previous_state.request_pending
                and sps_state == previous_state
                and v > old_max_speed
            ):
                window_missed = True
                break
            current_sp = sps_state.target_stopping_point_index

        assert request_seen, "No stepping request occurred in real-data replay"
        assert window_missed, "Expected the replay to expose a missed step window"
        assert current_sp < num_sp - 1


def test_concatenate_curves_list_empty() -> None:
    x, y = concatenate_curves_list([])
    assert x.size == 0
    assert y.size == 0
    assert x.dtype == np.float64
    assert y.dtype == np.float64


def test_concatenate_curves_list_normal() -> None:
    c1 = np.array([[0.0, 1.0], [10.0, 20.0]], dtype=np.float64)
    c2 = np.array([[2.0, 3.0, 4.0], [30.0, 40.0, 50.0]], dtype=np.float64)

    x, y = concatenate_curves_list([c1, c2])

    assert x.shape == (5,)
    assert y.shape == (5,)
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y, [10.0, 20.0, 30.0, 40.0, 50.0])


def test_find_2curves_crosspoint_standard() -> None:
    # Line 1: y = 2x from x=0 to 10
    c1 = np.array([[0.0, 10.0], [0.0, 20.0]], dtype=np.float64)
    # Line 2: y = -x + 15 from x=0 to 10. Intersection at x=5, y=10.
    c2 = np.array([[0.0, 10.0], [15.0, 5.0]], dtype=np.float64)

    x_cross, y_cross = find_2curves_crosspoint(c1, c2)
    assert x_cross == pytest.approx(5.0, abs=1e-2)
    assert y_cross == pytest.approx(10.0, abs=1e-2)


def test_find_2curves_crosspoint_errors() -> None:
    # No x-domain overlap
    c1 = np.array([[0.0, 5.0], [0.0, 10.0]], dtype=np.float64)
    c2 = np.array([[6.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="overlapping x-domain"):
        _ = find_2curves_crosspoint(c1, c2)

    # Overlapping domain but parallel without intersection
    c3 = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    c4 = np.array([[0.0, 10.0], [10.0, 20.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="intersect"):
        _ = find_2curves_crosspoint(c3, c4)


def test_cut_curve_by_crosspoint() -> None:
    s = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    v = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

    cut_s, cut_v = cut_curve_by_crosspoint(s, v, 3.5)
    np.testing.assert_allclose(cut_s, [4.0, 6.0, 8.0])
    np.testing.assert_allclose(cut_v, [30.0, 40.0, 50.0])


def test_cal_regions() -> None:
    # Above curve 1: y = 20 - x (x in 0..10)
    c_above = np.array([[0.0, 10.0], [20.0, 10.0]], dtype=np.float64)
    # Below curve 1: y = x (x in 0..10), cross at x=10, y=10
    c_below = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)

    cross_pts, above_parts, below_parts = cal_regions([c_above], [c_below])

    assert cross_pts.shape == (2, 1)
    assert cross_pts[0, 0] == pytest.approx(10.0, abs=1e-2)
    assert cross_pts[1, 0] == pytest.approx(10.0, abs=1e-2)
    assert len(above_parts) == 1
    assert len(below_parts) == 1


def test_cal_regions_mismatched_length_raises() -> None:
    c = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="same size"):
        _ = cal_regions([c], [])


def test_pad_2curves_and_pad_2curve_lists() -> None:
    c1_x = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    c1_y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    c2_x = np.array([0.0, 5.0], dtype=np.float64)
    c2_y = np.array([4.0, 5.0], dtype=np.float64)

    p1_x, p1_y, p2_x, p2_y = pad_2curves(c1_x, c1_y, c2_x, c2_y)
    assert len(p1_x) == 3
    assert len(p2_x) == 3
    assert p2_x[-1] == pytest.approx(10.0)
    assert p2_y[-1] == pytest.approx(5.0)

    # Test reverse direction padding
    p1_x_r, p1_y_r, p2_x_r, p2_y_r = pad_2curves(c2_x, c2_y, c1_x, c1_y)
    assert len(p1_x_r) == 3
    assert len(p2_x_r) == 3

    # Test list padding
    c1 = np.stack([c1_x, c1_y], axis=0)
    c2 = np.stack([c2_x, c2_y], axis=0)
    list1_p, list2_p = pad_2curve_lists([c1], [c2])
    assert len(list1_p) == 1
    assert len(list2_p) == 1
    assert list1_p[0].shape == (2, 3)
    assert list2_p[0].shape == (2, 3)

    with pytest.raises(ValueError, match="same size"):
        _ = pad_2curve_lists([c1], [])
