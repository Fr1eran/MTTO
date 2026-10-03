from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mtto.domain.line import Line
from mtto.domain.safeguard import (
    SPS,
    SafeguardParams,
    build_safeguard,
)
from mtto.domain.scenario import (
    EnergyParams,
    ScheduleChange,
    StopState,
    Task,
)
from mtto.io.scenario import compute_scenario_hash, load_scenario, load_tasks

# -----------------------------------------------------------------------------
# 1. 构造校验测试
# -----------------------------------------------------------------------------


def test_schedule_change_valid():
    sc = ScheduleChange(trigger_position_m=100.0, new_schedule_time_s=450.0)
    assert sc.trigger_position_m == 100.0
    assert sc.new_schedule_time_s == 450.0


@pytest.mark.parametrize(
    ("trigger", "new_time", "match"),
    [
        (float("nan"), 100.0, "finite"),
        (float("inf"), 100.0, "finite"),
        (100.0, float("nan"), "finite"),
        (100.0, float("inf"), "finite"),
        (100.0, 0.0, "positive"),
        (100.0, -10.0, "positive"),
    ],
)
def test_schedule_change_invalid(trigger: float, new_time: float, match: str):
    with pytest.raises(ValueError, match=match):
        ScheduleChange(trigger_position_m=trigger, new_schedule_time_s=new_time)


def test_task_valid():
    task = Task(
        start_position_m=100.0,
        target_position_m=1000.0,
        schedule_time_s=400.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=ScheduleChange(500.0, 450.0),
    )
    assert task.start_position_m == 100.0
    assert task.target_position_m == 1000.0


def test_task_valid_none_schedule_time():
    task = Task(
        start_position_m=100.0,
        target_position_m=1000.0,
        schedule_time_s=None,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=None,
    )
    assert task.schedule_time_s is None


@pytest.mark.parametrize(
    (
        "start",
        "target",
        "sched_time",
        "max_acc",
        "max_stop",
        "max_arr",
        "change",
        "match",
    ),
    [
        # 反向运行 / 目标小于等于起点
        (
            1000.0,
            100.0,
            400.0,
            0.75,
            0.3,
            10.0,
            None,
            "反向运行暂不支持",
        ),
        (
            100.0,
            100.0,
            400.0,
            0.75,
            0.3,
            10.0,
            None,
            "反向运行暂不支持",
        ),
        # 非有限值
        (float("nan"), 1000.0, 400.0, 0.75, 0.3, 10.0, None, "finite"),
        (100.0, float("inf"), 400.0, 0.75, 0.3, 10.0, None, "finite"),
        (100.0, 1000.0, float("nan"), 0.75, 0.3, 10.0, None, "finite"),
        (100.0, 1000.0, 400.0, float("inf"), 0.3, 10.0, None, "finite"),
        (100.0, 1000.0, 400.0, 0.75, float("nan"), 10.0, None, "finite"),
        (100.0, 1000.0, 400.0, 0.75, 0.3, float("inf"), None, "finite"),
        # schedule_time_s 非正
        (100.0, 1000.0, 0.0, 0.75, 0.3, 10.0, None, "positive or None"),
        (100.0, 1000.0, -10.0, 0.75, 0.3, 10.0, None, "positive or None"),
        # max_jerk_mps3 非正
        (100.0, 1000.0, 400.0, 0.0, 0.3, 10.0, None, "max_jerk_mps3"),
        (100.0, 1000.0, 400.0, -0.5, 0.3, 10.0, None, "max_jerk_mps3"),
        # max_stop_error_m 非正
        (100.0, 1000.0, 400.0, 0.75, 0.0, 10.0, None, "max_stop_error_m"),
        (100.0, 1000.0, 400.0, 0.75, -0.1, 10.0, None, "max_stop_error_m"),
        # max_arr_time_error_s 非正
        (100.0, 1000.0, 400.0, 0.75, 0.3, 0.0, None, "max_arr_time_error_s"),
        (100.0, 1000.0, 400.0, 0.75, 0.3, -1.0, None, "max_arr_time_error_s"),
        # schedule_change 时 schedule_time_s 不能为 None
        (
            100.0,
            1000.0,
            None,
            0.75,
            0.3,
            10.0,
            ScheduleChange(500.0, 450.0),
            "schedule_time_s must not be None",
        ),
        # trigger_position_m < start_position_m
        (
            100.0,
            1000.0,
            400.0,
            0.75,
            0.3,
            10.0,
            ScheduleChange(50.0, 450.0),
            "must be in",
        ),
        # trigger_position_m >= target_position_m
        (
            100.0,
            1000.0,
            400.0,
            0.75,
            0.3,
            10.0,
            ScheduleChange(1000.0, 450.0),
            "must be in",
        ),
        (
            100.0,
            1000.0,
            400.0,
            0.75,
            0.3,
            10.0,
            ScheduleChange(1200.0, 450.0),
            "must be in",
        ),
    ],
)
def test_task_invalid(
    start: float,
    target: float,
    sched_time: float | None,
    max_acc: float,
    max_stop: float,
    max_arr: float,
    change: ScheduleChange | None,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        Task(
            start_position_m=start,
            target_position_m=target,
            schedule_time_s=sched_time,
            max_jerk_mps3=max_acc,
            max_stop_error_m=max_stop,
            max_arr_time_error_s=max_arr,
            schedule_change=change,
        )


def test_safeguard_parameters_valid():
    params = SafeguardParams(
        factor=0.99,
        step_delay_s=2.0,
        distance_step_m=1.0,
        position_error_m=1.0,
        speed_error_mps=0.1,
        traction_cutoff_delay_s=0.5,
        vortex_brake_delay_s=0.5,
        min_curve_position_offset_m=0.0,
        generation_danger_points_m=(540.0, 1000.0, 2000.0),
    )
    assert params.factor == 0.99
    assert len(params.generation_danger_points_m) == 3


@pytest.mark.parametrize(
    (
        "factor",
        "step_delay",
        "ds",
        "pos_err",
        "spd_err",
        "tc_delay",
        "vb_delay",
        "offset",
        "dps",
        "match",
    ),
    [
        # 非有限值
        (
            float("nan"),
            2.0,
            1.0,
            1.0,
            0.1,
            0.5,
            0.5,
            0.0,
            (540.0,),
            "finite",
        ),
        # factor 范围 (0, 1]
        (0.0, 2.0, 1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "factor"),
        (-0.1, 2.0, 1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "factor"),
        (1.01, 2.0, 1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "factor"),
        # step_delay_s > 0
        (0.99, 0.0, 1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "step_delay_s"),
        (0.99, -1.0, 1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "step_delay_s"),
        # distance_step_m > 0
        (0.99, 2.0, 0.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "distance_step_m"),
        (0.99, 2.0, -1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (540.0,), "distance_step_m"),
        # position_error_m >= 0
        (
            0.99,
            2.0,
            1.0,
            -0.1,
            0.1,
            0.5,
            0.5,
            0.0,
            (540.0,),
            "position_error_m",
        ),
        # speed_error_mps >= 0
        (
            0.99,
            2.0,
            1.0,
            1.0,
            -0.1,
            0.5,
            0.5,
            0.0,
            (540.0,),
            "speed_error_mps",
        ),
        # traction_cutoff_delay_s >= 0
        (
            0.99,
            2.0,
            1.0,
            1.0,
            0.1,
            -0.5,
            0.5,
            0.0,
            (540.0,),
            "traction_cutoff_delay_s",
        ),
        # vortex_brake_delay_s >= 0
        (
            0.99,
            2.0,
            1.0,
            1.0,
            0.1,
            0.5,
            -0.5,
            0.0,
            (540.0,),
            "vortex_brake_delay_s",
        ),
        # generation_danger_points_m 空
        (0.99, 2.0, 1.0, 1.0, 0.1, 0.5, 0.5, 0.0, (), "empty"),
        # generation_danger_points_m 非有限值
        (
            0.99,
            2.0,
            1.0,
            1.0,
            0.1,
            0.5,
            0.5,
            0.0,
            (float("nan"),),
            "finite",
        ),
        # generation_danger_points_m 非严格递增
        (
            0.99,
            2.0,
            1.0,
            1.0,
            0.1,
            0.5,
            0.5,
            0.0,
            (540.0, 540.0),
            "strictly increasing",
        ),
        (
            0.99,
            2.0,
            1.0,
            1.0,
            0.1,
            0.5,
            0.5,
            0.0,
            (1000.0, 540.0),
            "strictly increasing",
        ),
    ],
)
def test_safeguard_parameters_invalid(
    factor: float,
    step_delay: float,
    ds: float,
    pos_err: float,
    spd_err: float,
    tc_delay: float,
    vb_delay: float,
    offset: float,
    dps: tuple[float, ...],
    match: str,
):
    with pytest.raises(ValueError, match=match):
        SafeguardParams(
            factor=factor,
            step_delay_s=step_delay,
            distance_step_m=ds,
            position_error_m=pos_err,
            speed_error_mps=spd_err,
            traction_cutoff_delay_s=tc_delay,
            vortex_brake_delay_s=vb_delay,
            min_curve_position_offset_m=offset,
            generation_danger_points_m=dps,
        )


# -----------------------------------------------------------------------------
# 2. Task.stop_state 状态与阈值边界测试
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_task() -> Task:
    return Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=100.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,  # 30 * 0.3 = 9.0
        max_arr_time_error_s=10.0,
    )


@pytest.mark.parametrize(
    ("position", "speed", "expected_state"),
    [
        # 四种状态常规用例
        (995.0, 0.005, StopState.STOPPED_IN_ZONE),  # 误差 5.0 <= 9.0, 速度 <= 0.01
        (980.0, 0.005, StopState.STOPPED_SHORT),  # 误差 20.0 > 9.0, 速度 <= 0.01
        (1000.0, 2.0, StopState.MOVING),  # 带速到达目标点：继续运行，不再判失败
        (900.0, 2.0, StopState.MOVING),  # 误差 100 > 1e-6, 速度 > 0.01
        # 边界 1: 速度 0.01 阈值两侧 (误差 5.0 <= 9.0)
        (995.0, 0.01, StopState.STOPPED_IN_ZONE),  # speed <= 0.01
        (995.0, 0.01001, StopState.MOVING),  # speed > 0.01 且 误差 > 1e-6
        # 边界 1: 速度 0.01 阈值两侧 (误差 20.0 > 9.0)
        (980.0, 0.01, StopState.STOPPED_SHORT),  # speed <= 0.01
        (980.0, 0.01001, StopState.MOVING),  # speed > 0.01 且 误差 > 1e-6
        # 边界 2: 停车误差 30 * max_stop_error_m = 9.0 两侧 (速度 <= 0.01)
        (991.0, 0.005, StopState.STOPPED_IN_ZONE),  # 误差 = 9.0 <= 9.0
        (990.999, 0.005, StopState.STOPPED_SHORT),  # 误差 = 9.001 > 9.0
        (1009.0, 0.005, StopState.STOPPED_IN_ZONE),  # 误差 = 9.0 <= 9.0
        (1009.001, 0.005, StopState.OVERRAN),  # 越过目标点 9.001 > 9.0 后停车
        # 边界 3: 越过目标点的距离 9.0 两侧 (速度 > 0.01)
        (1005.0, 2.0, StopState.MOVING),  # 越过 5.0，仍可在停车区内停下
        (1009.0, 2.0, StopState.MOVING),  # 越过 = 9.0
        (1009.001, 2.0, StopState.OVERRAN),  # 越过 9.001 > 9.0，冲出停车区
    ],
)
def test_task_stop_state(
    sample_task: Task, position: float, speed: float, expected_state: StopState
):
    assert sample_task.stop_state(position, speed) == expected_state


# -----------------------------------------------------------------------------
# 3. Task.final_schedule_time 测试
# -----------------------------------------------------------------------------


def test_final_schedule_time_no_schedule_time():
    task = Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=None,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    assert (
        task.final_schedule_time(np.array([0.0, 500.0, 1000.0], dtype=np.float64))
        is None
    )


def test_final_schedule_time_no_schedule_change():
    task = Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=100.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=None,
    )
    assert (
        task.final_schedule_time(np.array([0.0, 500.0, 1000.0], dtype=np.float64))
        == 100.0
    )


def test_final_schedule_time_start_trigger():
    task = Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=100.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=ScheduleChange(
            trigger_position_m=0.0, new_schedule_time_s=120.0
        ),
    )
    # 起点触发，节点 0 为非末节点
    assert (
        task.final_schedule_time(np.array([0.0, 500.0, 1000.0], dtype=np.float64))
        == 120.0
    )


def test_final_schedule_time_non_end_node_reached():
    task = Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=100.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=ScheduleChange(
            trigger_position_m=500.0, new_schedule_time_s=120.0
        ),
    )
    # 在节点 1 到达 500.0 (非末节点)
    assert (
        task.final_schedule_time(
            np.array([0.0, 500.0, 800.0, 1000.0], dtype=np.float64)
        )
        == 120.0
    )


def test_final_schedule_time_only_at_end_node_not_effective():
    task = Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=100.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=ScheduleChange(
            trigger_position_m=500.0, new_schedule_time_s=120.0
        ),
    )
    # 只在末节点达到 500.0，不生效
    assert task.final_schedule_time(np.array([0.0, 500.0], dtype=np.float64)) == 100.0


def test_final_schedule_time_not_reached():
    task = Task(
        start_position_m=0.0,
        target_position_m=1000.0,
        schedule_time_s=100.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
        schedule_change=ScheduleChange(
            trigger_position_m=800.0, new_schedule_time_s=120.0
        ),
    )
    assert (
        task.final_schedule_time(np.array([0.0, 200.0, 400.0, 600.0], dtype=np.float64))
        == 100.0
    )


# -----------------------------------------------------------------------------
# 4. load_scenario 与 load_tasks 测试
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_scenario():
    spec_path = Path("paper/specs/scenario.toml")
    line_dir = Path("paper/data/line")
    return load_scenario(spec_path, line_dir)


@pytest.fixture(scope="module")
def loaded_tasks():
    spec_path = Path("paper/specs/tasks.toml")
    return load_tasks(spec_path)


def test_paper_tasks_use_the_jerk_threshold(loaded_tasks, tmp_path: Path):
    assert loaded_tasks["longyang_to_airport"].max_jerk_mps3 == 0.75
    # A specification still carrying the former per-step acceleration-change
    # key under its old name lacks the jerk threshold and is rejected.
    legacy = Path("paper/specs/tasks.toml").read_text(encoding="utf-8")
    legacy_path = tmp_path / "legacy_tasks.toml"
    legacy_path.write_text(
        legacy.replace("max_jerk_mps3", "max_acceleration_change"), encoding="utf-8"
    )
    with pytest.raises(KeyError, match="max_jerk_mps3"):
        load_tasks(legacy_path)


def test_safeguard_curves_reproducibility():
    spec_path = Path("paper/specs/scenario.toml")
    line_dir = Path("paper/data/line")
    s1 = load_scenario(spec_path, line_dir)
    s2 = load_scenario(spec_path, line_dir)

    assert len(s1.safeguard.min_curves) == 9
    assert len(s1.safeguard.max_curves) == 10
    assert len(s2.safeguard.min_curves) == 9
    assert len(s2.safeguard.max_curves) == 10

    for c1, c2 in zip(s1.safeguard.min_curves, s2.safeguard.min_curves, strict=True):
        assert np.array_equal(c1, c2)
    for c1, c2 in zip(s1.safeguard.max_curves, s2.safeguard.max_curves, strict=True):
        assert np.array_equal(c1, c2)


def test_paper_scenario_danger_points_assembly(loaded_scenario):
    accel_path = Path("paper/data/line/acceleration_zones.json")
    with accel_path.open("r", encoding="utf-8") as f:
        accel_data = json.load(f)
    expected_accel_end = float(accel_data["uplink"]["end"])

    gen_dps = loaded_scenario.safeguard.params.generation_danger_points_m
    track_dps = tuple(loaded_scenario.line.danger_points_m)

    assert len(gen_dps) == 10
    assert gen_dps[0] == expected_accel_end
    assert gen_dps[1:] == track_dps
    assert len(loaded_scenario.line.danger_points_m) == 9


def test_scenario_without_prepend_acceleration_zone_end_rejected(tmp_path: Path):
    spec_content = Path("paper/specs/scenario.toml").read_text(encoding="utf-8")
    modified_spec_content = spec_content.replace(
        "prepend_acceleration_zone_end = true",
        "prepend_acceleration_zone_end = false",
    )
    spec_path = tmp_path / "scenario_no_accel.toml"
    spec_path.write_text(modified_spec_content, encoding="utf-8")
    line_dir = Path("paper/data/line")

    expected_error = (
        r"Safeguard requires len\(max_curves\) == "
        r"len\(min_curves\) \+ 1, got len\(max_curves\)=9 "
        r"and len\(min_curves\)=9"
    )
    with pytest.raises(ValueError, match=expected_error):
        load_scenario(spec_path, line_dir)


def test_arrays_read_only(loaded_scenario):
    t = loaded_scenario.line
    assert not t.slopes.flags.writeable
    assert not t.slope_intervals.flags.writeable
    assert not t.speed_limits.flags.writeable
    assert not t.speed_limit_intervals.flags.writeable

    sg = loaded_scenario.safeguard
    assert not sg.speed_limits.flags.writeable
    assert not sg.speed_limit_intervals.flags.writeable
    assert not sg.min_pos_packed.flags.writeable
    assert not sg.min_speed_packed.flags.writeable
    assert not sg.min_lengths.flags.writeable
    assert not sg.max_pos_packed.flags.writeable
    assert not sg.max_speed_packed.flags.writeable
    assert not sg.max_lengths.flags.writeable
    assert not sg.static_region.idp_points_x.flags.writeable
    for curve in sg.levi_curves + sg.brake_curves + sg.min_curves + sg.max_curves:
        assert not curve.flags.writeable
    with pytest.raises(ValueError):
        sg.min_curves[0][0, 0] = 999.0


def test_scenario_hash_reproducibility_and_sensitivity(loaded_scenario):
    spec_path = Path("paper/specs/scenario.toml")
    line_dir = Path("paper/data/line")
    second = load_scenario(spec_path, line_dir)
    assert loaded_scenario.scenario_hash == second.scenario_hash

    # 修改 vehicle
    mod_veh = dataclasses.replace(
        loaded_scenario.vehicle,
        mass=loaded_scenario.vehicle.mass + 1.0,
    )
    hash_veh = compute_scenario_hash(
        line=loaded_scenario.line,
        vehicle=mod_veh,
        energy=loaded_scenario.energy,
        safeguard=loaded_scenario.safeguard,
        prepend_acceleration_zone_end=True,
    )
    assert hash_veh != loaded_scenario.scenario_hash

    # 修改 energy
    mod_energy = EnergyParams(
        R_m=loaded_scenario.energy.R_m + 0.01,
        L_d=loaded_scenario.energy.L_d,
        R_k=loaded_scenario.energy.R_k,
        L_k=loaded_scenario.energy.L_k,
        Tau=loaded_scenario.energy.Tau,
        Psi_fd=loaded_scenario.energy.Psi_fd,
        k_c=loaded_scenario.energy.k_c,
        Phi_1=loaded_scenario.energy.Phi_1,
        Phi_2=loaded_scenario.energy.Phi_2,
    )
    hash_energy = compute_scenario_hash(
        line=loaded_scenario.line,
        vehicle=loaded_scenario.vehicle,
        energy=mod_energy,
        safeguard=loaded_scenario.safeguard,
        prepend_acceleration_zone_end=True,
    )
    assert hash_energy != loaded_scenario.scenario_hash

    # 修改 safeguard params
    mod_params = dataclasses.replace(loaded_scenario.safeguard.params, factor=0.95)
    mod_sg = dataclasses.replace(loaded_scenario.safeguard, params=mod_params)
    hash_sg = compute_scenario_hash(
        line=loaded_scenario.line,
        vehicle=loaded_scenario.vehicle,
        energy=loaded_scenario.energy,
        safeguard=mod_sg,
        prepend_acceleration_zone_end=True,
    )
    assert hash_sg != loaded_scenario.scenario_hash


# -----------------------------------------------------------------------------
# 5. Safeguard 与 SPS 长度校验反例
# -----------------------------------------------------------------------------


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
    slopes = np.array([0.0], dtype=np.float64)
    slope_intervals = np.array([0.0, 500.0], dtype=np.float64)
    speed_limits = np.array([100.0], dtype=np.float64)
    speed_limit_intervals = np.array([0.0, 500.0], dtype=np.float64)
    return Line(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        accessible_points_m=(100.0,),
        danger_points_m=(110.0,),
    )


def test_safeguard_length_validation():
    dummy_levi = [_make_dummy_curve(0.0, 100.0)]
    dummy_brake = [_make_dummy_curve(0.0, 100.0), _make_dummy_curve(100.0, 200.0)]
    dummy_min = [_make_dummy_curve(0.0, 100.0)]
    # 期望 len(max) == len(min) + 1 = 2
    # 反例 1: len(max) == 1 == len(min)
    with pytest.raises(ValueError, match="Safeguard requires"):
        build_safeguard(
            params=_make_dummy_params(),
            line=_make_dummy_line(),
            levi_curves=dummy_levi,
            brake_curves=dummy_brake,
            min_curves=dummy_min,
            max_curves=[_make_dummy_curve(0.0, 100.0)],
        )

    # 反例 2: len(max) == 3 != len(min) + 1
    with pytest.raises(ValueError, match="Safeguard requires"):
        build_safeguard(
            params=_make_dummy_params(),
            line=_make_dummy_line(),
            levi_curves=dummy_levi,
            brake_curves=dummy_brake,
            min_curves=dummy_min,
            max_curves=[
                _make_dummy_curve(0.0, 100.0),
                _make_dummy_curve(100.0, 200.0),
                _make_dummy_curve(200.0, 300.0),
            ],
        )


def test_sps_length_validation():
    # 构造合法的 Safeguard (min 1 条, max 2 条, 且 min[0] 与 max[1] 相交)
    valid_sg = build_safeguard(
        params=_make_dummy_params(),
        line=_make_dummy_line(),
        levi_curves=[_make_dummy_curve(0.0, 100.0)],
        brake_curves=[
            _make_dummy_curve(0.0, 100.0),
            _make_dummy_curve(0.0, 100.0),
        ],
        min_curves=[_make_dummy_curve(0.0, 100.0, 0.0, 50.0)],
        max_curves=[
            _make_dummy_curve(0.0, 100.0, 50.0, 0.0),
            _make_dummy_curve(0.0, 100.0, 50.0, 0.0),
        ],
    )
    # len(max)=2 == len(min)+1=2 == len(accessible)+1. 合法 accessible 应为 1 个点
    # 反例: accessible 为 2 个点 (len(accessible)=2, 但 len(max)=2 != 3)
    with pytest.raises(ValueError, match="SPS requires"):
        SPS(
            safeguard=valid_sg,
            accessible_positions_m=[100.0, 200.0],
            danger_positions_m=[110.0, 210.0],
            step_delay_s=2.0,
        )
