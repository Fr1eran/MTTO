import numpy as np
import pytest

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import EnergyParams, segment_energy
from mtto.domain.kinematics import accel_between, run_time
from mtto.domain.line import Line
from mtto.domain.safeguard import Safeguard, SafeguardParams, StaticRegion
from mtto.domain.srtsp import build_srtsp_curve
from mtto.dp.graph import _calculate_transition_with_context
from paper.figures import load_paper_scenario


@pytest.fixture(scope="module")
def physics_context() -> tuple[EnergyParams, Vehicle, Line]:
    line = Line(
        slopes=np.asarray([0.0, 0.01, -0.01], dtype=np.float64),
        slope_intervals=np.asarray([0.0, 500.0, 1000.0, 5000.0], dtype=np.float64),
        speed_limits=np.asarray([100.0 / 3.6], dtype=np.float64),
        speed_limit_intervals=np.asarray([0.0, 5000.0], dtype=np.float64),
        accessible_points_m=(),
        danger_points_m=(),
    )
    vehicle = Vehicle(
        mass=317.5,
        numoftrainsets=5,
        length=128.5,
        max_speed=500.0 / 3.6,
        max_acc=1.0,
        max_dec=-1.0,
        max_slope_capacity=4.0,
        levi_power_per_mass=1.7,
    )
    energy_params = EnergyParams(
        R_m=0.2796,
        L_d=0.00292,
        R_k=0.0736,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.5,
        Phi_1=0.1049,
        Phi_2=1.006,
    )
    return energy_params, vehicle, line


@pytest.mark.parametrize(
    ("s", "v0", "a", "dt"),
    [
        # 加速
        (100.0, 10.0, 0.5, 1.0),
        (250.0, 15.0, 0.8, 2.0),
        # 减速（正常行进不提前停车）
        (200.0, 20.0, -0.4, 1.5),
        (600.0, 25.0, -0.6, 2.0),
        # 较小但大于阈值的加速度
        (300.0, 12.0, 0.01, 0.5),
        (400.0, 18.0, -0.01, 1.0),
    ],
)
def test_dp_edge_and_rl_step_consistency(
    physics_context: tuple[EnergyParams, Vehicle, Line],
    s: float,
    v0: float,
    a: float,
    dt: float,
) -> None:
    energy_params, vehicle, line = physics_context

    # 1. RL 单步运动学
    v1, d, t = run_time(v0, a, dt)

    # 2. DP 边反算加速度与时长
    a_inferred, t_inferred = accel_between(v0, v1, d)

    assert a_inferred == pytest.approx(a, rel=1e-9)
    assert t_inferred == pytest.approx(t, rel=1e-9)

    # 3. 两条路径用 segment_energy 计算的能耗一致
    e_rl = segment_energy(
        energy_params,
        vehicle,
        line,
        begin_pos=s,
        begin_speed=v0,
        acc=a,
        distance=d,
        direction=1,
        operation_time=t,
    )
    e_dp = segment_energy(
        energy_params,
        vehicle,
        line,
        begin_pos=s,
        begin_speed=v0,
        acc=a_inferred,
        distance=d,
        direction=1,
        operation_time=t_inferred,
    )

    assert e_rl[0] == pytest.approx(e_dp[0], rel=1e-9)
    assert e_rl[1] == pytest.approx(e_dp[1], rel=1e-9)


def _make_dummy_safeguard() -> Safeguard:
    return Safeguard(
        params=SafeguardParams(
            factor=1.0,
            step_delay_s=1.0,
            distance_step_m=1.0,
            position_error_m=0.0,
            speed_error_mps=0.0,
            traction_cutoff_delay_s=0.1,
            vortex_brake_delay_s=0.1,
            min_curve_position_offset_m=0.0,
            generation_danger_points_m=(1010.0,),
        ),
        speed_limits=np.array([1000.0], dtype=np.float64),
        speed_limit_intervals=np.array([0.0], dtype=np.float64),
        levi_curves=(),
        brake_curves=(),
        min_curves=(),
        max_curves=(),
        min_pos_packed=np.empty((0, 0), dtype=np.float64),
        min_speed_packed=np.empty((0, 0), dtype=np.float64),
        min_lengths=np.zeros(0, dtype=np.int32),
        max_pos_packed=np.empty((0, 0), dtype=np.float64),
        max_speed_packed=np.empty((0, 0), dtype=np.float64),
        max_lengths=np.zeros(0, dtype=np.int32),
        static_region=StaticRegion(
            idp_points_x=np.empty(0, dtype=np.float64),
            min_curves_part_x_padded=(),
            min_curves_part_y_padded=(),
            max_curves_part_x_padded=(),
            max_curves_part_y_padded=(),
            num_regions=0,
        ),
    )


@pytest.mark.parametrize(
    ("begin_speed", "acceleration", "dt", "expected"),
    [
        # 惰行 (|a| < 1e-6)
        (10.0, 0.0, 3.0, (10.0, 30.0, 3.0)),
        (15.0, 1e-7, 2.0, (15.0, 30.0, 2.0)),
        # 静止且不牵引：零位移步
        (0.0, 0.0, 1.0, (0.0, 0.0, 0.0)),
        (0.0, -1.0, 1.0, (0.0, 0.0, 0.0)),
        # 从静止起步
        (0.0, 1.0, 2.0, (2.0, 2.0, 2.0)),
        # 加速、未停车的减速
        (10.0, 1.0, 2.0, (12.0, 22.0, 2.0)),
        (20.0, -0.5, 1.5, (19.25, 29.4375, 1.5)),
        # 步内停车：截到停车时刻，停车点为 v^2 / (2|a|)
        (10.0, -2.0, 30.0, (0.0, 25.0, 5.0)),
        (8.0, -1.0, 8.0, (0.0, 32.0, 8.0)),
    ],
)
def test_run_time(
    begin_speed: float,
    acceleration: float,
    dt: float,
    expected: tuple[float, float, float],
) -> None:
    assert run_time(begin_speed, acceleration, dt) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("begin_speed", "end_speed", "distance", "expected"),
    [
        (10.0, 10.0, 30.0, (0.0, 3.0)),
        (10.0, 12.0, 30.0, (44.0 / 60.0, 30.0 / 11.0)),
        (12.0, 10.0, 30.0, (-44.0 / 60.0, 30.0 / 11.0)),
    ],
)
def test_accel_between(
    begin_speed: float,
    end_speed: float,
    distance: float,
    expected: tuple[float, float],
) -> None:
    actual = accel_between(
        begin_speed,
        end_speed,
        distance,
    )

    assert actual == pytest.approx(expected)


def test_dp_transition_uses_shared_end_speed_transition() -> None:
    scenario = load_paper_scenario()
    srtsp_curve = build_srtsp_curve([0.0, 30.0], [100.0, 100.0])

    transition = _calculate_transition_with_context(
        pos_k=0.0,
        speed_k=10.0,
        displacement=30.0,
        speed_k_1=12.0,
        vehicle=scenario.vehicle,
        safeguard=_make_dummy_safeguard(),
        energy=scenario.energy,
        track=scenario.line,
        srtsp_curve=srtsp_curve,
    )

    assert transition is not None
    energy, duration = transition

    expected_acc = 44.0 / 60.0
    expected_duration = 30.0 / 11.0
    expected_prop, expected_levi = segment_energy(
        scenario.energy,
        scenario.vehicle,
        scenario.line,
        begin_pos=0.0,
        begin_speed=10.0,
        acc=expected_acc,
        distance=30.0,
        direction=1,
        operation_time=expected_duration,
    )
    assert energy == pytest.approx(expected_prop + expected_levi)
    assert duration == pytest.approx(expected_duration)


def test_dp_transition_rejects_sample_above_minimum_time_upper_curve() -> None:
    scenario = load_paper_scenario()
    srtsp_curve = build_srtsp_curve(
        [0.0, 10.0, 20.0, 30.0],
        [100.0, 10.0, 10.0, 100.0],
    )

    transition = _calculate_transition_with_context(
        pos_k=0.0,
        speed_k=10.0,
        displacement=30.0,
        speed_k_1=12.0,
        vehicle=scenario.vehicle,
        safeguard=_make_dummy_safeguard(),
        energy=scenario.energy,
        track=scenario.line,
        srtsp_curve=srtsp_curve,
    )

    assert transition is None
