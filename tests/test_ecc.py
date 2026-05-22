import numpy as np
import pytest
from scipy.integrate import trapezoid

from model.common import ECC
from model.track import TrackInfo, get_slope
from model.vehicle import VehicleInfo, calc_longitudinal_force


@pytest.fixture(scope="module")
def energy_consumption_calculator_case():
    track = TrackInfo(
        slopes=np.asarray([0.0, 0.8, -0.4], dtype=np.float64),
        slope_intervals=np.asarray([0.0, 500.0, 1000.0, 20000.0], dtype=np.float64),
        speed_limits=np.asarray([120.0 / 3.6], dtype=np.float64),
        speed_limit_intervals=np.asarray([0.0, 20000.0], dtype=np.float64),
    )
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    ecc = ECC(
        R_m=0.2796,
        L_d=0.0002,
        R_k=50.0,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.8,
    )
    return ecc, vehicle, track


def _calc_energy_constant_acc_reference(
    ecc: ECC,
    vehicle: VehicleInfo,
    track: TrackInfo,
    *,
    begin_pos: float,
    begin_speed: float,
    acc: float,
    distance: float,
    direction: int,
    operation_time: float | None,
) -> tuple[float, float]:
    if np.abs(distance) < 1e-6:
        f_longitudinal = calc_longitudinal_force(
            mass=vehicle.mass,
            numoftrainsets=vehicle.numoftrainsets,
            acc=acc,
            speed=begin_speed,
            slope=get_slope(
                begin_pos,
                track.slopes,
                track.slope_intervals,
                dtype=np.float64,
            ),
        )
        mechanic_energy_consumption = np.abs(f_longitudinal * distance)
        motor_energy_consumption = 0.0
    else:
        n_samples = max(10, int(np.abs(distance) / 1.0))
        d_nodes = np.linspace(0.0, distance, n_samples + 1)
        delta_d = np.diff(d_nodes)
        p_nodes = begin_pos + d_nodes * direction
        acc_nodes = np.full_like(d_nodes, acc, dtype=np.float64)

        speed_nodes = np.empty_like(d_nodes)
        speed_nodes[0] = begin_speed
        for i in range(n_samples):
            next_speed_squared = speed_nodes[i] ** 2 + 2.0 * acc_nodes[i] * delta_d[i]
            speed_nodes[i + 1] = np.sqrt(np.maximum(next_speed_squared, 0.0))

        t_nodes = np.zeros_like(d_nodes)
        for i in range(n_samples):
            avg_speed = np.maximum(
                (speed_nodes[i] + speed_nodes[i + 1]) / 2.0, 1e-6
            )
            t_nodes[i + 1] = t_nodes[i] + np.abs(delta_d[i]) / avg_speed

        f_longitudinal = calc_longitudinal_force(
            mass=vehicle.mass,
            numoftrainsets=vehicle.numoftrainsets,
            acc=acc_nodes,
            speed=speed_nodes,
            slope=get_slope(
                p_nodes,
                track.slopes,
                track.slope_intervals,
                dtype=np.float64,
            ),
        )
        mechanic_energy_consumption = np.sum(
            0.5
            * (np.abs(f_longitudinal[:-1]) + np.abs(f_longitudinal[1:]))
            * np.abs(delta_d)
        )
        motor_energy_consumption = trapezoid(
            y=(2 * f_longitudinal**2 / (3 * ecc.h**2))
            * (ecc.R_m + ecc.k_c**2 * ecc.R_k + (1 - ecc.k_c) ** 2 * ecc.R_k),
            x=t_nodes,
        ) + trapezoid(
            y=(np.abs(f_longitudinal) * 2 / (3 * ecc.h**2))
            * (ecc.L_d + ecc.k_c**2 * ecc.L_k + (1 - ecc.k_c) ** 2 * ecc.L_k),
            x=np.abs(f_longitudinal),
        )

    if operation_time is None:
        if np.abs(acc) < 1e-9:
            time = distance / np.maximum(begin_speed, 1e-6)
        else:
            next_speed_squared = begin_speed**2 + 2 * acc * distance
            next_speed = np.sqrt(np.maximum(next_speed_squared, 0))
            time = (next_speed - begin_speed) / acc
    else:
        time = operation_time

    propulsion_energy_consumption = (
        mechanic_energy_consumption + motor_energy_consumption
    )
    leviation_energy_consumption = (
        ecc.Phi_1 * distance + ecc.Phi_2 * vehicle.mass * time
    )
    return float(propulsion_energy_consumption), float(leviation_energy_consumption)


@pytest.mark.parametrize(
    ("begin_pos", "begin_speed", "acc", "distance", "direction", "operation_time"),
    [
        (100.0, 10.0, 0.35, 200.0, 1, None),
        (1200.0, 18.0, -0.2, 350.0, -1, None),
        (500.0, 12.0, 0.25, 100.0, 1, 12.34),
        (200.0, 7.5, 0.2, 1e-8, 1, None),
    ],
)
def test_calc_energy_constant_acc_matches_reference(
    energy_consumption_calculator_case,
    begin_pos,
    begin_speed,
    acc,
    distance,
    direction,
    operation_time,
):
    ecc, vehicle, track = energy_consumption_calculator_case
    expected_pec, expected_lec = _calc_energy_constant_acc_reference(
        ecc,
        vehicle,
        track,
        begin_pos=begin_pos,
        begin_speed=begin_speed,
        acc=acc,
        distance=distance,
        direction=direction,
        operation_time=operation_time,
    )
    pec, lec = ecc.calc_energy(
        begin_pos=begin_pos,
        begin_speed=begin_speed,
        acc=acc,
        distance=distance,
        direction=direction,
        operation_time=operation_time,
        vehicle=vehicle,
        track=track,
    )

    assert np.isfinite(pec)
    assert np.isfinite(lec)
    assert pec == pytest.approx(expected_pec, rel=1e-10, abs=1e-10)
    assert lec == pytest.approx(expected_lec, rel=1e-10, abs=1e-10)


def test_calc_energy_rejects_callable_acc(energy_consumption_calculator_case):
    ecc, vehicle, track = energy_consumption_calculator_case

    with pytest.raises(TypeError, match="acc must be a float constant"):
        ecc.calc_energy(
            begin_pos=200.0,
            begin_speed=7.5,
            acc=lambda _: 0.2,  # type: ignore[arg-type]
            distance=80.0,
            direction=1,
            operation_time=None,
            vehicle=vehicle,
            track=track,
        )
