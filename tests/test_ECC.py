import numpy as np
import pytest

from model.ecc import ECC
from model.track import Track, TrackProfile
from model.vehicle import Vehicle


@pytest.fixture(scope="module")
def energy_consumption_calculator_case():
    track = Track(
        slopes=np.asarray([0.0], dtype=np.float64),
        slope_intervals=np.asarray([0.0, 20000.0], dtype=np.float64),
        speed_limits=np.asarray([120.0 / 3.6], dtype=np.float64),
        speed_limit_intervals=np.asarray([0.0, 20000.0], dtype=np.float64),
    )
    track_profile = TrackProfile(track)
    vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
    ecc = ECC(
        R_m=0.2796,
        L_d=0.0002,
        R_k=50.0,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.8,
    )
    return ecc, vehicle, track_profile


def test_calc_energy_constant_function_equivalent_to_scalar(
    energy_consumption_calculator_case,
):
    ecc, vehicle, track_profile = energy_consumption_calculator_case
    begin_pos = 100.0
    begin_speed = 10.0
    distance = 200.0
    scalar_acc = 0.35

    def const_acc_profile(s):
        s_arr = np.asarray(s, dtype=np.float64)
        return np.full_like(s_arr, scalar_acc, dtype=np.float64)

    scalar_pec, scalar_lec = ecc.calc_energy(
        begin_pos=begin_pos,
        begin_speed=begin_speed,
        acc=scalar_acc,
        distance=distance,
        direction=1,
        operation_time=None,
        vehicle=vehicle,
        trackprofile=track_profile,
    )
    func_pec, func_lec = ecc.calc_energy(
        begin_pos=begin_pos,
        begin_speed=begin_speed,
        acc=const_acc_profile,
        distance=distance,
        direction=1,
        operation_time=None,
        vehicle=vehicle,
        trackprofile=track_profile,
    )

    assert np.isfinite(func_pec)
    assert np.isfinite(func_lec)
    assert abs(func_pec - scalar_pec) / max(abs(scalar_pec), 1.0) < 2e-2
    assert abs(func_lec - scalar_lec) / max(abs(scalar_lec), 1.0) < 2e-2


def test_calc_energy_accepts_distance_dependent_acc(energy_consumption_calculator_case):
    ecc, vehicle, track_profile = energy_consumption_calculator_case

    def acc_profile(s):
        s_arr = np.asarray(s, dtype=np.float64)
        return 0.2 + 0.001 * s_arr

    pec, lec = ecc.calc_energy(
        begin_pos=1000.0,
        begin_speed=8.0,
        acc=acc_profile,
        distance=150.0,
        direction=1,
        operation_time=None,
        vehicle=vehicle,
        trackprofile=track_profile,
    )

    assert np.isfinite(pec)
    assert np.isfinite(lec)
    assert pec > 0.0
    assert lec > 0.0


def test_calc_energy_callable_respects_operation_time_override(
    energy_consumption_calculator_case,
):
    ecc, vehicle, track_profile = energy_consumption_calculator_case
    distance = 100.0
    forced_time = 12.34

    def acc_profile(s):
        s_arr = np.asarray(s, dtype=np.float64)
        return 0.25 + 0.0002 * s_arr

    _, lec = ecc.calc_energy(
        begin_pos=500.0,
        begin_speed=12.0,
        acc=acc_profile,
        distance=distance,
        direction=1,
        operation_time=forced_time,
        vehicle=vehicle,
        trackprofile=track_profile,
    )

    expected_lec = ecc.Phi_1 * distance + ecc.Phi_2 * vehicle.mass * forced_time
    assert lec == pytest.approx(expected_lec, rel=1e-8, abs=1e-8)


def test_calc_energy_accepts_array_only_callback(energy_consumption_calculator_case):
    ecc, vehicle, track_profile = energy_consumption_calculator_case

    def acc_profile_array_only(s):
        assert isinstance(s, np.ndarray)
        return 0.15 + 0.0005 * s

    pec, lec = ecc.calc_energy(
        begin_pos=200.0,
        begin_speed=7.5,
        acc=acc_profile_array_only,
        distance=80.0,
        direction=1,
        operation_time=None,
        vehicle=vehicle,
        trackprofile=track_profile,
    )

    assert np.isfinite(pec)
    assert np.isfinite(lec)


def test_calc_energy_array_only_callback_with_tiny_distance(
    energy_consumption_calculator_case,
):
    ecc, vehicle, track_profile = energy_consumption_calculator_case

    def acc_profile_array_only(s):
        assert isinstance(s, np.ndarray)
        return np.full_like(s, 0.2, dtype=np.float64)

    pec, lec = ecc.calc_energy(
        begin_pos=200.0,
        begin_speed=7.5,
        acc=acc_profile_array_only,
        distance=1e-8,
        direction=1,
        operation_time=None,
        vehicle=vehicle,
        trackprofile=track_profile,
    )

    assert np.isfinite(pec)
    assert np.isfinite(lec)


def test_calc_energy_rejects_non_vectorized_callback(
    energy_consumption_calculator_case,
):
    ecc, vehicle, track_profile = energy_consumption_calculator_case

    def scalar_only_callback(s):
        if not isinstance(s, (float, np.floating)):
            raise TypeError("scalar input only")
        return 0.2

    with pytest.raises(TypeError, match="vectorized callback required"):
        ecc.calc_energy(
            begin_pos=200.0,
            begin_speed=7.5,
            acc=scalar_only_callback,
            distance=80.0,
            direction=1,
            operation_time=None,
            vehicle=vehicle,
            trackprofile=track_profile,
        )


def test_calc_energy_rejects_scalar_output_callback(energy_consumption_calculator_case):
    ecc, vehicle, track_profile = energy_consumption_calculator_case

    def scalar_output_callback(s):
        assert isinstance(s, np.ndarray)
        return 0.2

    with pytest.raises(ValueError, match="same ndim and shape"):
        ecc.calc_energy(
            begin_pos=200.0,
            begin_speed=7.5,
            acc=scalar_output_callback,
            distance=80.0,
            direction=1,
            operation_time=None,
            vehicle=vehicle,
            trackprofile=track_profile,
        )
