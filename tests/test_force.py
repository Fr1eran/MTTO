import numpy as np
import pytest

from model.force.numpy.brake import (
    sledge_frictional_brake_force,
    vortex_brake_force,
    wear_plate_frictional_brake_force,
)
from model.force.numpy.resis import (
    air_resis_force,
    guideway_vortex_resis_force,
    linear_generator_resis_force,
    slope_resis_force,
)
from model.vehicle.vehicle import (
    calc_brake_deceleration,
    calc_brake_deceleration_scalar_numba,
    calc_levi_deceleration,
    calc_levi_deceleration_scalar_numba,
    calc_longitudinal_force,
    calc_longitudinal_force_scalar_numba,
)


@pytest.fixture
def v_sample():
    return np.arange(0.0, 600.0, 0.01)


def test_total_resis_force_shape(v_sample):
    total_resis_force = (
        air_resis_force(v_sample, 5)
        + guideway_vortex_resis_force(v_sample, 5)
        + linear_generator_resis_force(v_sample, 5)
        + sledge_frictional_brake_force(v_sample, 4.35, 0, 1)
        + slope_resis_force(4.35, 0)
        + vortex_brake_force(v_sample, 5, 0)
        + wear_plate_frictional_brake_force(v_sample, 5)
    )
    # 检查输出长度是否与输入速度样本一致
    assert total_resis_force.shape == v_sample.shape


def test_total_resis_force_nonnegative(v_sample):
    total_resis_force = (
        air_resis_force(v_sample, 5)
        + guideway_vortex_resis_force(v_sample, 5)
        + linear_generator_resis_force(v_sample, 5)
        + sledge_frictional_brake_force(v_sample, 4.35, 0, 1)
        + slope_resis_force(4.35, 0)
        + vortex_brake_force(v_sample, 5, 0)
        + wear_plate_frictional_brake_force(v_sample, 5)
    )
    # 检查总阻力是否全部为非负
    assert np.all(total_resis_force >= 0)


def test_vehicle_module_scalar_interfaces_are_consistent():
    mass = 317.5
    numoftrainsets = 5
    speed = 20.0
    slope = 1.0
    acc = 0.2
    level = 0

    levi_np = calc_levi_deceleration(
        mass=mass,
        numoftrainsets=numoftrainsets,
        speed=speed,
        slope=slope,
    )
    levi_nb = calc_levi_deceleration_scalar_numba(
        speed, slope, mass, numoftrainsets
    )
    brake_np = calc_brake_deceleration(
        mass=mass,
        numoftrainsets=numoftrainsets,
        speed=speed,
        slope=slope,
        level=level,
    )
    brake_nb = calc_brake_deceleration_scalar_numba(
        speed, slope, mass, numoftrainsets, level
    )
    longitudinal_np = calc_longitudinal_force(
        mass=mass,
        numoftrainsets=numoftrainsets,
        acc=acc,
        speed=speed,
        slope=slope,
    )
    longitudinal_nb = calc_longitudinal_force_scalar_numba(
        speed, slope, acc, mass, numoftrainsets
    )

    assert levi_nb == pytest.approx(float(levi_np), rel=1e-10, abs=1e-10)
    assert brake_nb == pytest.approx(float(brake_np), rel=1e-10, abs=1e-10)
    assert longitudinal_nb == pytest.approx(
        float(longitudinal_np), rel=1e-10, abs=1e-10
    )


def test_vehicle_module_vectorized_interfaces_shape():
    mass = 317.5
    numoftrainsets = 5
    speed = np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    slope = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    acc = np.asarray([0.2, 0.2, 0.2, 0.2], dtype=np.float64)

    levi = calc_levi_deceleration(
        mass=mass,
        numoftrainsets=numoftrainsets,
        speed=speed,
        slope=slope,
    )
    brake = calc_brake_deceleration(
        mass=mass,
        numoftrainsets=numoftrainsets,
        speed=speed,
        slope=slope,
        level=0,
    )
    longitudinal = calc_longitudinal_force(
        mass=mass,
        numoftrainsets=numoftrainsets,
        acc=acc,
        speed=speed,
        slope=slope,
    )

    assert levi.shape == speed.shape
    assert brake.shape == speed.shape
    assert longitudinal.shape == speed.shape
