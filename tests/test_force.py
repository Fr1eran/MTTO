import numpy as np
import pytest

from model.force.resis import (
    air_resis_force,
    guideway_vortex_resis_force,
    linear_generator_resis_force,
    slope_resis_force,
)
from model.force.brake import (
    sledge_frictional_brake_force,
    vortex_brake_force,
    wear_plate_frictional_brake_force,
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
