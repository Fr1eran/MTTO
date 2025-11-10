import numpy as np
import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.force.resis import AirResis, GuidewayVortexResis, LinearGeneResis, SlopeResis
from model.force.brake import (
    SledgeFrictionalBrake,
    VortexBrake,
    WearPlateFrictionalBrake,
)


@pytest.fixture
def v_sample():
    return np.arange(0.0, 600.0, 0.01)


def test_total_resis_force_shape(v_sample):
    total_resis_force = (
        AirResis(v_sample, 5)
        + GuidewayVortexResis(v_sample, 5)
        + LinearGeneResis(v_sample, 5)
        + SledgeFrictionalBrake(v_sample, 4.35, 0, 1)
        + SlopeResis(4.35, 0)
        + VortexBrake(v_sample, 5, 0)
        + WearPlateFrictionalBrake(v_sample, 5)
    )
    # 检查输出长度是否与输入速度样本一致
    assert total_resis_force.shape == v_sample.shape


def test_total_resis_force_nonnegative(v_sample):
    total_resis_force = (
        AirResis(v_sample, 5)
        + GuidewayVortexResis(v_sample, 5)
        + LinearGeneResis(v_sample, 5)
        + SledgeFrictionalBrake(v_sample, 4.35, 0, 1)
        + SlopeResis(4.35, 0)
        + VortexBrake(v_sample, 5, 0)
        + WearPlateFrictionalBrake(v_sample, 5)
    )
    # 检查总阻力是否全部为非负
    assert np.all(total_resis_force >= 0)
