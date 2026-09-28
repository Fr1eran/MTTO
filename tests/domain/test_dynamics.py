import numpy as np
import pytest
from numpy.typing import NDArray

from mtto.domain.dynamics import (
    air_resis_force_numba,
    guideway_vortex_resis_force_numba,
    linear_generator_resis_force_numba,
    sledge_frictional_brake_force_numba,
    slope_resis_force_numba,
    vortex_brake_force_numba,
    wear_plate_frictional_brake_force_numba,
)


@pytest.fixture
def v_sample() -> NDArray[np.float64]:
    return np.arange(0.0, 600.0, 1.0)


def _total_resis_force(speed: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        [
            air_resis_force_numba(v, 5)
            + guideway_vortex_resis_force_numba(v, 5)
            + linear_generator_resis_force_numba(v, 5)
            + sledge_frictional_brake_force_numba(v, 4.35, 0)
            + slope_resis_force_numba(4.35, 0)
            + vortex_brake_force_numba(v, 5, 0)
            + wear_plate_frictional_brake_force_numba(v, 5)
            for v in speed
        ],
        dtype=np.float64,
    )


def test_total_resis_force_nonnegative(v_sample: NDArray[np.float64]) -> None:
    total_resis_force = _total_resis_force(v_sample)
    # 检查总阻力是否全部为非负
    assert np.all(total_resis_force >= 0)
