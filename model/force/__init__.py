from model.force.numba import (
    air_resis_force_numba,
    guideway_vortex_resis_force_numba,
    linear_generator_resis_force_numba,
    sledge_frictional_brake_force_numba,
    slope_resis_force_numba,
    vortex_brake_force_numba,
    wear_plate_frictional_brake_force_numba,
)
from model.force.numpy import (
    air_resis_force,
    guideway_vortex_resis_force,
    linear_generator_resis_force,
    sledge_frictional_brake_force,
    slope_resis_force,
    vortex_brake_force,
    wear_plate_frictional_brake_force,
)

__all__ = [
    # 基于 numpy 库实现的向量化版本
    "air_resis_force",
    "guideway_vortex_resis_force",
    "linear_generator_resis_force",
    "slope_resis_force",
    "sledge_frictional_brake_force",
    "vortex_brake_force",
    "wear_plate_frictional_brake_force",
    # 基于 numba 库实现的标量化加速版本
    "air_resis_force_numba",
    "guideway_vortex_resis_force_numba",
    "linear_generator_resis_force_numba",
    "slope_resis_force_numba",
    "sledge_frictional_brake_force_numba",
    "vortex_brake_force_numba",
    "wear_plate_frictional_brake_force_numba",
]
