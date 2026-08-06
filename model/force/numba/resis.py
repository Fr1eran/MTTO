from numba import (
    njit,
)


@njit(cache=True)
def air_resis_force_numba(speed: float, numoftrainsets: float):
    return 2.8 * (0.53 * numoftrainsets / 2.0 + 0.3) * speed**2 / 1000.0


@njit(cache=True)
def guideway_vortex_resis_force_numba(speed: float, numoftrainsets: float):
    speed_km = 3.6 * speed
    return numoftrainsets * (0.1 * speed_km**0.5 + 0.02 * speed_km**0.7)


@njit(cache=True)
def linear_generator_resis_force_numba(speed: float, numoftrainsets: float):
    speed_km = 3.6 * speed
    if speed_km < 20.0:
        return 0.0
    if speed_km < 70.0:
        return 7.3 * numoftrainsets
    if speed_km < 600.0:
        return 146.0 * 3.6 * numoftrainsets / speed_km - 0.2
    return 0.0


@njit(cache=True)
def slope_resis_force_numba(mass: float, slope: float):
    return 9.8 * mass * slope / 100.0


__all__ = [
    "air_resis_force_numba",
    "guideway_vortex_resis_force_numba",
    "linear_generator_resis_force_numba",
    "slope_resis_force_numba",
]
