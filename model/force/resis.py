import numpy as np
from numpy.typing import NDArray
from typing import Union

Numeric = Union[float, np.number, NDArray]


def AddedResis(
    speed: Numeric,
    mass: Numeric,
    kind: int,
) -> NDArray[np.float64]:
    """
    计算轨道对列车的固定附加阻力

    Args:
        speed: 列车速度大小(单位: m/s)
        mass: 列车质量(单位: T)
        kind: 0代表最小安全悬浮速度, 1代表最大安全制动速度

    Returns:
        列车收到的附加阻力(单位: kN)
    """

    speed_km = 3.6 * np.asarray(speed, dtype=np.float32)
    added_resis_force = np.piecewise(
        speed_km,
        [
            (kind == 0) & (speed_km > 10),
            (kind == 1) & (speed_km > 10),
            (speed_km <= 10),
        ],
        [lambda v: 0.08 * mass, lambda v: -0.2 * mass, lambda v: 0],
    )

    return added_resis_force


def AirResis(
    speed: Numeric,
    numoftrainsets: int,
) -> NDArray[np.float64]:
    """
    计算空气阻力

    Args:
        numoftrainsets: 列车编组数
        speed: 列车速度大小(单位: m/s)

    Returns:
        列车受到的空气阻力(单位: kN)
    """
    # 参考文献：《高速磁浮与高速轮轨交通系统比较》中的公式

    speed = np.asarray(speed, dtype=np.float64)
    air_resis_force = 2.8 * (0.53 * numoftrainsets / 2 + 0.3) * speed**2 / 1000.0
    return air_resis_force


def GuidewayVortexResis(
    speed: Numeric,
    numoftrainsets: int,
) -> NDArray[np.float64]:
    """
    计算线路两侧导向轨的磁化阻力

    Args:
        speed: 列车速度大小(单位: m/s)
        numoftrainsets: 列车编组数

    Returns:
        列车受到的导向轨磁化(单位: kN)

    """

    # 参考文献：《高速磁浮与高速轮轨交通系统比较》

    v_km = 3.6 * np.asarray(speed, dtype=np.float64)
    guideway_vortex_resis_force = numoftrainsets * (
        0.1 * np.power(v_km, 0.5) + 0.02 * np.power(v_km, 0.7)
    )

    return guideway_vortex_resis_force


def LinearGeneResis(
    speed: Numeric,
    numoftrainsets: int,
) -> NDArray[np.float64]:
    """
    计算磁浮列车的直线发电机产生的运行阻力

    Args:
        speed: 列车速度大小(单位: m/s)
        numoftrainsets: 列车编组数
    Returns:
        列车受到的直线发电机运行阻力(单位: kN)
    """

    # 参考文献：《高速磁浮与高速轮轨交通系统比较》

    speed_km = 3.6 * np.asarray(speed, np.float64)
    lineargene_resis_force = np.piecewise(
        speed_km,
        [
            speed_km < 20,
            (speed_km >= 20) & (speed_km < 70),
            (speed_km >= 70) & (speed_km < 600),
        ],
        [
            lambda v: 0.0,
            lambda v: 7.3 * numoftrainsets,
            lambda v: 146 * 3.6 * numoftrainsets / v - 0.2,
        ],
    )
    return lineargene_resis_force


def SlopeResis(
    mass: Numeric,
    slope: Numeric,
) -> NDArray[np.float64]:
    """
    计算斜坡阻力

    Args:
        mass: 列车质量(单位: T)
        slope: 坡度(百分位, 每100m上升或下降的高度)
    Returns:
        列车受到的斜坡阻力(单位: kN)
    """

    # return 9.8 * mass * slope / 100
    slope = np.asarray(slope, dtype=np.float64)
    slope_resis_force = 9.8 * mass * slope / 100
    return slope_resis_force
