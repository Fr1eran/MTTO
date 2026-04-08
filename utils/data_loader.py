from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(relative_path: str | Path) -> dict[str, Any]:
    file_path = resolve_project_path(relative_path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(relative_path: str | Path) -> Any:
    file_path = resolve_project_path(relative_path)
    with file_path.open("rb") as f:
        return pickle.load(f)


def load_excel(relative_path: str | Path, **kwargs: Any):
    import pandas as pd

    file_path = resolve_project_path(relative_path)
    return pd.read_excel(file_path, **kwargs)


def load_slopes() -> tuple[np.ndarray, np.ndarray]:
    data = load_json("data/rail/raw/slopes.json")
    return np.asarray(data["slopes"], dtype=np.float64), np.asarray(
        data["intervals"], dtype=np.float64
    )


def load_speed_limits(
    to_mps: bool = True,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    data = load_json("data/rail/raw/speed_limits.json")
    speed_limits = np.asarray(data["speed_limits"], dtype=dtype)
    if to_mps:
        speed_limits = speed_limits / 3.6
    return speed_limits, np.asarray(data["intervals"], dtype=np.float64)


def load_auxiliary_stopping_areas_ap_and_dp() -> tuple[list[float], list[float]]:
    data = load_json("data/rail/raw/auxiliary_parking_areas.json")
    return data["accessible_points"], data["dangerous_points"]


def load_stations() -> dict[str, Any]:
    return load_json("data/rail/raw/stations.json")


def load_acceleration_zones() -> dict[str, Any]:
    return load_json("data/rail/raw/acceleration_zones.json")


def load_safeguard_curve(curve_name: str):
    return load_pickle(f"data/rail/safeguard/{curve_name}.pkl")


def load_safeguard_curves(*curve_names: str) -> tuple[Any, ...]:
    return tuple(load_safeguard_curve(name) for name in curve_names)


def load_stations_goal_positions() -> tuple[float, float]:
    stations = load_stations()
    return float(stations["start_station"]["target"]), float(
        stations["end_station"]["target"]
    )
