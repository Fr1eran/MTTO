"""Disk caching mechanisms, key hashing, and serialization for DP transition graphs."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import pickle
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

import mtto
from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import EnergyParams
from mtto.domain.line import Line
from mtto.domain.safeguard import Safeguard
from mtto.dp.graph import (
    DP_UPPER_SPEED_ENVELOPE_VERSION,
    STATIC_REGION_SAMPLE_STEP_M,
    TransitionGraph,
)

__all__ = [
    "TRANSITION_CACHE_ALGORITHM_VERSION",
    "TRANSITION_CACHE_SCHEMA_VERSION",
    "compute_cache_input_hash",
    "load_transition_graph_from_disk",
    "make_cache_folder_name",
    "save_transition_graph_to_disk",
    "validate_transition_graph",
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

TRANSITION_CACHE_SCHEMA_VERSION = 3
TRANSITION_CACHE_ALGORITHM_VERSION = "task-upper-speed-envelope-v1"
_CACHE_ENDPOINT_SPEED_MPS = 0.0


def _format_float_token(value: float, *, decimals: int = 10) -> str:
    """Format float into a path-safe string token."""
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    token = f"{round(float(value), decimals):.{decimals}f}".rstrip("0").rstrip(".")
    if token in {"", "-0", "0"}:
        token = "0"
    if "." not in token:
        token = f"{token}.0"
    return token.replace("-", "neg").replace(".", "p")


def _hash_value(hasher: hashlib._Hash, name: str, value: object) -> None:
    hasher.update(f"{name}={value!r}\0".encode())


def _hash_array(
    hasher: hashlib._Hash,
    name: str,
    values: Sequence[object] | NDArray[np.generic],
) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    hasher.update(f"{name}|dtype={array.dtype.str}|shape={array.shape}\0".encode())
    hasher.update(array.tobytes())
    hasher.update(b"\0")


def compute_cache_input_hash(
    *,
    stages: NDArray[np.float64],
    speed_states: NDArray[np.float64],
    stage_speed_upper_idx: NDArray[np.int_],
    start_position: float,
    target_position: float,
    speed_grid_upper_mps: float,
    delta_speed: float,
    stage_division: str,
    sub_stage_count: int,
    uniform_step_size: float,
    vehicle: Vehicle,
    energy: EnergyParams,
    safeguard: Safeguard,
    track: Line,
    scenario_hash: str,
    task_max_stop_error_m: float,
) -> str:
    """Hash every input that can change a transition graph."""
    hasher = hashlib.sha256()
    _hash_value(hasher, "cache_schema_version", TRANSITION_CACHE_SCHEMA_VERSION)
    _hash_value(hasher, "algorithm_version", TRANSITION_CACHE_ALGORITHM_VERSION)
    _hash_value(hasher, "cache_start_speed", _CACHE_ENDPOINT_SPEED_MPS)
    _hash_value(hasher, "cache_target_speed", _CACHE_ENDPOINT_SPEED_MPS)
    _hash_value(hasher, "start_position", float(start_position))
    _hash_value(hasher, "target_position", float(target_position))
    _hash_value(hasher, "speed_grid_upper_mps", speed_grid_upper_mps)
    _hash_value(hasher, "delta_speed", delta_speed)
    _hash_value(hasher, "stage_division", stage_division)
    _hash_value(hasher, "sub_stage_count", sub_stage_count)
    _hash_value(hasher, "uniform_step_size", uniform_step_size)

    _hash_array(hasher, "stages", stages)
    _hash_array(hasher, "speed_states", speed_states)
    _hash_array(hasher, "stage_speed_upper_idx", stage_speed_upper_idx)
    for name in (
        "mass",
        "numoftrainsets",
        "length",
        "max_speed",
        "max_acc",
        "max_dec",
        "max_slope_capacity",
        "levi_power_per_mass",
    ):
        _hash_value(hasher, f"vehicle.{name}", getattr(vehicle, name))

    for name in (
        "R_m",
        "L_d",
        "R_k",
        "L_k",
        "Tau",
        "Psi_fd",
        "k_c",
        "Phi_1",
        "Phi_2",
    ):
        _hash_value(hasher, f"ecc.{name}", getattr(energy, name))

    _hash_array(hasher, "safeguard.speed_limits", safeguard.speed_limits)
    _hash_array(
        hasher,
        "safeguard.speed_limit_intervals",
        safeguard.speed_limit_intervals,
    )
    _hash_value(hasher, "safeguard.gamma", safeguard.params.factor)
    for attr_name, hash_name in (
        ("levi_curves", "levi_curves_list"),
        ("brake_curves", "brake_curves_list"),
        ("min_curves", "min_curves_list"),
        ("max_curves", "max_curves_list"),
    ):
        curves = getattr(safeguard, attr_name)
        _hash_value(hasher, f"safeguard.{hash_name}.count", len(curves))
        for index, curve in enumerate(curves):
            _hash_array(hasher, f"safeguard.{hash_name}.{index}", curve)

    _hash_array(hasher, "track.slopes", track.slopes)
    _hash_array(hasher, "track.slope_intervals", track.slope_intervals)
    _hash_array(hasher, "track.speed_limits", track.speed_limits)
    _hash_array(
        hasher,
        "track.speed_limit_intervals",
        track.speed_limit_intervals,
    )
    _hash_value(hasher, "scenario_hash", scenario_hash)
    _hash_value(hasher, "task.max_stop_error_m", task_max_stop_error_m)
    _hash_value(hasher, "static_region_sample_step_m", STATIC_REGION_SAMPLE_STEP_M)
    _hash_value(hasher, "mtto_version", mtto.__version__)
    return hasher.hexdigest()


def make_cache_folder_name(
    *,
    stage_division: str,
    sub_stage_count: int,
    uniform_step_size: float,
    speed_grid_upper_mps: float,
    delta_speed: float,
    content_hash: str,
) -> str:
    """Generate a versioned, readable cache directory name."""
    delta_token = _format_float_token(delta_speed)
    speed_upper_token = _format_float_token(speed_grid_upper_mps)
    hash_prefix = content_hash[:16]
    if stage_division == "uniform":
        div_token = f"uni{_format_float_token(uniform_step_size)}"
    else:
        div_token = f"var{sub_stage_count}"
    return (
        f"v{TRANSITION_CACHE_SCHEMA_VERSION}_{div_token}_"
        f"{speed_upper_token}_{delta_token}_{hash_prefix}"
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace a cache file atomically, leaving the old file on failure."""
    file_descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = None
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def save_transition_graph_to_disk(
    *,
    cache_base_dir: str | Path,
    graph_cache: TransitionGraph,
    start_position: float,
    target_position: float,
    content_hash: str,
    stage_division: str,
    sub_stage_count: int,
    uniform_step_size: float,
    speed_grid_upper_mps: float,
    delta_speed: float,
) -> Path | None:
    """Save graph cache payload and metadata to disk."""
    folder_name = make_cache_folder_name(
        stage_division=stage_division,
        sub_stage_count=sub_stage_count,
        uniform_step_size=uniform_step_size,
        speed_grid_upper_mps=speed_grid_upper_mps,
        delta_speed=delta_speed,
        content_hash=content_hash,
    )
    cache_dir = Path(cache_base_dir) / folder_name
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("无法创建缓存目录 %s: %s", cache_dir, exc)
        return None

    graph_path = cache_dir / "graph_data.pkl.gz"
    meta_path = cache_dir / "metadata.json"
    try:
        graph_bytes = gzip.compress(
            pickle.dumps(graph_cache, protocol=5), compresslevel=5
        )
        _atomic_write_bytes(graph_path, graph_bytes)
    except Exception as exc:
        logger.warning("写入缓存文件失败 (%s): %s", graph_path, exc)
        return None

    metadata = {
        "cache_schema_version": TRANSITION_CACHE_SCHEMA_VERSION,
        "algorithm_version": TRANSITION_CACHE_ALGORITHM_VERSION,
        "dp_upper_speed_envelope_version": DP_UPPER_SPEED_ENVELOPE_VERSION,
        "content_hash": content_hash,
        "file_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "stage_division": stage_division,
        "sub_stage_count": sub_stage_count,
        "uniform_step_size": uniform_step_size,
        "speed_grid_upper_mps": speed_grid_upper_mps,
        "delta_speed": delta_speed,
        "start_position": float(start_position),
        "target_position": float(target_position),
        "cache_start_speed": _CACHE_ENDPOINT_SPEED_MPS,
        "cache_target_speed": _CACHE_ENDPOINT_SPEED_MPS,
        "num_stages": int(len(graph_cache["stages"])),
        "num_speed_states": int(len(graph_cache["speed_states"])),
        "total_valid_edges": int(graph_cache["total_valid_edges"]),
        "created_at": datetime.now().isoformat(),
    }
    try:
        metadata_bytes = json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write_bytes(meta_path, metadata_bytes)
    except Exception as exc:
        logger.warning("写入缓存元数据失败 (%s): %s", meta_path, exc)

    return cache_dir


def _metadata_float_matches(
    metadata: dict[str, object], name: str, expected: float
) -> bool:
    value = metadata.get(name)
    if not isinstance(value, (int, float)):
        return False
    return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)


def validate_transition_graph(
    graph_cache: object,
    *,
    expected_stages: NDArray[np.float64],
    expected_speed_states: NDArray[np.float64],
    expected_stage_speed_upper_idx: NDArray[np.int_],
) -> tuple[bool, str]:
    """Validate cached graph shape, values, and sparse edge payloads."""
    try:
        if not isinstance(graph_cache, dict):
            return False, "root is not a dictionary"
        required_keys = {
            "stages",
            "speed_states",
            "stage_speed_upper_idx",
            "transitions",
            "total_valid_edges",
        }
        if not required_keys.issubset(graph_cache):
            return False, "required graph keys are missing"

        stages = graph_cache["stages"]
        speed_states = graph_cache["speed_states"]
        upper_idx = graph_cache["stage_speed_upper_idx"]
        transitions = graph_cache["transitions"]
        total_valid_edges = graph_cache["total_valid_edges"]
        if not all(
            isinstance(array, np.ndarray) for array in (stages, speed_states, upper_idx)
        ):
            return False, "grid fields are not numpy arrays"
        if stages.ndim != 1 or speed_states.ndim != 1 or upper_idx.ndim != 1:
            return False, "grid fields must be one-dimensional"
        if not np.issubdtype(upper_idx.dtype, np.integer):
            return False, "upper-bound indices are not integral"
        if not np.array_equal(stages, expected_stages):
            return False, "stage grid does not match the cache key"
        if not np.array_equal(speed_states, expected_speed_states):
            return False, "speed grid does not match the cache key"
        if not np.array_equal(upper_idx, expected_stage_speed_upper_idx):
            return False, "stage speed bounds do not match the cache key"
        if not np.all(np.isfinite(stages)) or not np.all(np.isfinite(speed_states)):
            return False, "grid contains non-finite values"
        if speed_states.size == 0 or not math.isclose(
            float(speed_states[0]), 0.0, abs_tol=1e-9
        ):
            return False, "speed grid does not start at zero"
        if speed_states.size > 1 and not np.all(np.diff(speed_states) > 0.0):
            return False, "speed grid is not strictly increasing"
        if stages.size < 2:
            return False, "stage grid has fewer than two points"
        stage_diff = np.diff(stages)
        if not (np.all(stage_diff > 0.0) or np.all(stage_diff < 0.0)):
            return False, "stage grid is not strictly monotonic"
        if np.any(upper_idx < -1) or np.any(upper_idx >= speed_states.size):
            return False, "stage speed bounds are outside the speed grid"
        if not isinstance(transitions, list) or len(transitions) != stages.size - 1:
            return False, "transition stage count does not match the grid"
        if not isinstance(total_valid_edges, (int, np.integer)):
            return False, "total_valid_edges is not integral"
        if int(total_valid_edges) < 0:
            return False, "total_valid_edges is negative"

        counted_edges = 0
        for stage_index, rows in enumerate(transitions):
            if not isinstance(rows, list) or len(rows) != speed_states.size:
                return False, f"transition row {stage_index} has wrong width"
            for speed_index, transition in enumerate(rows):
                if transition is None:
                    continue
                if not isinstance(transition, (tuple, list)) or len(transition) != 3:
                    return False, "transition payload has wrong shape"
                next_indices, delta_energy, delta_time = transition
                if not all(
                    isinstance(array, np.ndarray)
                    and array.ndim == 1
                    and np.issubdtype(array.dtype, np.number)
                    for array in (next_indices, delta_energy, delta_time)
                ):
                    return False, "transition payload arrays are invalid"
                if not np.issubdtype(next_indices.dtype, np.integer):
                    return False, "transition indices are not integral"
                if not (len(next_indices) == len(delta_energy) == len(delta_time)):
                    return False, "transition payload lengths differ"
                if next_indices.size > 1 and not np.all(np.diff(next_indices) > 0):
                    return False, "transition indices are not strictly increasing"
                if np.any(next_indices < 0) or np.any(
                    next_indices >= speed_states.size
                ):
                    return False, "transition index is outside the speed grid"
                if not np.all(np.isfinite(delta_energy)) or not np.all(
                    np.isfinite(delta_time)
                ):
                    return False, "transition payload contains non-finite values"
                if np.any(delta_time <= 0.0):
                    return False, "transition time must be positive"
                if speed_index > int(upper_idx[stage_index]):
                    return False, "transition exists above its stage bound"
                counted_edges += int(next_indices.size)

        if counted_edges != int(total_valid_edges):
            return False, "total_valid_edges does not match payloads"
    except Exception as exc:
        return False, f"validation raised {type(exc).__name__}: {exc}"

    return True, ""


def load_transition_graph_from_disk(
    *,
    cache_base_dir: str | Path,
    content_hash: str,
    expected_stages: NDArray[np.float64],
    expected_speed_states: NDArray[np.float64],
    expected_stage_speed_upper_idx: NDArray[np.int_],
    start_position: float,
    target_position: float,
    stage_division: str,
    sub_stage_count: int,
    uniform_step_size: float,
    speed_grid_upper_mps: float,
    delta_speed: float,
) -> TransitionGraph | None:
    """Load graph cache from disk if valid and matching all expected parameters."""
    folder_name = make_cache_folder_name(
        stage_division=stage_division,
        sub_stage_count=sub_stage_count,
        uniform_step_size=uniform_step_size,
        speed_grid_upper_mps=speed_grid_upper_mps,
        delta_speed=delta_speed,
        content_hash=content_hash,
    )
    cache_dir = Path(cache_base_dir) / folder_name
    graph_path = cache_dir / "graph_data.pkl.gz"
    meta_path = cache_dir / "metadata.json"
    if not graph_path.is_file() or not meta_path.is_file():
        return None

    try:
        metadata_value = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata_value, dict):
            raise ValueError("metadata root is not a dictionary")
        metadata: dict[str, object] = metadata_value
    except Exception as exc:
        logger.warning("缓存元数据损坏，将重新计算 (%s): %s", meta_path, exc)
        return None

    expected_ints = {
        "cache_schema_version": TRANSITION_CACHE_SCHEMA_VERSION,
        "sub_stage_count": sub_stage_count,
        "num_stages": len(expected_stages),
        "num_speed_states": len(expected_speed_states),
    }
    for name, expected in expected_ints.items():
        if metadata.get(name) != expected:
            logger.warning("缓存元数据 %s 不匹配，将重新计算 (%s)", name, meta_path)
            return None
    if metadata.get("algorithm_version") != TRANSITION_CACHE_ALGORITHM_VERSION:
        logger.warning("缓存算法版本不匹配，将重新计算 (%s)", meta_path)
        return None
    if metadata.get("content_hash") != content_hash:
        logger.warning("缓存参数签名不匹配，将重新计算 (%s)", meta_path)
        return None
    if metadata.get("stage_division") != stage_division:
        logger.warning("缓存阶段划分配置不匹配，将重新计算 (%s)", meta_path)
        return None
    if (
        not _metadata_float_matches(metadata, "uniform_step_size", uniform_step_size)
        or not _metadata_float_matches(
            metadata, "speed_grid_upper_mps", speed_grid_upper_mps
        )
        or not _metadata_float_matches(metadata, "delta_speed", delta_speed)
        or not _metadata_float_matches(metadata, "start_position", start_position)
        or not _metadata_float_matches(metadata, "target_position", target_position)
        or not _metadata_float_matches(
            metadata, "cache_start_speed", _CACHE_ENDPOINT_SPEED_MPS
        )
        or not _metadata_float_matches(
            metadata, "cache_target_speed", _CACHE_ENDPOINT_SPEED_MPS
        )
    ):
        logger.warning("缓存任务参数不匹配，将重新计算 (%s)", meta_path)
        return None

    expected_file_hash = metadata.get("file_sha256")
    if not isinstance(expected_file_hash, str):
        logger.warning("缓存文件缺少完整性校验值，将重新计算 (%s)", meta_path)
        return None
    try:
        graph_bytes = graph_path.read_bytes()
    except OSError as exc:
        logger.warning("读取缓存文件失败 (%s): %s", graph_path, exc)
        return None
    if hashlib.sha256(graph_bytes).hexdigest() != expected_file_hash:
        logger.warning("缓存文件完整性校验失败 (%s)，将重新计算", graph_path)
        return None

    try:
        graph_cache: object = pickle.loads(gzip.decompress(graph_bytes))
    except Exception as exc:
        logger.warning("缓存文件反序列化失败 (%s): %s", graph_path, exc)
        return None

    valid, reason = validate_transition_graph(
        graph_cache,
        expected_stages=expected_stages,
        expected_speed_states=expected_speed_states,
        expected_stage_speed_upper_idx=expected_stage_speed_upper_idx,
    )
    if not valid:
        logger.warning("缓存结构校验失败，将重新计算 (%s): %s", graph_path, reason)
        return None
    typed_graph_cache = cast(TransitionGraph, graph_cache)
    if metadata.get("total_valid_edges") != typed_graph_cache["total_valid_edges"]:
        logger.warning("缓存边数量元数据不匹配，将重新计算 (%s)", meta_path)
        return None

    logger.info(
        "从磁盘缓存加载状态转移图: %s (%s 条可行转移边)",
        cache_dir.name,
        typed_graph_cache["total_valid_edges"],
    )
    return typed_graph_cache
