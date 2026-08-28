from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing import (
    event_accumulator,
)

from rl.reward_diagnostics import (
    LEGACY_UNKNOWN_VIOLATION_CODE,
    REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    REWARD_NAMES,
    REWARD_SIGNAL_COUNT,
    TOTAL_REWARD_INDEX,
)


@dataclass(frozen=True)
class ScalarSeries:
    tag: str
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray


@dataclass(frozen=True)
class RewardDiagnosticsArtifact:
    reward_names: tuple[str, ...]
    rollout_end_step: np.ndarray
    rollout_transition_count: np.ndarray
    rollout_reward_sum: np.ndarray
    rollout_reward_abs_sum: np.ndarray
    rollout_reward_nonzero_count: np.ndarray
    rollout_reward_cross_product: np.ndarray
    episode_end_step: np.ndarray
    episode_worker_rank: np.ndarray
    episode_index: np.ndarray
    episode_length: np.ndarray
    episode_terminated: np.ndarray
    episode_truncated: np.ndarray
    episode_complete: np.ndarray
    episode_violation_code: np.ndarray
    episode_reward_sums: np.ndarray


@dataclass(frozen=True)
class CompleteEpisodeSeries:
    end_step: np.ndarray
    total_reward: np.ndarray
    length: np.ndarray


@dataclass(frozen=True)
class CompleteEpisodeSequence:
    """Individual complete episodes ordered across all vector workers."""

    episode_number: np.ndarray
    total_reward: np.ndarray
    length: np.ndarray
    violation_code: np.ndarray


def extract_complete_episode_sequence(
    artifact: RewardDiagnosticsArtifact,
) -> CompleteEpisodeSequence:
    """Return every complete episode with a global cumulative episode index.

    Worker-local episode indices are not globally unique.  Ordering by end
    step, worker rank, and local index yields a deterministic merged sequence
    suitable for an episode-number learning-curve axis.
    """
    complete = artifact.episode_complete
    end_steps = artifact.episode_end_step[complete]
    worker_ranks = artifact.episode_worker_rank[complete]
    worker_indices = artifact.episode_index[complete]
    rewards = artifact.episode_reward_sums[complete, TOTAL_REWARD_INDEX]
    lengths = artifact.episode_length[complete].astype(np.float64)
    violation_codes = artifact.episode_violation_code[complete]
    if end_steps.size == 0:
        return CompleteEpisodeSequence(
            episode_number=np.empty(0, dtype=np.int64),
            total_reward=np.empty(0, dtype=np.float64),
            length=np.empty(0, dtype=np.float64),
            violation_code=np.empty(0, dtype=np.int8),
        )

    order = np.lexsort((worker_indices, worker_ranks, end_steps))
    episode_count = end_steps.size
    return CompleteEpisodeSequence(
        episode_number=np.arange(1, episode_count + 1, dtype=np.int64),
        total_reward=np.asarray(rewards[order], dtype=np.float64),
        length=np.asarray(lengths[order], dtype=np.float64),
        violation_code=np.asarray(violation_codes[order], dtype=np.int8),
    )


def extract_complete_episode_series(
    artifact: RewardDiagnosticsArtifact,
) -> CompleteEpisodeSeries:
    """Build a strictly ordered curve, averaging episodes ending at one step."""
    complete = artifact.episode_complete
    steps = artifact.episode_end_step[complete]
    rewards = artifact.episode_reward_sums[complete, TOTAL_REWARD_INDEX]
    lengths = artifact.episode_length[complete].astype(np.float64)
    if steps.size == 0:
        return CompleteEpisodeSeries(
            end_step=np.empty(0, dtype=np.int64),
            total_reward=np.empty(0, dtype=np.float64),
            length=np.empty(0, dtype=np.float64),
        )
    unique_steps, inverse, counts = np.unique(
        steps, return_inverse=True, return_counts=True
    )
    reward_sum = np.bincount(inverse, weights=rewards)
    length_sum = np.bincount(inverse, weights=lengths)
    return CompleteEpisodeSeries(
        end_step=unique_steps.astype(np.int64, copy=False),
        total_reward=reward_sum / counts,
        length=length_sum / counts,
    )


def load_reward_diagnostics_artifact(
    path: str | Path,
) -> RewardDiagnosticsArtifact:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Reward diagnostics artifact not found: {artifact_path}"
        )
    required = (
        "schema_version",
        "reward_names",
        "rollout_end_step",
        "rollout_transition_count",
        "rollout_reward_sum",
        "rollout_reward_abs_sum",
        "rollout_reward_nonzero_count",
        "rollout_reward_cross_product",
        "episode_end_step",
        "episode_worker_rank",
        "episode_index",
        "episode_length",
        "episode_terminated",
        "episode_truncated",
        "episode_complete",
        "episode_reward_sums",
    )
    with np.load(artifact_path, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"Reward diagnostics artifact is missing {missing}")
        version = np.asarray(data["schema_version"], dtype=np.int16)
        names = tuple(str(name) for name in np.asarray(data["reward_names"]))
        values = {name: np.asarray(data[name]).copy() for name in required[2:]}
        if "episode_violation_code" in data.files:
            values["episode_violation_code"] = np.asarray(
                data["episode_violation_code"]
            ).copy()
    if version.shape != (1,) or int(version[0]) not in (
        2,
        REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    ):
        raise ValueError("Unsupported reward diagnostics schema version")
    if names != REWARD_NAMES:
        raise ValueError("Reward diagnostics names do not match the schema")

    rollout_end = np.asarray(values["rollout_end_step"], dtype=np.int64)
    rollout_count = np.asarray(values["rollout_transition_count"], dtype=np.int64)
    rollout_nonzero = np.asarray(values["rollout_reward_nonzero_count"], dtype=np.int64)
    rollout_rows = rollout_count.size
    expected_rollout_shapes = {
        "rollout_end_step": (rollout_rows,),
        "rollout_reward_sum": (rollout_rows, REWARD_SIGNAL_COUNT),
        "rollout_reward_abs_sum": (rollout_rows, REWARD_SIGNAL_COUNT),
        "rollout_reward_nonzero_count": (rollout_rows, REWARD_SIGNAL_COUNT),
        "rollout_reward_cross_product": (
            rollout_rows,
            REWARD_SIGNAL_COUNT,
            REWARD_SIGNAL_COUNT,
        ),
    }
    for name, shape in expected_rollout_shapes.items():
        if np.asarray(values[name]).shape != shape:
            raise ValueError(f"Reward diagnostics has invalid {name} shape")
    episode_end = np.asarray(values["episode_end_step"], dtype=np.int64)
    episode_rows = episode_end.size
    for name in (
        "episode_worker_rank",
        "episode_index",
        "episode_length",
        "episode_terminated",
        "episode_truncated",
        "episode_complete",
    ):
        if np.asarray(values[name]).shape != (episode_rows,):
            raise ValueError(f"Reward diagnostics has invalid {name} shape")
    episode_rewards = np.asarray(values["episode_reward_sums"], dtype=np.float64)
    if episode_rewards.shape != (episode_rows, REWARD_SIGNAL_COUNT):
        raise ValueError("Reward diagnostics has invalid episode_reward_sums shape")
    if int(version[0]) == REWARD_DIAGNOSTICS_SCHEMA_VERSION:
        if "episode_violation_code" not in values:
            raise ValueError("Reward diagnostics is missing episode_violation_code")
        episode_violation_code = np.asarray(
            values["episode_violation_code"], dtype=np.int8
        )
        if episode_violation_code.shape != (episode_rows,):
            raise ValueError(
                "Reward diagnostics has invalid episode_violation_code shape"
            )
    else:
        episode_violation_code = np.full(
            episode_rows, LEGACY_UNKNOWN_VIOLATION_CODE, dtype=np.int8
        )

    reward_sum = np.asarray(values["rollout_reward_sum"], dtype=np.float64)
    reward_abs_sum = np.asarray(values["rollout_reward_abs_sum"], dtype=np.float64)
    cross = np.asarray(values["rollout_reward_cross_product"], dtype=np.float64)
    if (
        np.any(rollout_count < 0)
        or np.any(rollout_end < 0)
        or np.any(np.diff(rollout_end) <= 0)
        or np.any(reward_abs_sum < 0.0)
        or np.any(rollout_nonzero < 0)
        or np.any(rollout_nonzero > rollout_count[:, None])
        or not np.all(np.isfinite(reward_sum))
        or not np.all(np.isfinite(reward_abs_sum))
        or not np.all(np.isfinite(cross))
        or not np.all(np.isfinite(episode_rewards))
    ):
        raise ValueError("Reward diagnostics contains invalid values")
    if not np.allclose(cross, np.swapaxes(cross, 1, 2), atol=1e-8):
        raise ValueError("Reward diagnostics cross products are not symmetric")
    episode_length = np.asarray(values["episode_length"], dtype=np.int32)
    episode_terminated = np.asarray(values["episode_terminated"], dtype=np.bool_)
    episode_truncated = np.asarray(values["episode_truncated"], dtype=np.bool_)
    episode_complete = np.asarray(values["episode_complete"], dtype=np.bool_)
    episode_worker_rank = np.asarray(values["episode_worker_rank"], dtype=np.int16)
    episode_index = np.asarray(values["episode_index"], dtype=np.int64)
    if (
        np.any(episode_end < 0)
        or np.any(np.diff(episode_end) < 0)
        or np.any(episode_worker_rank < 0)
        or np.any(episode_index < 0)
        or np.any(episode_length <= 0)
        or np.any(episode_terminated & episode_truncated)
        or np.any(episode_complete != (episode_terminated | episode_truncated))
        or (
            int(version[0]) == REWARD_DIAGNOSTICS_SCHEMA_VERSION
            and np.any(~np.isin(episode_violation_code, [0, 1, 2, 3, 4]))
        )
        or int(episode_length.sum()) != int(rollout_count.sum())
    ):
        raise ValueError("Reward diagnostics contains invalid episode metadata")
    component_slice = slice(0, TOTAL_REWARD_INDEX)
    if not np.allclose(
        reward_sum[:, TOTAL_REWARD_INDEX],
        reward_sum[:, component_slice].sum(axis=1),
        rtol=1e-6,
        atol=1e-3,
    ) or not np.allclose(
        episode_rewards[:, TOTAL_REWARD_INDEX],
        episode_rewards[:, component_slice].sum(axis=1),
        rtol=1e-6,
        atol=1e-3,
    ):
        raise ValueError("Reward diagnostics total does not equal component sum")

    return RewardDiagnosticsArtifact(
        reward_names=names,
        rollout_end_step=rollout_end,
        rollout_transition_count=rollout_count,
        rollout_reward_sum=reward_sum,
        rollout_reward_abs_sum=reward_abs_sum,
        rollout_reward_nonzero_count=rollout_nonzero,
        rollout_reward_cross_product=cross,
        episode_end_step=episode_end,
        episode_worker_rank=episode_worker_rank,
        episode_index=episode_index,
        episode_length=episode_length,
        episode_terminated=episode_terminated,
        episode_truncated=episode_truncated,
        episode_complete=episode_complete,
        episode_violation_code=episode_violation_code,
        episode_reward_sums=episode_rewards,
    )


DEFAULT_SAMPLING_HEALTH_TAGS = [
    "rollout/ep_rew_mean",
    "train/approx_kl",
]

EXCLUDED_ANALYSIS_TAG_PREFIXES = ("basic/", "constraint/", "event/")
EXCLUDED_ANALYSIS_TAGS = frozenset({"rewards/terminal_stopping", "rewards/punctuality"})


# Writers use the canonical names below.  Readers retain aliases so existing
# TensorBoard runs generated before the info-contract cleanup remain analyzable.
LEGACY_INFO_TAG_ALIASES = {
    "outcome/truncated": "constraint/is_truncated",
}


def _is_excluded_analysis_tag(tag: str) -> bool:
    return tag in EXCLUDED_ANALYSIS_TAGS or tag.startswith(
        EXCLUDED_ANALYSIS_TAG_PREFIXES
    )


def with_legacy_info_tag_aliases(
    series_map: dict[str, ScalarSeries],
) -> dict[str, ScalarSeries]:
    resolved = dict(series_map)
    for canonical_tag, legacy_tag in LEGACY_INFO_TAG_ALIASES.items():
        if canonical_tag not in resolved and legacy_tag in resolved:
            resolved[canonical_tag] = resolved[legacy_tag]
    return {
        tag: series
        for tag, series in resolved.items()
        if not _is_excluded_analysis_tag(tag)
    }


def list_run_directories(log_root: str | Path) -> list[Path]:
    root = Path(log_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime
    )


def resolve_run_directory(log_root: str | Path, run_name: str | None = None) -> Path:
    root = Path(log_root)
    if run_name:
        candidate = Path(run_name)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists() and candidate.is_dir():
            return candidate

        # SB3 会在 tb_log_name 后追加 _1、_2 等后缀
        # 因此精确匹配失败时按前缀查找最新的匹配目录
        run_dirs = list_run_directories(root)
        run_name_lower = candidate.name.lower()
        matching = [
            d
            for d in run_dirs
            if d.name.lower() == run_name_lower
            or d.name.lower().startswith(run_name_lower + "_")
        ]
        if matching:
            return matching[-1]
        raise FileNotFoundError(f"Run directory not found: {candidate}")

    run_dirs = list_run_directories(root)
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard run directories found in: {root}")
    return run_dirs[-1]


def _sort_and_keep_latest_by_step(
    steps: np.ndarray,
    values: np.ndarray,
    wall_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps.size == 0:
        return steps, values, wall_times

    order = np.argsort(steps, kind="stable")
    steps_sorted = steps[order]
    values_sorted = values[order]
    wall_times_sorted = wall_times[order]

    rev_steps = steps_sorted[::-1]
    _, rev_unique_indices = np.unique(rev_steps, return_index=True)
    keep = steps_sorted.size - 1 - rev_unique_indices
    keep.sort()

    return steps_sorted[keep], values_sorted[keep], wall_times_sorted[keep]


def load_scalar_series_from_run(run_dir: str | Path) -> dict[str, ScalarSeries]:
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    accumulator = event_accumulator.EventAccumulator(
        str(run_path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _ = accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    if not isinstance(scalar_tags, list):
        scalar_tags = []
    series_map: dict[str, ScalarSeries] = {}

    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        if not events:
            continue

        steps = np.asarray([event.step for event in events], dtype=np.int64)
        values = np.asarray([event.value for event in events], dtype=np.float64)
        wall_times = np.asarray([event.wall_time for event in events], dtype=np.float64)
        steps, values, wall_times = _sort_and_keep_latest_by_step(
            steps,
            values,
            wall_times,
        )
        if steps.size == 0:
            continue

        series_map[tag] = ScalarSeries(
            tag=tag,
            steps=steps,
            values=values,
            wall_times=wall_times,
        )

    return series_map


def compute_sampling_health(
    series_map: dict[str, ScalarSeries],
    *,
    key_tags: list[str] | None = None,
) -> dict[str, Any]:
    series_map = with_legacy_info_tag_aliases(series_map)
    tags = key_tags or DEFAULT_SAMPLING_HEALTH_TAGS
    available_tags = [tag for tag in tags if tag in series_map]

    if not available_tags:
        return {
            "available": False,
            "total_step_span": 0.0,
            "tag_metrics": {},
            "summary": {},
        }

    global_min_step = min(int(np.min(series_map[tag].steps)) for tag in available_tags)
    global_max_step = max(int(np.max(series_map[tag].steps)) for tag in available_tags)
    total_step_span = max(1, global_max_step - global_min_step)

    tag_metrics: dict[str, dict[str, float]] = {}
    samples_per_10k_values: list[float] = []
    mean_gap_values: list[float] = []
    p95_gap_values: list[float] = []
    max_gap_values: list[float] = []

    for tag in available_tags:
        steps = series_map[tag].steps.astype(np.int64)
        sample_count = int(steps.size)
        if sample_count <= 1:
            mean_gap = 0.0
            p95_gap = 0.0
            max_gap = 0.0
        else:
            gaps = np.diff(steps).astype(np.float64)
            mean_gap = float(np.mean(gaps))
            p95_gap = float(np.quantile(gaps, 0.95))
            max_gap = float(np.max(gaps))

        samples_per_10k = float(sample_count) * 10000.0 / float(total_step_span)
        samples_per_10k_values.append(samples_per_10k)
        mean_gap_values.append(mean_gap)
        p95_gap_values.append(p95_gap)
        max_gap_values.append(max_gap)

        tag_metrics[tag] = {
            "sample_count": float(sample_count),
            "mean_step_gap": mean_gap,
            "p95_step_gap": p95_gap,
            "max_step_gap": max_gap,
            "samples_per_10k_steps": samples_per_10k,
            "step_start": float(int(steps[0])) if sample_count > 0 else 0.0,
            "step_end": float(int(steps[-1])) if sample_count > 0 else 0.0,
        }

    summary = {
        "observed_tag_count": float(len(available_tags)),
        "min_sample_count": float(
            min(int(tag_metrics[tag]["sample_count"]) for tag in available_tags)
        ),
        "mean_samples_per_10k_steps": float(np.mean(samples_per_10k_values)),
        "max_mean_step_gap": float(np.max(mean_gap_values)),
        "max_p95_step_gap": float(np.max(p95_gap_values)),
        "max_max_step_gap": float(np.max(max_gap_values)),
    }

    return {
        "available": True,
        "total_step_span": float(total_step_span),
        "tag_metrics": tag_metrics,
        "summary": summary,
    }
