from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mtto.rl.diagnostics import TOTAL_REWARD_INDEX, RewardDiagnostics


@dataclass(frozen=True)
class ScalarSeries:
    tag: str
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray


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
    termination_reason: np.ndarray


def extract_complete_episode_sequence(
    diagnostics: RewardDiagnostics,
) -> CompleteEpisodeSequence:
    """Return every complete episode with a global cumulative episode index.

    Worker-local episode indices are not globally unique.  Ordering by end
    step, worker rank, and local index yields a deterministic merged sequence
    suitable for an episode-number learning-curve axis.
    """
    complete = diagnostics.episode_complete
    end_steps = diagnostics.episode_end_step[complete]
    worker_ranks = diagnostics.episode_worker_rank[complete]
    worker_indices = diagnostics.episode_index[complete]
    rewards = diagnostics.episode_reward_sums[complete, TOTAL_REWARD_INDEX]
    lengths = diagnostics.episode_length[complete].astype(np.float64)
    termination_reasons = diagnostics.episode_termination_reason[complete]
    if end_steps.size == 0:
        return CompleteEpisodeSequence(
            episode_number=np.empty(0, dtype=np.int64),
            total_reward=np.empty(0, dtype=np.float64),
            length=np.empty(0, dtype=np.float64),
            termination_reason=np.empty(0, dtype=np.int8),
        )

    order = np.lexsort((worker_indices, worker_ranks, end_steps))
    episode_count = end_steps.size
    return CompleteEpisodeSequence(
        episode_number=np.arange(1, episode_count + 1, dtype=np.int64),
        total_reward=np.asarray(rewards[order], dtype=np.float64),
        length=np.asarray(lengths[order], dtype=np.float64),
        termination_reason=np.asarray(termination_reasons[order], dtype=np.int8),
    )


def extract_complete_episode_series(
    diagnostics: RewardDiagnostics,
) -> CompleteEpisodeSeries:
    """Build a strictly ordered curve, averaging episodes ending at one step."""
    complete = diagnostics.episode_complete
    steps = diagnostics.episode_end_step[complete]
    rewards = diagnostics.episode_reward_sums[complete, TOTAL_REWARD_INDEX]
    lengths = diagnostics.episode_length[complete].astype(np.float64)
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


DEFAULT_SAMPLING_HEALTH_TAGS = [
    "rollout/ep_rew_mean",
    "train/approx_kl",
]


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


def compute_sampling_health(
    series_map: dict[str, ScalarSeries],
    *,
    key_tags: list[str] | None = None,
) -> dict[str, Any]:
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
