import json
from pathlib import Path

import numpy as np
import pytest

from scripts.show_reward_ablation import (
    ABLATION_MANIFEST_FILENAME,
    EPISODE_METRICS_FILENAME,
    SAFETY_VIOLATION_BINS_FILENAME,
    build_arg_parser,
    build_curve_aggregates,
    build_safety_violation_bin_aggregates,
    load_ablation_manifest,
    panel_label_for_index,
    select_representative_trajectory_candidates,
)


def _write_episode_metrics(
    final_output_dir: Path,
    *,
    steps: list[float],
    rewards: list[float],
    lengths: list[float],
) -> Path:
    final_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_output_dir / EPISODE_METRICS_FILENAME
    np.savez(
        output_path,
        index=np.asarray(steps, dtype=np.float64),
        ep_reward=np.asarray(rewards, dtype=np.float64),
        ep_len=np.asarray(lengths, dtype=np.float64),
    )
    return output_path


def _write_safety_violation_bins(
    final_output_dir: Path,
    *,
    bin_start_m: list[float],
    bin_end_m: list[float],
    sample_violation_rate: list[float],
    episode_violation_rate: list[float],
    sample_exposure_count: list[float] | None = None,
    sample_violation_count: list[float] | None = None,
    episode_exposure_count: list[float] | None = None,
    episode_violation_count: list[float] | None = None,
) -> Path:
    final_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_output_dir / SAFETY_VIOLATION_BINS_FILENAME
    payload = {
        "bin_start_m": np.asarray(bin_start_m, dtype=np.float64),
        "bin_end_m": np.asarray(bin_end_m, dtype=np.float64),
        "sample_violation_rate": np.asarray(sample_violation_rate, dtype=np.float64),
        "episode_violation_rate": np.asarray(episode_violation_rate, dtype=np.float64),
    }
    if sample_exposure_count is not None:
        payload["sample_exposure_count"] = np.asarray(
            sample_exposure_count, dtype=np.float64
        )
    if sample_violation_count is not None:
        payload["sample_violation_count"] = np.asarray(
            sample_violation_count, dtype=np.float64
        )
    if episode_exposure_count is not None:
        payload["episode_exposure_count"] = np.asarray(
            episode_exposure_count, dtype=np.float64
        )
    if episode_violation_count is not None:
        payload["episode_violation_count"] = np.asarray(
            episode_violation_count, dtype=np.float64
        )
    np.savez(output_path, allow_pickle=True, **payload)
    return output_path


def _write_trajectory_artifact(
    output_dir: Path,
    *,
    source_dir_name: str,
    metrics: dict[str, object],
) -> None:
    run_dir = output_dir / source_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _ = (run_dir / "best_trajectory.npz").write_bytes(b"curve")
    _ = (run_dir / "best_trajectory_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )


def _build_manifest(tmp_path: Path) -> tuple[dict[str, object], Path]:
    ablation_root = tmp_path / "ablation"
    run_basic_r1 = ablation_root / "430p0_100p0__basic__r01"
    run_basic_r2 = ablation_root / "430p0_100p0__basic__r02"
    run_safety_r1 = ablation_root / "430p0_100p0__basic_safety__r01"

    _ = _write_episode_metrics(
        run_basic_r1 / "final",
        steps=[0, 100, 200],
        rewards=[1.0, 2.0, 3.0],
        lengths=[10.0, 9.0, 8.0],
    )
    _ = _write_episode_metrics(
        run_basic_r2 / "final",
        steps=[0, 200],
        rewards=[2.0, 6.0],
        lengths=[12.0, 6.0],
    )
    _ = _write_episode_metrics(
        run_safety_r1 / "final",
        steps=[0, 100, 200],
        rewards=[3.0, 4.0, 5.0],
        lengths=[8.0, 7.0, 6.0],
    )
    _ = _write_safety_violation_bins(
        run_basic_r1 / "final",
        bin_start_m=[0, 250, 500, 750, 1000],
        bin_end_m=[250, 500, 750, 1000, 1500],
        sample_violation_rate=[0.2, 0.2, 0.4, 0.4, 0.0],
        episode_violation_rate=[0.1, 0.3, 0.4, 0.6, 0.0],
        sample_exposure_count=[10, 10, 10, 10, 10],
        sample_violation_count=[2, 2, 4, 4, 0],
        episode_exposure_count=[10, 10, 10, 10, 10],
        episode_violation_count=[1, 3, 4, 6, 0],
    )
    _ = _write_safety_violation_bins(
        run_basic_r2 / "final",
        bin_start_m=[0, 1000],
        bin_end_m=[500, 1500],
        sample_violation_rate=[0.4, 0.2],
        episode_violation_rate=[0.3, 0.1],
        sample_exposure_count=[10, 10],
        sample_violation_count=[4, 2],
        episode_exposure_count=[10, 10],
        episode_violation_count=[3, 1],
    )
    _ = _write_safety_violation_bins(
        run_safety_r1 / "final",
        bin_start_m=[0, 500, 1000],
        bin_end_m=[500, 1000, 1500],
        sample_violation_rate=[0.1, 0.2, 0.0],
        episode_violation_rate=[0.0, 0.1, 0.0],
        sample_exposure_count=[10, 10, 10],
        sample_violation_count=[1, 2, 0],
        episode_exposure_count=[10, 10, 10],
        episode_violation_count=[0, 1, 0],
    )

    _ = _write_run_metadata(
        run_basic_r1 / "final",
        {"rollout_record_trigger_mode": "steps"},
    )
    _ = _write_run_metadata(
        run_basic_r2 / "final",
        {"rollout_record_trigger_mode": "steps"},
    )
    _ = _write_run_metadata(
        run_safety_r1 / "final",
        {"rollout_record_trigger_mode": "steps"},
    )

    _write_trajectory_artifact(
        run_basic_r1,
        source_dir_name="best_steps",
        metrics={
            "reward_profile_name": "basic",
            "success": True,
            "total_reward": 5.0,
            "total_energy_j": 5_000.0,
            "stop_error_m": 0.2,
            "time_error_s": 0.5,
            "strict_stop_error_limit_m": 0.3,
            "strict_time_error_limit_s": 22.0,
        },
    )
    _write_trajectory_artifact(
        run_basic_r2,
        source_dir_name="best_steps",
        metrics={
            "reward_profile_name": "basic",
            "success": True,
            "total_reward": 2.0,
            "total_energy_j": 4_000.0,
            "stop_error_m": 0.2,
            "time_error_s": 0.5,
            "strict_stop_error_limit_m": 0.3,
            "strict_time_error_limit_s": 22.0,
        },
    )
    _write_trajectory_artifact(
        run_safety_r1,
        source_dir_name="best_steps",
        metrics={
            "reward_profile_name": "basic_safety",
            "success": False,
            "total_reward": 8.0,
            "total_energy_j": 9_000.0,
            "stop_error_m": 1.5,
            "time_error_s": 10.0,
            "strict_stop_error_limit_m": 0.3,
            "strict_time_error_limit_s": 22.0,
        },
    )

    manifest: dict[str, object] = {
        "ablation_output_root": str(ablation_root),
        "reward_profiles": ["basic", "basic_safety"],
        "runs": [
            {
                "reward_profile_name": "basic",
                "repeat_index": 0,
                "seed": 10,
                "output_dir": str(run_basic_r1),
                "final_output_dir": str(run_basic_r1 / "final"),
                "best_eval_output_dir": str(run_basic_r1 / "best_steps"),
                "status": "completed",
            },
            {
                "reward_profile_name": "basic",
                "repeat_index": 1,
                "seed": 11,
                "output_dir": str(run_basic_r2),
                "final_output_dir": str(run_basic_r2 / "final"),
                "best_eval_output_dir": str(run_basic_r2 / "best_steps"),
                "status": "completed",
            },
            {
                "reward_profile_name": "basic_safety",
                "repeat_index": 0,
                "seed": 12,
                "output_dir": str(run_safety_r1),
                "final_output_dir": str(run_safety_r1 / "final"),
                "best_eval_output_dir": str(run_safety_r1 / "best_steps"),
                "status": "completed",
            },
        ],
    }
    manifest_path = ablation_root / ABLATION_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path


def test_show_reward_ablation_cli_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.ablation_root == "output/optimal/rl/reward_ablation"
    assert args.trajectory_source == "best"
    assert args.trajectory_layout == "separate"
    assert args.violation_rate_metric == "episode"
    assert args.no_safety_violation_boxplot is False
    assert args.dry_run is False


def test_load_ablation_manifest_reads_expected_file(tmp_path: Path) -> None:
    manifest, manifest_path = _build_manifest(tmp_path)

    loaded = load_ablation_manifest(str(manifest_path.parent))

    assert loaded == manifest


def test_build_curve_aggregates_aligns_repeats_by_steps(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)

    aggregates, warnings = build_curve_aggregates(manifest)

    assert warnings == []
    assert [aggregate.reward_profile_name for aggregate in aggregates] == [
        "basic",
        "basic_safety",
    ]

    basic_aggregate = aggregates[0]
    assert basic_aggregate.valid_repeat_count == 2
    np.testing.assert_allclose(basic_aggregate.reference_steps, [0.0, 100.0, 200.0])
    np.testing.assert_allclose(basic_aggregate.mean_reward, [1.5, 3.0, 4.5])
    np.testing.assert_allclose(basic_aggregate.mean_length, [11.0, 9.0, 7.0])


def test_build_safety_violation_bin_aggregates_aligns_repeats_by_position_bin(
    tmp_path: Path,
) -> None:
    manifest, _ = _build_manifest(tmp_path)

    aggregates, warnings = build_safety_violation_bin_aggregates(manifest)

    assert warnings == []
    assert [aggregate.reward_profile_name for aggregate in aggregates] == [
        "basic",
        "basic_safety",
    ]

    basic_aggregate = aggregates[0]
    assert basic_aggregate.valid_repeat_count == 2
    np.testing.assert_allclose(basic_aggregate.bin_start_m, [0.0])
    np.testing.assert_allclose(basic_aggregate.bin_end_m, [5000.0])
    assert basic_aggregate.display_bin_size_m == 5000.0
    np.testing.assert_allclose(
        basic_aggregate.mean_violation_rate,
        [0.24],
    )
    np.testing.assert_allclose(
        basic_aggregate.var_violation_rate,
        [0.0016],
    )

    sample_aggregates, sample_warnings = build_safety_violation_bin_aggregates(
        manifest,
        rate_metric="sample",
    )
    assert sample_warnings == []
    np.testing.assert_allclose(
        sample_aggregates[0].mean_violation_rate,
        [0.27],
    )


def test_build_safety_violation_bin_aggregates_warns_for_missing_artifact(
    tmp_path: Path,
) -> None:
    ablation_root = tmp_path / "ablation_missing_safety"
    run_dir = ablation_root / "430p0_100p0__basic__r01"

    manifest = {
        "ablation_output_root": str(ablation_root),
        "reward_profiles": ["basic"],
        "runs": [
            {
                "reward_profile_name": "basic",
                "repeat_index": 0,
                "seed": 10,
                "output_dir": str(run_dir),
                "final_output_dir": str(run_dir / "final"),
                "best_eval_output_dir": str(run_dir / "best_steps"),
                "status": "completed",
            }
        ],
    }

    aggregates, warnings = build_safety_violation_bin_aggregates(manifest)

    assert aggregates == []
    assert any("artifact is missing" in warning for warning in warnings)
    assert any(
        "No valid safety violation bin artifacts" in warning for warning in warnings
    )


def test_select_representative_trajectory_candidates_uses_existing_comparison_key(
    tmp_path: Path,
) -> None:
    manifest, _ = _build_manifest(tmp_path)

    selected, warnings = select_representative_trajectory_candidates(
        manifest,
        trajectory_source="best_steps",
    )

    assert warnings == []
    assert [candidate.reward_profile_name for candidate in selected] == [
        "basic",
        "basic_safety",
    ]
    assert selected[0].repeat_index == 1
    assert selected[0].seed == 11
    assert selected[0].metrics["total_energy_j"] == 4_000.0


def _write_run_metadata(
    output_dir: Path,
    metadata: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    _ = metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_build_curve_aggregates_rejects_episode_mode_artifacts(tmp_path: Path) -> None:
    ablation_root = tmp_path / "ablation_ep"
    run_dir = ablation_root / "430p0_100p0__basic__r01"

    _ = _write_episode_metrics(
        run_dir / "final",
        steps=[0, 1, 2],
        rewards=[1.0, 3.0, 5.0],
        lengths=[10.0, 9.0, 8.0],
    )
    _ = _write_run_metadata(
        run_dir / "final",
        {"rollout_record_trigger_mode": "episodes"},
    )

    manifest = {
        "ablation_output_root": str(ablation_root),
        "reward_profiles": ["basic"],
        "runs": [
            {
                "reward_profile_name": "basic",
                "repeat_index": 0,
                "seed": 10,
                "output_dir": str(run_dir),
                "final_output_dir": str(run_dir / "final"),
                "status": "completed",
            }
        ],
    }

    with pytest.raises(ValueError, match="no longer supports episodes-based"):
        _ = build_curve_aggregates(manifest)


def test_build_curve_aggregates_rejects_missing_record_mode(tmp_path: Path) -> None:
    ablation_root = tmp_path / "ablation_missing_mode"
    run_dir = ablation_root / "430p0_100p0__basic__r01"

    _ = _write_episode_metrics(
        run_dir / "final",
        steps=[0, 100, 200],
        rewards=[1.0, 2.0, 3.0],
        lengths=[10.0, 9.0, 8.0],
    )
    _ = _write_run_metadata(run_dir / "final", {"reward_profile_name": "basic"})

    manifest = {
        "ablation_output_root": str(ablation_root),
        "reward_profiles": ["basic"],
        "runs": [
            {
                "reward_profile_name": "basic",
                "repeat_index": 0,
                "seed": 10,
                "output_dir": str(run_dir),
                "final_output_dir": str(run_dir / "final"),
                "status": "completed",
            }
        ],
    }

    with pytest.raises(ValueError, match="rollout_record_trigger_mode='steps'"):
        _ = build_curve_aggregates(manifest)


def test_build_curve_aggregates_rejects_invalid_record_mode(tmp_path: Path) -> None:
    ablation_root = tmp_path / "ablation_invalid_mode"
    run_dir = ablation_root / "430p0_100p0__basic__r01"

    _ = _write_episode_metrics(
        run_dir / "final",
        steps=[0, 100, 200],
        rewards=[1.0, 2.0, 3.0],
        lengths=[10.0, 9.0, 8.0],
    )
    _ = _write_run_metadata(
        run_dir / "final",
        {"rollout_record_trigger_mode": "invalid"},
    )

    manifest = {
        "ablation_output_root": str(ablation_root),
        "reward_profiles": ["basic"],
        "runs": [
            {
                "reward_profile_name": "basic",
                "repeat_index": 0,
                "seed": 10,
                "output_dir": str(run_dir),
                "final_output_dir": str(run_dir / "final"),
                "status": "completed",
            }
        ],
    }

    with pytest.raises(ValueError, match="rollout_record_trigger_mode='steps'"):
        _ = build_curve_aggregates(manifest)


def test_panel_label_for_index_uses_sci_style() -> None:
    assert panel_label_for_index(0) == "(a)"
    assert panel_label_for_index(3) == "(d)"
