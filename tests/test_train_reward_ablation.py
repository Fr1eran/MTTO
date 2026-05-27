from pathlib import Path

import pytest

from scripts.train_reward_ablation import (
    DEFAULT_ABLATION_REWARD_PROFILES,
    build_arg_parser,
    resolve_ablation_run_matrix,
    resolve_ablation_seeds,
)


def test_ablation_cli_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.ablation_output_root == "output/optimal/rl/reward_ablation"
    assert args.reward_profiles is None
    assert args.repeats == 1
    assert args.base_seed is None
    assert args.dry_run is False
    assert not hasattr(args, "subproc_start_method")


def test_ablation_cli_hides_train_rl_run_mode_and_overrides() -> None:
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--run-mode", "monitor_best"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--enable-best-eval"])


def test_ablation_cli_rejects_removed_subproc_start_method_option() -> None:
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--subproc-start-method", "spawn"])


def test_ablation_cli_rejects_removed_rollout_record_trigger_mode_option() -> None:
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-record-trigger-mode", "episodes"])


def test_resolve_ablation_seeds_prefers_seed_list() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--repeats",
        "3",
        "--seed-list",
        "11",
        "12",
    ])

    assert resolve_ablation_seeds(args) == [11, 12]


def test_resolve_ablation_seeds_requires_source_for_multiple_repeats() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--repeats", "2"])

    with pytest.raises(ValueError, match="--base-seed 或 --seed-list"):
        resolve_ablation_seeds(args)


def test_resolve_ablation_run_matrix_enforces_monitor_best_semantics() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--base-seed",
        "10",
        "--repeats",
        "2",
        "--ablation-tag",
        "trial_a",
        "--log-interval",
        "3",
    ])

    run_entries = resolve_ablation_run_matrix(args)

    assert len(run_entries) == len(DEFAULT_ABLATION_REWARD_PROFILES) * 2
    first_entry = run_entries[0]
    fifth_entry = run_entries[len(DEFAULT_ABLATION_REWARD_PROFILES)]

    assert first_entry.reward_profile_name == "basic"
    assert first_entry.repeat_index == 0
    assert first_entry.seed == 10
    assert first_entry.train_args.run_mode == "monitor_best"
    assert first_entry.training_run_spec.enable_tb is True
    assert first_entry.training_run_spec.enable_callback is False
    assert first_entry.training_run_spec.enable_best_eval is True
    assert first_entry.training_run_spec.log_interval == 3
    assert first_entry.train_args.rollout_record_trigger_mode == "steps"
    assert (
        Path(first_entry.training_run_spec.output_dir).name
        == "430p0_30p0__basic__trial_a_r01"
    )

    assert fifth_entry.reward_profile_name == "basic"
    assert fifth_entry.repeat_index == 1
    assert fifth_entry.seed == 11


def test_resolve_ablation_run_matrix_allows_subset_profiles() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--reward-profiles",
        "basic_safety",
        "basic_safety_stopping",
        "--base-seed",
        "5",
    ])

    run_entries = resolve_ablation_run_matrix(args)

    assert [entry.reward_profile_name for entry in run_entries] == [
        "basic_safety",
        "basic_safety_stopping",
    ]

