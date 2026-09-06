import pytest

from rl.experiment_utils import (
    DEFAULT_CURRICULUM_PROFILE_NAME,
    DEFAULT_REWARD_PRESET_NAME,
    dspdl_protocol_parameters,
    resolve_training_run_spec,
)
from scripts.train_rl import build_cli_parser


def test_training_cli_uses_rollout_evaluation_interval() -> None:
    args = build_cli_parser().parse_args([])

    assert args.evaluation_interval_rollouts == 12
    assert args.curriculum_profile == DEFAULT_CURRICULUM_PROFILE_NAME
    assert args.curriculum_profile == "dspdl"
    assert args.reward_preset == DEFAULT_REWARD_PRESET_NAME
    assert args.reward_preset == "basic_safety_punctuality"
    assert not hasattr(args, "evaluation_trigger_mode")
    assert not hasattr(args, "evaluation_trigger_interval")


@pytest.mark.parametrize(
    "removed_option",
    ("--evaluation-trigger-mode", "--evaluation-trigger-interval"),
)
def test_training_cli_rejects_removed_evaluation_options(
    removed_option: str,
) -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args([removed_option, "steps"])


def test_training_cli_rejects_survival_reward_scale() -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(["--survival-reward-scale", "50"])


def test_training_cli_rejects_removed_bayesian_profile() -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(
            ["--curriculum-profile", "dspdl_completion_bayes"]
        )


def test_training_cli_rejects_removed_completion_profile_and_option() -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(
            ["--curriculum-profile", "dspdl_completion_nn"]
        )
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(["--completion-alpha-max", "0.05"])


def test_training_cli_accepts_dspdl_profile() -> None:
    args = build_cli_parser().parse_args(["--curriculum-profile", "dspdl"])
    assert args.curriculum_profile == "dspdl"


def test_training_cli_uses_fixed_punctuality_potential_parameters() -> None:
    args = build_cli_parser().parse_args(["--reference-curve-dir", "."])
    spec = resolve_training_run_spec(args)
    assert spec.reward_config.enable_potential_punctuality
    assert not hasattr(spec.reward_config, "punctuality_potential_scale")
    assert not hasattr(spec.reward_config, "punctuality_potential_sigma_s")
    assert spec.run_metadata.reward_config.punctuality_potential_scale == 5.0
    assert spec.run_metadata.reward_config.punctuality_potential_sigma_s == 20.0


@pytest.mark.parametrize(
    "removed_option",
    (
        "--punctuality-potential-scale",
        "--punctuality-potential-sigma-s",
        "--critic-zeta",
        "--critic-relative-entropy-bound",
        "--critic-update-interval-rollouts",
        "--critic-alpha-warmup-updates",
        "--critic-context-value-estimator",
    ),
)
def test_training_cli_rejects_retired_tuning_options(removed_option: str) -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args([removed_option, "1"])


def test_training_metadata_records_fixed_dspdl_protocol() -> None:
    args = build_cli_parser().parse_args(["--reference-curve-dir", "."])
    spec = resolve_training_run_spec(args)
    assert spec.run_metadata.curriculum.dspdl_protocol == dspdl_protocol_parameters()


@pytest.mark.parametrize(
    "removed_option",
    (
        "--bayes-forgetting-factor",
        "--bayes-quality-ema-alpha",
        "--bayes-graph-smoothing-lambda",
        "--bayes-temporal-smoothing-tau",
    ),
)
def test_training_cli_rejects_removed_bayesian_options(removed_option: str) -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args([removed_option, "0.2"])
