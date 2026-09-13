import pytest

from rl.experiment_utils import (
    DEFAULT_CURRICULUM_PROFILE_NAME,
    DEFAULT_REWARD_PRESET_NAME,
    dspl_protocol_parameters,
    resolve_training_run_spec,
)
from scripts.train_rl import build_cli_parser


def test_training_cli_uses_rollout_evaluation_interval() -> None:
    args = build_cli_parser().parse_args([])

    spec = resolve_training_run_spec(
        build_cli_parser().parse_args(["--curriculum-profile", "none"])
    )
    assert args.evaluation_interval_rollouts is None
    assert spec.evaluation_interval_rollouts == 12
    assert spec.evaluation_interval_episodes is None
    assert args.curriculum_profile == DEFAULT_CURRICULUM_PROFILE_NAME
    assert args.curriculum_profile == "dspl"
    assert args.reward_preset == DEFAULT_REWARD_PRESET_NAME
    assert args.reward_preset == "basic_safety_punctuality"
    assert not hasattr(args, "evaluation_trigger_mode")
    assert not hasattr(args, "evaluation_trigger_interval")


def test_training_cli_accepts_episode_evaluation_interval() -> None:
    args = build_cli_parser().parse_args(
        [
            "--curriculum-profile",
            "none",
            "--evaluation-interval-episodes",
            "100",
        ]
    )
    spec = resolve_training_run_spec(args)

    assert spec.evaluation_interval_rollouts is None
    assert spec.evaluation_interval_episodes == 100


def test_training_cli_accepts_environment_step_budget() -> None:
    args = build_cli_parser().parse_args(
        [
            "--budget-mode",
            "environment_steps",
            "--training-rollouts",
            "500",
            "--curriculum-profile",
            "none",
        ]
    )

    spec = resolve_training_run_spec(args)

    assert spec.training_episodes is None
    assert spec.training_rollouts == 500
    assert spec.total_timesteps == 500 * spec.rollout_steps_per_update
    assert spec.run_metadata.training_budget is not None
    assert spec.run_metadata.training_budget.mode == "environment_steps"
    assert "budget_mode" not in spec.run_metadata.to_mapping()


@pytest.mark.parametrize(
    "arguments",
    (
        ["--budget-mode", "environment_steps", "--training-episodes", "5"],
        ["--budget-mode", "completed_episodes", "--training-rollouts", "5"],
    ),
)
def test_training_budget_modes_reject_the_other_budget_unit(arguments) -> None:
    args = build_cli_parser().parse_args([*arguments, "--curriculum-profile", "none"])
    with pytest.raises(ValueError):
        resolve_training_run_spec(args)


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
            ["--curriculum-profile", "dspl_completion_bayes"]
        )


def test_training_cli_rejects_removed_completion_profile_and_option() -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(
            ["--curriculum-profile", "dspl_completion_nn"]
        )
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(["--completion-alpha-max", "0.05"])


def test_training_cli_accepts_dspl_profile() -> None:
    args = build_cli_parser().parse_args(["--curriculum-profile", "dspl"])
    assert args.curriculum_profile == "dspl"


def test_training_cli_rejects_removed_context_coverage_option() -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args(["--dspl-context-coverage-constant", "2"])


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


def test_training_metadata_records_fixed_dspl_protocol() -> None:
    args = build_cli_parser().parse_args(["--reference-curve-dir", "."])
    spec = resolve_training_run_spec(args)
    assert spec.run_metadata.curriculum.dspl_protocol == dspl_protocol_parameters()


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
