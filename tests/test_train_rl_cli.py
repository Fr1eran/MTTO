import pytest

from scripts.train_rl import build_cli_parser


def test_training_cli_uses_rollout_evaluation_interval() -> None:
    args = build_cli_parser().parse_args([])

    assert args.evaluation_interval_rollouts == 12
    assert args.training_episodes == 7000
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


@pytest.mark.parametrize(
    "removed_option",
    ("--total-timesteps", "--target-completed-episodes"),
)
def test_training_cli_rejects_removed_timestep_budget_options(
    removed_option: str,
) -> None:
    with pytest.raises(SystemExit):
        _ = build_cli_parser().parse_args([removed_option, "1"])
