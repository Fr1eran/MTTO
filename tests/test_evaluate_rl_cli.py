from scripts.evaluate_rl import build_arg_parser


def test_evaluate_rl_cli_accepts_dry_run_and_shared_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dry-run",
        "--schedule-time-s",
        "430.0",
        "--reward-profile",
        "basic_safety",
        "--reward-discount",
        "0.95",
        "--device",
        "cuda",
    ])

    assert args.dry_run is True
    assert args.schedule_time_s == 430.0
    assert args.reward_profile == "basic_safety"
    assert args.reward_discount == 0.95
    assert args.device == "cuda"
