from scripts.benchmark_tune import build_arg_parser, run_environment_benchmark


def test_benchmark_parser_defaults() -> None:
    args = build_arg_parser().parse_args([])
    assert args.steps == 8192
    assert args.rollout_capacity == 2048


def test_environment_benchmark_reports_compact_telemetry_metrics() -> None:
    result = run_environment_benchmark(steps=4, rollout_capacity=2)

    assert result["steps"] == 4.0
    assert result["steps_per_s"] > 0.0
    assert result["mean_info_top_level_fields"] <= 3.0
    assert result["rollout_drain_count"] == 2.0
