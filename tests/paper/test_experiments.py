"""Small real training matrix and recovery regressions."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mtto.io.artifacts import read_completed_run
from mtto.io.scenario import load_tasks
from paper import analysis
from paper.experiments import runner, schedule_change, step_distance
from paper.experiments.method_ablation import summarize, write_summary
from paper.experiments.spec import expand_matrix, load_experiment_spec
from paper.plotting.ablation import (
    method_figures,
    require_clean,
    schedule_change_figure,
    step_distance_figure,
)


@pytest.fixture(scope="module")
def small_spec(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("method_ablation")
    tasks = root / "tasks.toml"
    tasks.write_text(
        "[short]\nstart_position_m = 135.0\ntarget_position_m = 1135.0\n"
        "schedule_time_s = 120.0\nmax_acc_change = 0.75\n"
        "max_stop_error_m = 0.3\nmax_arr_time_error_s = 10.0\n",
        encoding="utf-8",
    )
    spec = root / "method.toml"
    repo = Path(__file__).resolve().parents[2]
    spec.write_text(
        "[experiment]\nname = 'method'\n"
        f"scenario = '{repo / 'paper/specs/scenario.toml'}'\n"
        f"line_dir = '{repo / 'paper/data/line'}'\n"
        f"tasks = '{tasks}'\ntask = 'short'\n"
        f"output_root = '{root / 'runs'}'\nseeds = [11]\n\n"
        "[train]\nstep_distance_m = 100.0\ngamma = 0.998\n"
        "budget_mode = 'environment_steps'\n"
        "training_rollouts = 1\nnum_envs = 1\nn_steps_per_env = 512\n"
        "evaluation_deterministic = true\nkeep_best = true\n"
        "safety_truncation_bin_size_m = 5000.0\ndevice = 'cpu'\n\n"
        "[[variants]]\nid = 'ppo'\nlabel = 'PPO'\nreward_preset = 'basic'\n\n"
        "[[variants]]\nid = 'ppo_safety'\nlabel = 'PPO+Safety'\n"
        "reward_preset = 'basic_safety'\n",
        encoding="utf-8",
    )
    return spec


def test_strict_spec_and_matrix(small_spec: Path) -> None:
    spec = load_experiment_spec(small_spec)
    assert spec.train["training_episodes"] is None
    assert spec.train["evaluation_interval_rollouts"] is None
    assert spec.train["evaluation_interval_episodes"] is None
    assert [run.run_label for run in expand_matrix(spec)] == [
        "method__ppo__seed0011",
        "method__ppo_safety__seed0011",
    ]
    source = small_spec.read_text(encoding="utf-8")
    for bad in (
        source.replace("gamma = 0.998\n", ""),
        source.replace("gamma = 0.998", "gamma = 0.998\nextra = 1"),
    ):
        path = small_spec.with_name("invalid.toml")
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_experiment_spec(path)


@pytest.mark.parametrize("entrypoint", ["execute_matrix", "completed_matrix"])
def test_io_error_does_not_discard_run(
    small_spec: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    spec = replace(load_experiment_spec(small_spec), output_root=tmp_path)
    run_dir = tmp_path / f"{expand_matrix(spec)[0].run_label}__01"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", True))

    def fail_read(_path: Path) -> None:
        raise OSError("transient read failure")

    monkeypatch.setattr(runner, "read_completed_run", fail_read)
    with pytest.raises(OSError, match="transient read failure"):
        getattr(runner, entrypoint)(spec)
    assert run_dir.is_dir()
    assert (run_dir / "run.json").exists()


def test_run_reuse_recovery_and_summary(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_experiment_spec(small_spec)
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", True))
    first = runner.execute_matrix(spec)
    assert len(first) == 2 and all(not item.reused for item in first)
    assert all(item.directory.name.endswith("__01") for item in first)
    for item in first:
        completed = read_completed_run(item.directory)
        paper = json.loads((item.directory / "paper.json").read_text())
        assert paper["reuse_key"] == runner.reuse_key(completed.record)

    original_train = runner.train_workflow.train
    calls = []

    def counting_train(*args: object, **kwargs: object) -> object:
        calls.append(args[3])
        return original_train(*args, **kwargs)

    monkeypatch.setattr(runner.train_workflow, "train", counting_train)
    reused = runner.execute_matrix(spec)
    assert all(item.reused for item in reused) and not calls
    assert len(list(spec.output_root.iterdir())) == 2

    (first[0].directory / "quality.json").unlink()
    recovered = runner.execute_matrix(spec)
    assert len(calls) == 1
    assert recovered[0].directory.name.endswith("__02")
    assert not first[0].directory.exists() and recovered[1].reused

    summary = summarize(spec, tuple(item.directory for item in recovered))
    write_summary(summary, tmp_path / "summary")
    written = json.loads((tmp_path / "summary/summary.json").read_text())
    payload = read_completed_run(recovered[0].directory).payload
    assert written["raw_seed_metrics"]["ppo"][0]["stop_error_m"] == abs(
        (payload.best or payload).quality.metrics.stop_error_m
    )
    performance_table = tmp_path / "summary/method_performance_table.md"
    assert "PPO+Safety" in performance_table.read_text()

    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", False))
    clean = runner.execute_matrix(spec)
    assert len(calls) == 3 and all(not item.reused for item in clean)
    assert recovered[0].directory.exists()
    assert all(item.reused for item in runner.execute_matrix(spec))

    changed = replace(spec, train={**spec.train, "gamma": 0.997})
    changed_runs = runner.execute_matrix(changed)
    assert len(calls) == 5 and all(not item.reused for item in changed_runs)
    assert all(item.directory.exists() for item in clean)
    assert runner.completed_matrix(spec) == tuple(item.directory for item in clean)
    assert runner.completed_matrix(changed) == tuple(
        item.directory for item in changed_runs
    )

    result_path = changed_runs[0].directory / "result.json"
    data = json.loads(result_path.read_text())
    data["training"]["target_reached"] = False
    result_path.write_text(json.dumps(data))
    with pytest.raises(runner.ExperimentStopped):
        runner.execute_matrix(changed)
    assert len(calls) == 5


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "align_exact",
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0]), np.array([5.0, 7.0])),
        ),
        (
            "smooth_episode_curve",
            (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])),
        ),
        ("aggregate_matrix", (np.array([[1.0, 2.0], [3.0, np.nan]]),)),
        (
            "aggregate_indexed_series",
            (
                [
                    (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
                    (np.array([1.0]), np.array([5.0])),
                ],
            ),
        ),
        (
            "aggregate_step_binned_series",
            (
                [
                    (np.array([1.0, 4.0]), np.array([2.0, 6.0])),
                    (np.array([2.0]), np.array([4.0])),
                ],
            ),
        ),
    ],
)
def test_copied_statistics_match_legacy(name: str, args: tuple) -> None:
    kwargs = (
        {"window": 2}
        if name == "smooth_episode_curve"
        else {"bin_width": 2, "axis_max": 4}
        if name == "aggregate_step_binned_series"
        else {}
    )
    result = getattr(analysis, name)(*args, **kwargs)
    fixed = {
        "align_exact": (np.array([5.0, np.nan, 7.0]),),
        "smooth_episode_curve": (np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0])),
        "aggregate_matrix": (
            np.array([2.0, 2.0]),
            np.array([np.sqrt(2.0), 0.0]),
            np.array([2, 1]),
        ),
        "aggregate_indexed_series": (
            np.array([1.0, 2.0]),
            np.array([4.0, 4.0]),
            np.array([np.sqrt(2.0), 0.0]),
            np.array([2, 1]),
        ),
        "aggregate_step_binned_series": (
            np.array([2.0, 4.0]),
            np.array([3.0, 6.0]),
            np.array([np.sqrt(2.0), 0.0]),
            np.array([2, 1]),
        ),
    }
    arrays = result if isinstance(result, tuple) else (result,)
    for current, previous in zip(arrays, fixed[name], strict=True):
        np.testing.assert_allclose(current, previous, equal_nan=True)


@pytest.mark.parametrize("time_error", [2.0, 10.0])
def test_copied_constraint_assessment_matches_legacy(time_error: float) -> None:
    metrics = {
        "termination_reason": "STOPPED_IN_ZONE",
        "stop_error_m": 0.1,
        "time_error_s": time_error,
        "min_safety_margin_mps": 0.2,
        "safety_violation_count": 0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 10.0,
        "success": True,
        "precise_arrival": True,
        "punctual_arrival": time_error < 10.0,
        "safe": True,
        "feasible": time_error < 10.0,
    }
    assert asdict(analysis.assess_constraints(metrics)) == {
        "success": True,
        "precise_arrival": True,
        "punctual_arrival": time_error < 10.0,
        "safe": True,
        "feasible": time_error < 10.0,
        "failure_reasons": () if time_error < 10.0 else ("time_error",),
    }


def test_generic_variants_and_step_distance(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = small_spec.read_text(encoding="utf-8")
    prefix = source.split("[[variants]]")[0]
    prefix = prefix.replace("name = 'method'", "name = 'step_distance'")
    prefix = prefix.replace(
        "device = 'cpu'", "device = 'cpu'\nreward_preset = 'basic_safety_punctuality'"
    )
    prefix = prefix.replace(str(small_spec.parent / "runs"), str(tmp_path / "steps"))
    path = tmp_path / "step.toml"
    path.write_text(
        prefix
        + "[[variants]]\nid = '100p0'\nlabel = '100 m'\nstep_distance_m = 100.0\n\n"
        "[[variants]]\nid = '50p0'\nlabel = '50 m'\nstep_distance_m = 50.0\n",
        encoding="utf-8",
    )
    spec = load_experiment_spec(path)
    assert [planned.config.step_distance_m for planned in expand_matrix(spec)] == [
        100.0,
        50.0,
    ]
    assert [planned.run_label for planned in expand_matrix(spec)] == [
        "step_distance__100p0__seed0011",
        "step_distance__50p0__seed0011",
    ]
    for bad in (
        path.read_text().replace("step_distance_m = 50.0", "seed = 50"),
        path.read_text().replace("step_distance_m = 50.0", "unknown = 50"),
    ):
        invalid = tmp_path / "invalid_step.toml"
        invalid.write_text(bad)
        with pytest.raises(ValueError):
            load_experiment_spec(invalid)
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", False))
    results = step_distance.run(path)
    assert len(results) == 2
    summary = step_distance.summarize(spec, tuple(item.directory for item in results))
    step_distance.write_summary(summary, tmp_path / "step_summary")
    assert (tmp_path / "step_summary/step_distance_summary.json").exists()
    assert "50 m" in (tmp_path / "step_summary/step_distance_table.md").read_text()
    require_clean(tuple(item.directory for item in results))
    from paper.experiments.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper.experiments",
            "step_distance",
            "figures",
            "--spec",
            str(path),
            "--output",
            str(tmp_path / "step_figures"),
        ],
    )
    main()
    assert (tmp_path / "step_figures/step_distance_learning_curves.pdf").exists()


def test_schedule_change_evaluation_reuse_and_figures(
    small_spec: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = small_spec.read_text(encoding="utf-8")
    source = source.split("[[variants]]")[0].replace(
        str(small_spec.parent / "runs"), str(tmp_path / "methods")
    )
    source = source.replace(
        "training_rollouts = 1",
        "training_rollouts = 2\nevaluation_interval_rollouts = 1",
    )
    source += (
        "[[variants]]\nid = 'ppo_pirs'\nlabel = 'PPO+PIRS'\n"
        "reward_preset = 'basic_safety_punctuality'\n"
    )
    method_path = tmp_path / "method.toml"
    method_path.write_text(source)
    monkeypatch.setattr(runner, "git_state", lambda: ("test-commit", False))
    method_spec = load_experiment_spec(method_path)
    training = runner.execute_matrix(method_spec)
    assert len(training) == 1
    assert read_completed_run(training[0].directory).payload.best is not None
    from paper.experiments.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper.experiments",
            "method_ablation",
            "figures",
            "--spec",
            str(method_path),
            "--output",
            str(tmp_path / "cli_figures"),
        ],
    )
    main()
    assert (tmp_path / "cli_figures/method_training_curves.pdf").exists()
    assert (tmp_path / "cli_figures/method_trajectory_metrics.pdf").exists()
    case_spec = tmp_path / "schedule.toml"
    case_spec.write_text(
        "[experiment]\n"
        f"source_spec = '{method_path}'\nsource_variant = 'ppo_pirs'\n"
        "candidate_sources = ['best', 'final']\ndelta_times_s = [0.0, 30.0]\n"
        f"change_distance_m = 500.0\noutput_root = '{tmp_path / 'schedule'}'\n"
    )
    spec = schedule_change.load_schedule_spec(case_spec)
    first = schedule_change.run(case_spec)
    assert len(first) == 4 and all(not item.reused for item in first)
    for item in first:
        completed = read_completed_run(item.directory)
        paper = json.loads((item.directory / "paper.json").read_text())
        policy_file = training[0].directory / (
            "best/policy.zip" if item.planned.config.use_best else "policy.zip"
        )
        assert paper["reuse_key"] == runner.reuse_key(
            completed.record, runner.file_sha256(policy_file)
        )
        if "plus_30p0s" in item.planned.run_label:
            assert completed.record.task["schedule_change"] == {
                "trigger_position_m": 500.0,
                "new_schedule_time_s": 150.0,
            }
    again = schedule_change.run(case_spec)
    assert all(item.reused for item in again)
    summary = schedule_change.summarize(spec, tuple(item.directory for item in again))
    schedule_change.write_summary(summary, tmp_path / "schedule_summary")
    assert summary["candidate_count"] == 2
    assert (tmp_path / "schedule_summary/schedule_time_change_table.md").exists()
    require_clean(tuple(item.directory for item in first))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper.experiments",
            "schedule_change",
            "figures",
            "--spec",
            str(case_spec),
            "--output",
            str(tmp_path / "cli_schedule_figures"),
        ],
    )
    main()
    assert (
        tmp_path / "cli_schedule_figures/schedule_time_change_comparison.pdf"
    ).exists()
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    try:
        scenario, _ = schedule_change.planned_evaluations(spec)
        schedule_change_figure(summary, scenario, tmp_path / "schedule_axis")
    finally:
        monkeypatch.setattr(plt, "close", close)
        for figure in figures:
            close(figure)
    profiles = [
        read_completed_run(Path(case["run_dir"])).payload.profile
        for case in summary["cases"]
    ]
    left = min(float(np.min(profile.position_m)) for profile in profiles)
    right = max(float(np.max(profile.position_m)) for profile in profiles)
    margin = max((right - left) * 0.03, 1.0)
    assert figures[0].axes[0].get_xlim() == pytest.approx(
        (left - margin, right + margin)
    )
    dirty_path = first[0].directory / "paper.json"
    paper = json.loads(dirty_path.read_text())
    paper["dirty"] = True
    dirty_path.write_text(json.dumps(paper))
    with pytest.raises(ValueError, match=first[0].directory.name):
        require_clean(tuple(item.directory for item in first))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper.experiments",
            "schedule_change",
            "figures",
            "--spec",
            str(case_spec),
            "--output",
            str(tmp_path / "dirty_figures"),
        ],
    )
    with pytest.raises(ValueError, match=first[0].directory.name):
        main()
    paper["dirty"] = False
    dirty_path.write_text(json.dumps(paper))
    policy = training[0].directory / "policy.zip"
    policy.write_bytes(policy.read_bytes() + b"changed")
    changed = schedule_change.run(case_spec)
    assert sum(not item.reused for item in changed) == 2
    assert all(
        item.directory.name.endswith("__02") for item in changed if not item.reused
    )
    assert all(item.directory.exists() for item in first)


def test_schedule_change_case_and_rank_order() -> None:
    defaults = schedule_change.load_schedule_spec(
        Path("paper/specs/schedule_change.toml")
    )
    assert defaults.candidate_sources == ("best", "final")
    assert defaults.delta_times_s == (0.0, 30.0, -30.0)
    assert defaults.change_distance_m == 8000.0
    assert [
        schedule_change.build_schedule_change_case(delta).token
        for delta in (0, 30, -30)
    ] == ["original", "plus_30p0s", "minus_30p0s"]
    case = schedule_change.CaseResult(
        case=schedule_change.build_schedule_change_case(0),
        feasible=True,
        safe=True,
        success=True,
        precise_arrival=True,
        punctual_arrival=True,
        safety_violation_count=0,
        min_safety_margin_mps=1.0,
        stop_error_m=0.1,
        abs_time_error_s=1.0,
        time_error_s=1.0,
        total_energy_j=200_000.0,
        total_energy_kj=200.0,
        total_reward=1.0,
        comfort_tav=0.2,
        run_dir="run",
        schedule_change_triggered=False,
    )
    failed = replace(
        case, feasible=False, punctual_arrival=False, total_energy_j=50_000.0
    )
    assert schedule_change.build_candidate_rank_key((case,)) > (
        schedule_change.build_candidate_rank_key((failed,))
    )
    worse_error = replace(case, stop_error_m=0.3, total_energy_j=50_000.0)
    assert schedule_change.build_candidate_rank_key((case, case)) > (
        schedule_change.build_candidate_rank_key((case, worse_error))
    )
    rank = schedule_change.build_candidate_rank_key((case,))
    tied = [
        schedule_change.CandidateEvaluation("final", "b", 11, "final", (case,), rank),
        schedule_change.CandidateEvaluation("best-b", "b", 11, "best", (case,), rank),
        schedule_change.CandidateEvaluation("best-a", "a", 11, "best", (case,), rank),
    ]
    assert [
        item.candidate_id for item in schedule_change.rank_candidate_evaluations(tied)
    ] == ["best-a", "best-b", "final"]


def test_schedule_change_trigger_position_is_absolute() -> None:
    """Regression for 7h: change_distance_m is an absolute track position.

    The legacy driver (``scripts/run_schedule_time_change.py``) treated
    ``--change-distance-m`` as the absolute track position at which the
    schedule change triggers, not a distance relative to the task's start
    position. This must not regress to ``start_position_m + change_distance_m``.
    """
    spec = schedule_change.load_schedule_spec(Path("paper/specs/schedule_change.toml"))
    source_spec = load_experiment_spec(spec.source_spec)
    task = load_tasks(source_spec.tasks)[source_spec.task]
    assert task.schedule_time_s == 465.0
    for delta in spec.delta_times_s:
        case_task = schedule_change.build_case_task(task, delta, spec.change_distance_m)
        if delta == 0.0:
            assert case_task.schedule_change is None
        else:
            assert case_task.schedule_change is not None
            assert case_task.schedule_change.trigger_position_m == 8000.0
            assert case_task.schedule_change.new_schedule_time_s == 465.0 + delta


def test_formal_figure_axes_match_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    method_spec = load_experiment_spec(Path("paper/specs/method_ablation.toml"))
    step_spec = load_experiment_spec(Path("paper/specs/step_distance.toml"))
    rollout_steps = 8 * 1024
    steps = [rollout * rollout_steps for rollout in range(12, 400, 12)]
    series = {"mean": [0.5] * len(steps), "std": [0.1] * len(steps)}
    method_summary = {
        "methods": [variant.id for variant in method_spec.variants],
        "method_labels": {
            variant.id: variant.label for variant in method_spec.variants
        },
        "training_curves": {
            variant.id: {
                "training_steps": steps,
                "speed_violation_rate": series,
                "arrival_ratio": series,
            }
            for variant in method_spec.variants
        },
        "evaluation_curves": {
            variant.id: {
                "training_steps": steps,
                **dict.fromkeys(
                    ("stop_error_m", "time_error_s", "total_energy_j", "comfort_tav"),
                    series,
                ),
            }
            for variant in method_spec.variants
        },
    }
    step_summary = {
        "variants": {
            variant.id: {"label": variant.label} for variant in step_spec.variants
        },
        "curves": {
            variant.id: {
                "training_steps": steps,
                "route_completion_ratio": series,
                "feasible": series,
            }
            for variant in step_spec.variants
        },
    }
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    try:
        method_figures(method_summary, method_spec, tmp_path)
        step_distance_figure(step_summary, step_spec, tmp_path)
    finally:
        monkeypatch.setattr(plt, "close", close)
        for figure in figures:
            close(figure)
    assert len(figures) == 3
    axis_end = 400 * rollout_steps
    inset_end = 396 * rollout_steps
    training, metrics, distance = figures
    assert [axis.get_xlim() for axis in training.axes] == [(0, axis_end)] * 2
    assert training.axes[0].child_axes[0].get_xlim() == (
        200 * rollout_steps,
        inset_end,
    )
    assert [axis.get_xlim() for axis in metrics.axes] == [(0, axis_end)] * 4
    assert [axis.child_axes[0].get_xlim() for axis in metrics.axes[:2]] == [
        (300 * rollout_steps, inset_end),
    ] * 2
    assert [axis.get_xlim() for axis in distance.axes] == [(0, axis_end)] * 2
    for axis in (*training.axes, *metrics.axes, *distance.axes):
        assert axis.xaxis.get_major_formatter()._powerlimits == (6, 6)
