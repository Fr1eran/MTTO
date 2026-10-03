import warnings
from dataclasses import replace

import numpy as np
import pytest

from mtto.domain.energy import profile_energy, segment_energy
from mtto.domain.kinematics import segment_acceleration
from mtto.domain.safeguard import ViolationKind, dynamic_limits
from mtto.domain.scenario import ScheduleChange
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import (
    AuditViolation,
    QualityMetrics,
    QualityReport,
    SafetyAudit,
    SpsEventKind,
    ViolationCategory,
    assess,
    audit_dynamic_limits,
    best_update_reason,
    selection_key,
)
from mtto.rl.evaluate import run_policy
from mtto.rl.state import TerminationReason
from paper.figures import load_paper_scenario, load_paper_task
from tests.golden.cases import RL_CASES
from tests.golden.drive import build_env, load_actions


@pytest.fixture(scope="module")
def scenario():
    return load_paper_scenario()


@pytest.fixture(scope="module")
def task():
    return load_paper_task(schedule_time_s=465.0)


@pytest.mark.parametrize(
    ("scheduled", "offset", "duration", "unsafe", "expected"),
    [
        (True, 1.0, 465.0, False, (True, True, True, True)),
        (True, 1.0, 475.0, False, (True, True, False, False)),
        (True, 2.0, 465.0, False, (True, False, False, False)),
        (True, -40.0, 465.0, False, (False, False, False, False)),
        (True, 0.0, 465.0, True, (True, True, True, False)),
        (False, 1.0, 475.0, False, (True, True, None, True)),
        (False, 1.0, 475.0, True, (True, True, None, False)),
    ],
)
def test_assess_judgements(
    monkeypatch, scenario, task, scheduled, offset, duration, unsafe, expected
):
    audit = SafetyAudit(
        np.array([-1, -1]),
        np.array([False, False]),
        np.array([0.0, 0.0]),
        np.array([20.0, 20.0]),
        (
            AuditViolation(
                1,
                1.0,
                0.0,
                ViolationKind.UNDER_LOWER_LIMIT,
                1.0,
                ViolationCategory.PRE_TIMEOUT,
            ),
        )
        if unsafe
        else (),
        (),
        -1.0 if unsafe else 1.0,
    )
    monkeypatch.setattr(
        "mtto.evaluation.quality.audit_dynamic_limits", lambda *_: audit
    )
    selected_task = replace(
        task, schedule_time_s=465.0 if scheduled else None, max_stop_error_m=1.0
    )
    profile = SpeedProfile.from_arrays(
        [task.start_position_m, task.target_position_m + offset],
        [0.0, 0.0],
        [0.0, duration],
        [0.0, 0.0],
        [0.0, 0.0],
    )
    report = assess(profile, scenario, selected_task)
    assert (
        report.completed,
        report.precise_stop,
        report.punctual,
        report.feasible,
    ) == expected
    assert report.metrics.arrival_time_error_s == (
        duration - 465.0 if scheduled else None
    )


def test_schedule_change_arrival_time(monkeypatch, scenario, task):
    audit = SafetyAudit(
        np.array([-1, -1, -1]),
        np.array([False] * 3),
        np.zeros(3),
        np.ones(3),
        (),
        (),
        0.0,
    )
    monkeypatch.setattr(
        "mtto.evaluation.quality.audit_dynamic_limits", lambda *_: audit
    )
    changed_task = replace(
        task,
        schedule_change=ScheduleChange(
            trigger_position_m=200.0, new_schedule_time_s=500.0
        ),
    )
    profile = SpeedProfile.from_arrays(
        [task.start_position_m, 200.0, task.target_position_m],
        [0.0, 5.0, 0.0],
        [0.0, 10.0, 490.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    assert assess(profile, scenario, changed_task).metrics.arrival_time_error_s == -10.0


@pytest.mark.parametrize(
    ("position", "speed", "time", "expected"),
    [
        ([135.0], [0.0], [0.0], (0.0, 0.0, 0.0)),
        # ER% counts segments whose jerk |da| / dt exceeds 0.75 m/s^3.
        # |da| = 1 over 2 s is a 0.5 m/s^3 jerk.
        ([135.0, 135.0], [0.0, 2.0], [0.0, 2.0], (1.0, 1.0, 0.0)),
        # Jerks 0.5 and 1.0 m/s^3.
        ([135.0, 135.0, 135.0], [0.0, 2.0, 2.0], [0.0, 2.0, 3.0], (2.0, 1.0, 50.0)),
        # The same |da| = 0.5, 1.5 over 1 s and 0.5 s: jerks 0.5 and 3.0 m/s^3.
        (
            [135.0, 135.25, 136.25],
            [0.0, 0.5, 1.5],
            [0.0, 1.0, 1.5],
            (2.0, 1.25**0.5, 50.0),
        ),
        # A zero-duration segment is in neither the numerator nor the
        # denominator of ER%; TAV and RMS still count it.
        (
            [135.0, 135.0, 137.0, 139.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 3.0],
            (2.0, (2.0 / 3.0) ** 0.5, 50.0),
        ),
    ],
)
def test_comfort_includes_zero_length_and_single_node(
    scenario, task, position, speed, time, expected
):
    nodes = len(position)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = SpeedProfile.from_arrays(
            position, speed, time, np.zeros(nodes), np.zeros(nodes)
        )
        metrics = assess(profile, scenario, task).metrics
    np.testing.assert_allclose(
        (
            metrics.comfort_tav_mps2,
            metrics.comfort_rms_mps2,
            metrics.comfort_exceedance_pct,
        ),
        expected,
    )


def test_domain_array_functions_include_zero_length_energy(scenario):
    position = np.array([135.0, 135.0, 135.0])
    speed = np.array([0.0, 2.0, 2.0])
    time = np.array([0.0, 2.0, 2.0])
    acceleration = segment_acceleration(speed, time)
    np.testing.assert_array_equal(acceleration, [1.0, 0.0])
    propulsion, levitation = profile_energy(
        scenario.energy,
        scenario.vehicle,
        scenario.line,
        position,
        speed,
        time,
        acceleration,
    )
    first = segment_energy(
        scenario.energy,
        scenario.vehicle,
        scenario.line,
        begin_pos=135.0,
        begin_speed=0.0,
        acc=1.0,
        distance=0.0,
        direction=1,
        operation_time=2.0,
    )
    np.testing.assert_allclose([propulsion[1], levitation[1]], first)
    np.testing.assert_array_equal(
        [propulsion[2], levitation[2]], [propulsion[1], levitation[1]]
    )


def test_audit_records_every_violation_and_events(scenario):
    profile = SpeedProfile.from_arrays(
        [135.0, 165.0, 195.0, 225.0],
        [17.5, 7.0, 20.0, 20.0],
        [0.0, 1.0, 1.1, 1.2],
        [0.0] * 4,
        [0.0] * 4,
    )
    audit = audit_dynamic_limits(profile, scenario)
    assert audit.target_stopping_point.tolist() == [-1, -1, -1, -1]
    assert audit.request_pending.tolist() == [False, True, True, True]
    assert [v.node_index for v in audit.violations] == [0, 2, 3]
    assert [v.category for v in audit.violations] == [
        ViolationCategory.PRE_TIMEOUT,
        ViolationCategory.DELAY_RELATED,
        ViolationCategory.DELAY_RELATED,
    ]
    assert [event.kind for event in audit.events] == [
        SpsEventKind.REQUEST_START,
        SpsEventKind.REQUEST_UNFINISHED,
    ]
    for values in (
        audit.target_stopping_point,
        audit.request_pending,
        audit.lower_limit_mps,
        audit.upper_limit_mps,
    ):
        with pytest.raises(ValueError):
            values[0] = 0


def test_audit_initial_node_and_strict_upper_bound(scenario):
    _, upper = dynamic_limits(scenario.safeguard, 165.0, -1)
    profile = SpeedProfile.from_arrays([165.0], [upper], [0.0], [0.0], [0.0])
    audit = audit_dynamic_limits(profile, scenario)
    assert audit.target_stopping_point.tolist() == [-1]
    assert audit.request_pending.tolist() == [False]
    assert audit.violations == ()
    assert audit.min_margin_mps == 0.0


@pytest.mark.parametrize("case", RL_CASES, ids=lambda case: case.name)
def test_audit_matches_rl_transition_nodes(case):
    env = build_env("basic_safety_punctuality")
    actions = load_actions(case.name)
    state = env.initial_state()
    states = [state]
    reason = None
    for action in actions:
        acceleration = env.observation_builder.denormalize_action(float(action))
        result = env.transition(state, acceleration)
        states.append(result.step_end_state)
        state = result.next_state
        reason = result.termination_reason
        if reason is not None:
            break
    scripted = iter(actions)
    run = run_policy(lambda _: next(scripted), env)
    audit = audit_dynamic_limits(run.profile, env.scenario)
    assert len(states) == run.steps + 1 == len(actions) + 1
    np.testing.assert_array_equal(
        audit.target_stopping_point, [s.sps.target_stopping_point_index for s in states]
    )
    np.testing.assert_array_equal(
        audit.request_pending, [s.sps.request_pending for s in states]
    )
    np.testing.assert_allclose(
        audit.lower_limit_mps,
        [s.lower_limit_mps for s in states],
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        audit.upper_limit_mps,
        [s.upper_limit_mps for s in states],
        rtol=1e-9,
        atol=1e-12,
    )
    for event in audit.events:
        assert event.time_s == states[event.node_index].t_s
        assert (
            event.target_stopping_point
            == states[event.node_index].sps.target_stopping_point_index
        )
    assert run.termination_reason == reason
    if reason in (
        TerminationReason.UNDER_LOWER_LIMIT,
        TerminationReason.OVER_UPPER_LIMIT,
    ):
        assert len(audit.violations) == 1
        assert audit.violations[0].node_index == run.steps
        assert audit.violations[0].kind.value == reason.name
    elif reason is TerminationReason.OVERRAN:
        # 停车判定优先于越界判定，末节点的越界不改变终止原因
        assert all(v.node_index == run.steps for v in audit.violations)
    else:
        assert audit.violations == ()


def test_unscheduled_selection_key_and_reason():
    audit = SafetyAudit(
        np.array([0]), np.array([False]), np.array([0.0]), np.array([1.0]), (), (), 0.0
    )

    def report(energy, feasible, completed, safe, precise, stop_error):
        metrics = QualityMetrics(energy, 0.0, 1.0, stop_error, 0.0, 0.0, 0.0, None)
        return QualityReport(metrics, audit, completed, precise, safe, None, feasible)

    first = report(2.0, False, False, False, False, 10.0)
    second = report(3.0, False, True, False, True, 1.0)
    third = report(1.0, True, True, True, True, 0.0)
    fourth = report(0.5, True, True, True, True, 0.0)
    assert len(selection_key(first)) == 5
    assert (
        selection_key(first)
        < selection_key(second)
        < selection_key(third)
        < selection_key(fourth)
    )
    assert [
        best_update_reason(candidate, previous)
        for candidate, previous in (
            (first, None),
            (first, first),
            (second, first),
            (third, second),
            (fourth, third),
        )
    ] == [
        "first_evaluation",
        None,
        "safe_success_reached",
        "strict_feasibility_reached",
        "lower_energy_among_feasible",
    ]
