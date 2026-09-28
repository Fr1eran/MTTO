from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mtto.domain.safeguard import Safeguard, ViolationKind, build_safeguard
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import (
    SafetyAudit,
    SpsEventKind,
    ViolationCategory,
)
from mtto.io.artifacts import RunKind, read_completed_run, task_from_json
from paper.figures import load_paper_scenario
from paper.plotting.profiles import DANGER_VIEW_LAYERS, render_safeguard
from paper.plotting.style import (
    apply_sci_curve_style,
    apply_sci_grid,
    save_sci_figure,
)

_OUTPUT_MODE_TEXT = "text"
_OUTPUT_MODE_PLOT = "plot"
_OUTPUT_MODE_JSON = "json"
_VALID_OUTPUT_MODES = frozenset(
    {
        _OUTPUT_MODE_TEXT,
        _OUTPUT_MODE_PLOT,
        _OUTPUT_MODE_JSON,
    }
)
_ANALYSIS_MODE_COMPARE = "compare"
_ANALYSIS_MODE_SINGLE = "single"
_VALID_ANALYSIS_MODES = (_ANALYSIS_MODE_SINGLE, _ANALYSIS_MODE_COMPARE)
_TRAJECTORY_KIND_DP = "dp"
_TRAJECTORY_KIND_RL = "rl"
_VALID_TRAJECTORY_KINDS = (_TRAJECTORY_KIND_DP, _TRAJECTORY_KIND_RL)
COMPARE_FIGURE_FILENAME = "dp_rl_sps_compliance.pdf"
SINGLE_FIGURE_FILENAMES = {
    _TRAJECTORY_KIND_DP: "dp_sps_compliance.pdf",
    _TRAJECTORY_KIND_RL: "rl_sps_compliance.pdf",
}


def _parse_output_mode(raw_mode: str) -> set[str]:
    normalized = raw_mode.strip().lower()
    if not normalized:
        raise ValueError("output mode cannot be empty")

    if normalized == "text+plot":
        return {_OUTPUT_MODE_TEXT, _OUTPUT_MODE_PLOT}
    if normalized == "all":
        return set(_VALID_OUTPUT_MODES)

    tokens = [token for token in re.split(r"[,+]", normalized) if token]
    if not tokens:
        raise ValueError("output mode cannot be empty")

    modes = set(tokens)
    unknown_modes = sorted(mode for mode in modes if mode not in _VALID_OUTPUT_MODES)
    if unknown_modes:
        raise ValueError(
            "Unknown output mode(s): "
            + f"{unknown_modes}. Choices: text, plot, json, text+plot"
        )

    return modes


def _deduplicate_legend(ax: Any, *, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    filtered_handles: list[Any] = []
    filtered_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels, strict=False):
        if not label or label.startswith("_"):
            continue
        if label in seen:
            continue
        seen.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)

    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc=loc)


def _audit_to_sps_dict(
    label: str, profile: SpeedProfile, audit: SafetyAudit
) -> dict[str, Any]:
    request_count = sum(e.kind == SpsEventKind.REQUEST_START for e in audit.events)
    complete_count = sum(e.kind == SpsEventKind.STEP_COMPLETE for e in audit.events)
    unfinished_count = sum(
        e.kind == SpsEventKind.REQUEST_UNFINISHED for e in audit.events
    )

    min_violation_count = sum(
        v.kind == ViolationKind.UNDER_LOWER_LIMIT for v in audit.violations
    )
    max_violation_count = sum(
        v.kind == ViolationKind.OVER_UPPER_LIMIT for v in audit.violations
    )
    pre_timeout_min = sum(
        v.kind == ViolationKind.UNDER_LOWER_LIMIT
        and v.category == ViolationCategory.PRE_TIMEOUT
        for v in audit.violations
    )
    pre_timeout_max = sum(
        v.kind == ViolationKind.OVER_UPPER_LIMIT
        and v.category == ViolationCategory.PRE_TIMEOUT
        for v in audit.violations
    )
    delay_related_min = sum(
        v.kind == ViolationKind.UNDER_LOWER_LIMIT
        and v.category == ViolationCategory.DELAY_RELATED
        for v in audit.violations
    )
    delay_related_max = sum(
        v.kind == ViolationKind.OVER_UPPER_LIMIT
        and v.category == ViolationCategory.DELAY_RELATED
        for v in audit.violations
    )

    triggered_pass = request_count > 0
    delay_pass = (delay_related_min + delay_related_max) == 0

    first_failure_reason = None
    if not triggered_pass:
        first_failure_reason = "no_step_request_triggered"
    elif not delay_pass:
        first_failure_reason = "delay_related_boundary_violation"

    events_list: list[dict[str, Any]] = []
    for e in audit.events:
        events_list.append(
            {
                "kind": e.kind.value,
                "time_s": float(e.time_s),
                "pos_m": float(e.position_m),
                "speed_mps": float(profile.speed_mps[e.node_index]),
                "current_sp": int(e.target_stopping_point),
                "request_target_sp": (
                    int(e.target_stopping_point)
                    if e.kind == SpsEventKind.STEP_COMPLETE
                    else int(e.target_stopping_point + 1)
                ),
                "boundary": None,
            }
        )
    for v in audit.violations:
        b_name = "min" if v.kind == ViolationKind.UNDER_LOWER_LIMIT else "max"
        k_name = (
            "MIN_VIOLATION"
            if v.kind == ViolationKind.UNDER_LOWER_LIMIT
            else "MAX_VIOLATION"
        )
        events_list.append(
            {
                "kind": k_name,
                "time_s": float(profile.time_s[v.node_index]),
                "pos_m": float(v.position_m),
                "speed_mps": float(v.speed_mps),
                "current_sp": int(audit.target_stopping_point[v.node_index]),
                "request_target_sp": None,
                "boundary": b_name,
            }
        )
        if v.category == ViolationCategory.DELAY_RELATED:
            events_list.append(
                {
                    "kind": "DELAY_RELATED_VIOLATION",
                    "time_s": float(profile.time_s[v.node_index]),
                    "pos_m": float(v.position_m),
                    "speed_mps": float(v.speed_mps),
                    "current_sp": int(audit.target_stopping_point[v.node_index]),
                    "request_target_sp": int(
                        audit.target_stopping_point[v.node_index] + 1
                    ),
                    "boundary": b_name,
                }
            )
    events_list.sort(key=lambda x: (x["time_s"], x["pos_m"]))

    return {
        "label": label,
        "triggered": {
            "pass": triggered_pass,
            "request_count": request_count,
        },
        "delay_related_boundary_violation": {
            "pass": delay_pass,
            "delay_related_min_violation_count": delay_related_min,
            "delay_related_max_violation_count": delay_related_max,
            # delay_window_total_s 与 delay_tolerance_s 沿用旧脚本的固定值 0.0
            "delay_window_total_s": 0.0,
            "delay_tolerance_s": 0.0,
        },
        "counters": {
            "complete_count": complete_count,
            "unfinished_count": unfinished_count,
            "min_violation_count": min_violation_count,
            "max_violation_count": max_violation_count,
            "pre_timeout_min_violation_count": pre_timeout_min,
            "pre_timeout_max_violation_count": pre_timeout_max,
        },
        "first_failure_reason": first_failure_reason,
        "events": events_list,
    }


def _print_result_summary(result: dict[str, Any]) -> None:
    trigger_status = "PASS" if result["triggered"]["pass"] else "FAIL"
    delay_pass = result["delay_related_boundary_violation"]["pass"]
    delay_status = "PASS" if delay_pass else "FAIL"

    print(f"[{result['label']}]")
    print(f"  triggered: {trigger_status}")
    print(f"  delay_related_boundary_violation: {delay_status}")
    print(f"  request_count: {result['triggered']['request_count']}")
    print(f"  complete_count: {result['counters']['complete_count']}")
    print(f"  unfinished_count: {result['counters']['unfinished_count']}")
    min_v = result["counters"]["min_violation_count"]
    max_v = result["counters"]["max_violation_count"]
    print(f"  boundary_violations(min/max): {min_v}/{max_v}")
    del_min = result["delay_related_boundary_violation"][
        "delay_related_min_violation_count"
    ]
    del_max = result["delay_related_boundary_violation"][
        "delay_related_max_violation_count"
    ]
    print(f"  delay_related_violations(min/max): {del_min}/{del_max}")
    del_win = result["delay_related_boundary_violation"]["delay_window_total_s"]
    print(f"  delay_window_total_s: {del_win:.6f}")
    del_tol = result["delay_related_boundary_violation"]["delay_tolerance_s"]
    print(f"  delay_tolerance_s: {del_tol:.6f}")
    reason = result["first_failure_reason"] or "none"
    print(f"  first_failure_reason: {reason}")


def _print_comparison_summary(
    *,
    dp_result: dict[str, Any],
    rl_result: dict[str, Any],
) -> None:
    dp_req = dp_result["triggered"]["request_count"]
    rl_req = rl_result["triggered"]["request_count"]
    dp_cmp = dp_result["counters"]["complete_count"]
    rl_cmp = rl_result["counters"]["complete_count"]
    dp_win = dp_result["delay_related_boundary_violation"]["delay_window_total_s"]
    rl_win = rl_result["delay_related_boundary_violation"]["delay_window_total_s"]
    print("\n[DP vs RL summary]")
    print(f"  request_count (dp/rl): {dp_req}/{rl_req}")
    print(f"  complete_count (dp/rl): {dp_cmp}/{rl_cmp}")
    print(f"  delay_window_total_s (dp/rl): {dp_win:.6f}/{rl_win:.6f}")
    dp_delay_viols = (
        dp_result["delay_related_boundary_violation"][
            "delay_related_min_violation_count"
        ]
        + dp_result["delay_related_boundary_violation"][
            "delay_related_max_violation_count"
        ]
    )
    rl_delay_viols = (
        rl_result["delay_related_boundary_violation"][
            "delay_related_min_violation_count"
        ]
        + rl_result["delay_related_boundary_violation"][
            "delay_related_max_violation_count"
        ]
    )
    print(f"  delay_related_violation_count (dp/rl): {dp_delay_viols}/{rl_delay_viols}")


def _plot_event_markers(
    *,
    ax: Any,
    result: dict[str, Any],
    color: str,
    trajectory_label: str,
    annotation_mode: str,
    max_text_annotations: int,
) -> None:
    events = result["events"]
    request_events = [e for e in events if e["kind"] == "REQUEST_START"]
    complete_events = [e for e in events if e["kind"] == "STEP_COMPLETE"]

    if request_events:
        ax.scatter(
            [e["pos_m"] for e in request_events],
            [e["speed_mps"] * 3.6 for e in request_events],
            marker="^",
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            s=40,
            alpha=0.95,
            label=f"{trajectory_label} request start",
            zorder=6,
        )

    if complete_events:
        ax.scatter(
            [e["pos_m"] for e in complete_events],
            [e["speed_mps"] * 3.6 for e in complete_events],
            marker="o",
            facecolors=color,
            edgecolors="black",
            linewidths=0.6,
            s=28,
            alpha=0.9,
            label=f"{trajectory_label} step complete",
            zorder=7,
        )

    if annotation_mode == "marker-only":
        return

    total_event_count = len(request_events) + len(complete_events)
    if annotation_mode == "auto" and total_event_count > max_text_annotations:
        return

    for idx, e in enumerate(request_events, start=1):
        ax.annotate(
            f"R{idx}",
            xy=(e["pos_m"], e["speed_mps"] * 3.6),
            xytext=(3.0, 4.0),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    for idx, e in enumerate(complete_events, start=1):
        ax.annotate(
            f"C{idx}",
            xy=(e["pos_m"], e["speed_mps"] * 3.6),
            xytext=(3.0, -9.0),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )


def _resolve_safeguard(safeguard: Safeguard | None, factor: float | None) -> Safeguard:
    if safeguard is not None:
        return safeguard
    scenario = load_paper_scenario()
    effective_factor = (
        scenario.safeguard.params.factor if factor is None else float(factor)
    )
    if effective_factor != scenario.safeguard.params.factor:
        new_params = replace(scenario.safeguard.params, factor=effective_factor)
        return build_safeguard(
            params=new_params,
            line=scenario.line,
            levi_curves=scenario.safeguard.levi_curves,
            brake_curves=scenario.safeguard.brake_curves,
            min_curves=scenario.safeguard.min_curves,
            max_curves=scenario.safeguard.max_curves,
        )
    return scenario.safeguard


def _plot_sps_main_figure(
    *,
    dp_profile: SpeedProfile,
    rl_profile: SpeedProfile,
    dp_result: dict[str, Any],
    rl_result: dict[str, Any],
    no_safeguard: bool,
    factor: float | None,
    annotation_mode: str,
    max_text_annotations: int,
    safeguard: Safeguard | None,
    output_dir: Path | None = None,
    no_show: bool = False,
) -> None:
    apply_sci_curve_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if not no_safeguard:
        resolved_safeguard = _resolve_safeguard(safeguard, factor)
        render_safeguard(resolved_safeguard, ax=ax, layers=DANGER_VIEW_LAYERS)

    _ = ax.plot(
        dp_profile.position_m,
        dp_profile.speed_mps * 3.6,
        color="tab:red",
        linewidth=1.5,
        alpha=0.9,
        label="DP trajectory",
        zorder=4,
    )
    _ = ax.plot(
        rl_profile.position_m,
        rl_profile.speed_mps * 3.6,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.9,
        label="RL trajectory",
        zorder=5,
    )

    _plot_event_markers(
        ax=ax,
        result=dp_result,
        color="tab:red",
        trajectory_label="DP",
        annotation_mode=annotation_mode,
        max_text_annotations=max_text_annotations,
    )
    _plot_event_markers(
        ax=ax,
        result=rl_result,
        color="tab:blue",
        trajectory_label="RL",
        annotation_mode=annotation_mode,
        max_text_annotations=max_text_annotations,
    )

    _ = ax.set_title("DP/RL SPS compliance (speed-position)")
    _ = ax.set_xlabel("Position (m)")
    _ = ax.set_ylabel("Speed (km/h)")
    apply_sci_grid(ax)
    _deduplicate_legend(ax)

    fig.tight_layout()

    if output_dir is not None:
        saved_path = save_sci_figure(fig, output_dir / COMPARE_FIGURE_FILENAME)
        print(f"Saved figure to: {saved_path}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def _plot_sps_single_figure(
    *,
    profile: SpeedProfile,
    result: dict[str, Any],
    trajectory_kind: str,
    no_safeguard: bool,
    factor: float | None,
    annotation_mode: str,
    max_text_annotations: int,
    safeguard: Safeguard | None,
    output_dir: Path | None = None,
    no_show: bool = False,
) -> None:
    apply_sci_curve_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if not no_safeguard:
        resolved_safeguard = _resolve_safeguard(safeguard, factor)
        render_safeguard(resolved_safeguard, ax=ax, layers=DANGER_VIEW_LAYERS)

    if trajectory_kind == _TRAJECTORY_KIND_DP:
        curve_color = "tab:red"
        curve_label = "DP trajectory"
        marker_label = "DP"
    else:
        curve_color = "tab:blue"
        curve_label = "RL trajectory"
        marker_label = "RL"

    _ = ax.plot(
        profile.position_m,
        profile.speed_mps * 3.6,
        color=curve_color,
        linewidth=1.5,
        alpha=0.9,
        label=curve_label,
        zorder=5,
    )
    _plot_event_markers(
        ax=ax,
        result=result,
        color=curve_color,
        trajectory_label=marker_label,
        annotation_mode=annotation_mode,
        max_text_annotations=max_text_annotations,
    )

    _ = ax.set_title(f"{marker_label} SPS compliance (speed-position)")
    _ = ax.set_xlabel("Position (m)")
    _ = ax.set_ylabel("Speed (km/h)")
    apply_sci_grid(ax)
    _deduplicate_legend(ax)

    fig.tight_layout()

    if output_dir is not None:
        saved_path = save_sci_figure(
            fig, output_dir / SINGLE_FIGURE_FILENAMES[trajectory_kind]
        )
        print(f"Saved figure to: {saved_path}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SPS compliance for selected DP/RL trajectories in "
            "single or compare mode. "
            "Default output mode is text+plot."
        )
    )
    _ = parser.add_argument(
        "--dp-run",
        type=Path,
        default=None,
        help="DP solve run directory.",
    )
    _ = parser.add_argument(
        "--rl-run",
        type=Path,
        default=None,
        help="RL training or evaluation run directory.",
    )
    _ = parser.add_argument(
        "--rl-best",
        action="store_true",
        default=False,
        help="Use best/ payload for RL training run.",
    )
    _ = parser.add_argument(
        "--analysis-mode",
        choices=_VALID_ANALYSIS_MODES,
        default=_ANALYSIS_MODE_COMPARE,
        help="Analysis mode: single (DP or RL only) or compare (DP and RL).",
    )
    _ = parser.add_argument(
        "--trajectory-kind",
        choices=_VALID_TRAJECTORY_KINDS,
        default=None,
        help="Trajectory kind for single mode: dp or rl.",
    )
    _ = parser.add_argument(
        "--output-mode",
        default="text+plot",
        help="Output mode: text, plot, json, text+plot, or comma/plus combinations.",
    )
    _ = parser.add_argument(
        "--json-output-path",
        type=Path,
        default=None,
        help="When json output is enabled, optionally save payload to this path.",
    )
    _ = parser.add_argument(
        "--event-annotation",
        choices=("auto", "text", "marker-only"),
        default="auto",
        help="Event annotation mode on the main figure.",
    )
    _ = parser.add_argument(
        "--max-text-annotations",
        type=int,
        default=12,
        help="Max event labels for auto annotation mode.",
    )
    _ = parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="Do not render safeguard background on the main speed-position figure.",
    )
    _ = parser.add_argument(
        "--factor",
        type=float,
        default=None,
        help="Safeguard factor used for rendering and replay boundaries.",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the fixed SPS compliance PDF.",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the interactive plot window.",
    )
    return parser


def _validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.analysis_mode == _ANALYSIS_MODE_SINGLE:
        if args.trajectory_kind is None:
            parser.error("--trajectory-kind is required when --analysis-mode=single")
        if args.trajectory_kind == _TRAJECTORY_KIND_DP:
            if args.dp_run is None:
                parser.error("--dp-run is required when --trajectory-kind=dp")
            if args.rl_run is not None:
                parser.error("--rl-run is not allowed when --trajectory-kind=dp")
        elif args.trajectory_kind == _TRAJECTORY_KIND_RL:
            if args.rl_run is None:
                parser.error("--rl-run is required when --trajectory-kind=rl")
            if args.dp_run is not None:
                parser.error("--dp-run is not allowed when --trajectory-kind=rl")
    elif args.analysis_mode == _ANALYSIS_MODE_COMPARE:
        if args.trajectory_kind is not None:
            parser.error("--trajectory-kind is only valid when --analysis-mode=single")
        if args.dp_run is None:
            parser.error("--dp-run is required when --analysis-mode=compare")
        if args.rl_run is None:
            parser.error("--rl-run is required when --analysis-mode=compare")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)

    try:
        output_modes = _parse_output_mode(args.output_mode)
    except ValueError as exc:
        parser.error(str(exc))

    scenario = load_paper_scenario()

    if args.analysis_mode == _ANALYSIS_MODE_COMPARE:
        try:
            dp_run = read_completed_run(args.dp_run)
            rl_run = read_completed_run(args.rl_run)
        except Exception as exc:
            parser.error(f"Failed to read run: {exc}")

        if dp_run.record.kind != RunKind.DP_SOLVE:
            parser.error(
                f"Expected DP solve run for --dp-run, got {dp_run.record.kind.value}"
            )
        if rl_run.record.kind not in (RunKind.RL_TRAIN, RunKind.EVALUATION):
            parser.error(
                f"Expected RL run for --rl-run, got {rl_run.record.kind.value}"
            )
        if args.rl_best:
            if rl_run.record.kind != RunKind.RL_TRAIN:
                parser.error("--rl-best is only valid for RL training runs")
            if rl_run.payload.best is None:
                parser.error(
                    f"--rl-best specified but {args.rl_run} has no best/ artifacts"
                )

        for run_path, run_obj in [(args.dp_run, dp_run), (args.rl_run, rl_run)]:
            if run_obj.record.scenario_hash != scenario.scenario_hash:
                parser.error(
                    f"Scenario hash mismatch for run at {run_path}: "
                    f"expected {scenario.scenario_hash}, "
                    f"got {run_obj.record.scenario_hash}"
                )

        dp_task = task_from_json(dp_run.record.task)
        rl_task = task_from_json(rl_run.record.task)
        if (
            dp_task.schedule_time_s != rl_task.schedule_time_s
            or dp_task.target_position_m != rl_task.target_position_m
            or dp_task.start_position_m != rl_task.start_position_m
        ):
            parser.error(
                "Task mismatch between DP and RL runs; select runs with the same task."
            )

        dp_profile = dp_run.payload.profile
        dp_audit = dp_run.payload.quality.audit

        rl_payload = rl_run.payload.best if args.rl_best else rl_run.payload
        rl_profile = rl_payload.profile
        rl_audit = rl_payload.quality.audit

        dp_result = _audit_to_sps_dict("DP", dp_profile, dp_audit)
        rl_result = _audit_to_sps_dict("RL", rl_profile, rl_audit)

        step_delay_s = float(scenario.safeguard.params.step_delay_s)
        schedule_time_s = float(
            dp_task.schedule_time_s if dp_task.schedule_time_s is not None else 0.0
        )

        if _OUTPUT_MODE_TEXT in output_modes:
            print(f"Using DP run: {args.dp_run}")
            print(f"Using RL run: {args.rl_run}")
            print(f"Resolved schedule_time_s: {schedule_time_s:.6f}")
            print(f"Replay step_delay_s (T_s): {step_delay_s:.6f}")
            print()
            _print_result_summary(dp_result)
            print()
            _print_result_summary(rl_result)
            _print_comparison_summary(dp_result=dp_result, rl_result=rl_result)

        if _OUTPUT_MODE_JSON in output_modes:
            payload = {
                "schedule_time_s": schedule_time_s,
                "step_delay_s": step_delay_s,
                "artifacts": {
                    "dp": str(args.dp_run),
                    "rl": str(args.rl_run),
                },
                "results": {
                    "dp": dp_result,
                    "rl": rl_result,
                },
            }
            payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
            if args.json_output_path:
                output_path = Path(args.json_output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _ = output_path.write_text(payload_text, encoding="utf-8")
                print(f"JSON report saved to: {output_path}")
            else:
                print(payload_text)

        if _OUTPUT_MODE_PLOT in output_modes:
            safeguard_utility = _resolve_safeguard(None, args.factor)
            _plot_sps_main_figure(
                dp_profile=dp_profile,
                rl_profile=rl_profile,
                dp_result=dp_result,
                rl_result=rl_result,
                no_safeguard=args.no_safeguard,
                factor=args.factor,
                annotation_mode=args.event_annotation,
                max_text_annotations=args.max_text_annotations,
                safeguard=safeguard_utility,
                output_dir=args.output_dir,
                no_show=args.no_show,
            )
        return

    # Single mode
    trajectory_kind = str(args.trajectory_kind)
    run_dir = args.dp_run if trajectory_kind == _TRAJECTORY_KIND_DP else args.rl_run
    try:
        run_obj = read_completed_run(run_dir)
    except Exception as exc:
        parser.error(f"Failed to read run from {run_dir}: {exc}")

    if trajectory_kind == _TRAJECTORY_KIND_DP:
        if run_obj.record.kind != RunKind.DP_SOLVE:
            parser.error(f"Expected DP solve run, got {run_obj.record.kind.value}")
        target_payload = run_obj.payload
        label = "DP"
    else:
        if run_obj.record.kind not in (RunKind.RL_TRAIN, RunKind.EVALUATION):
            parser.error(f"Expected RL run, got {run_obj.record.kind.value}")
        if args.rl_best:
            if run_obj.record.kind != RunKind.RL_TRAIN:
                parser.error("--rl-best is only valid for RL training runs")
            if run_obj.payload.best is None:
                parser.error(
                    f"--rl-best specified but {run_dir} has no best/ artifacts"
                )
            target_payload = run_obj.payload.best
        else:
            target_payload = run_obj.payload
        label = "RL"

    if run_obj.record.scenario_hash != scenario.scenario_hash:
        parser.error(
            f"Scenario hash mismatch for run at {run_dir}: "
            f"expected {scenario.scenario_hash}, got {run_obj.record.scenario_hash}"
        )

    task = task_from_json(run_obj.record.task)
    profile = target_payload.profile
    audit = target_payload.quality.audit
    result_dict = _audit_to_sps_dict(label, profile, audit)

    step_delay_s = float(scenario.safeguard.params.step_delay_s)
    schedule_time_s = float(
        task.schedule_time_s if task.schedule_time_s is not None else 0.0
    )

    if _OUTPUT_MODE_TEXT in output_modes:
        print(f"Analysis mode: {args.analysis_mode}")
        print(f"Trajectory kind: {trajectory_kind}")
        print(f"Using {label} run: {run_dir}")
        print(f"Resolved schedule_time_s: {schedule_time_s:.6f}")
        print(f"Replay step_delay_s (T_s): {step_delay_s:.6f}")
        print()
        _print_result_summary(result_dict)

    if _OUTPUT_MODE_JSON in output_modes:
        payload = {
            "analysis_mode": args.analysis_mode,
            "trajectory_kind": trajectory_kind,
            "schedule_time_s": schedule_time_s,
            "step_delay_s": step_delay_s,
            "artifact": str(run_dir),
            "result": result_dict,
        }
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json_output_path:
            output_path = Path(args.json_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _ = output_path.write_text(payload_text, encoding="utf-8")
            print(f"JSON report saved to: {output_path}")
        else:
            print(payload_text)

    if _OUTPUT_MODE_PLOT in output_modes:
        safeguard_utility = _resolve_safeguard(None, args.factor)
        _plot_sps_single_figure(
            profile=profile,
            result=result_dict,
            trajectory_kind=trajectory_kind,
            no_safeguard=args.no_safeguard,
            factor=args.factor,
            annotation_mode=args.event_annotation,
            max_text_annotations=args.max_text_annotations,
            safeguard=safeguard_utility,
            output_dir=args.output_dir,
            no_show=args.no_show,
        )


if __name__ == "__main__":
    main()
