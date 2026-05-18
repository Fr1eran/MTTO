from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    resolve_dp_curve_artifact,
)
from model.ocs import SPS, SafeGuardUtility
from rl.experiment_utils import (
    DEFAULT_SCHEDULE_TIME_S,
    RL_DEFAULT_SEARCH_DIR,
    RL_TRAJECTORY_SOURCE_CHOICES,
    apply_rl_curve_plot_style,
    load_rl_curve_artifact,
    resolve_rl_curve_artifact,
)
from utils.scenario import build_safeguard_utility, build_scenario
from utils.trajectory import OptimizedCurveArtifact, recover_time_axis_from_trajectory

_OUTPUT_MODE_TEXT = "text"
_OUTPUT_MODE_PLOT = "plot"
_OUTPUT_MODE_JSON = "json"
_VALID_OUTPUT_MODES = frozenset({
    _OUTPUT_MODE_TEXT,
    _OUTPUT_MODE_PLOT,
    _OUTPUT_MODE_JSON,
})

_EVENT_REQUEST_START = "REQUEST_START"
_EVENT_STEP_COMPLETE = "STEP_COMPLETE"
_EVENT_MIN_VIOLATION = "MIN_VIOLATION"
_EVENT_MAX_VIOLATION = "MAX_VIOLATION"
_EVENT_DELAY_RELATED_VIOLATION = "DELAY_RELATED_VIOLATION"
_EVENT_REQUEST_UNFINISHED = "REQUEST_UNFINISHED"


@dataclass(frozen=True)
class SPSEventRecord:
    kind: str
    time_s: float
    pos_m: float
    speed_mps: float
    current_sp: int
    request_target_sp: int | None = None
    boundary: str | None = None


@dataclass(frozen=True)
class SPSComplianceResult:
    label: str
    triggered_pass: bool
    delay_related_boundary_violation_pass: bool
    request_count: int
    complete_count: int
    unfinished_count: int
    min_violation_count: int
    max_violation_count: int
    pre_timeout_min_violation_count: int
    pre_timeout_max_violation_count: int
    delay_related_min_violation_count: int
    delay_related_max_violation_count: int
    delay_window_total_s: float
    delay_tolerance_s: float
    first_failure_reason: str | None
    events: list[SPSEventRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "triggered": {
                "pass": self.triggered_pass,
                "request_count": self.request_count,
            },
            "delay_related_boundary_violation": {
                "pass": self.delay_related_boundary_violation_pass,
                "delay_related_min_violation_count": self.delay_related_min_violation_count,  # noqa: E501
                "delay_related_max_violation_count": self.delay_related_max_violation_count,  # noqa: E501
                "delay_window_total_s": self.delay_window_total_s,
                "delay_tolerance_s": self.delay_tolerance_s,
            },
            "counters": {
                "complete_count": self.complete_count,
                "unfinished_count": self.unfinished_count,
                "min_violation_count": self.min_violation_count,
                "max_violation_count": self.max_violation_count,
                "pre_timeout_min_violation_count": self.pre_timeout_min_violation_count,
                "pre_timeout_max_violation_count": self.pre_timeout_max_violation_count,
            },
            "first_failure_reason": self.first_failure_reason,
            "events": [asdict(event) for event in self.events],
        }


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _resolve_target_schedule_time(
    *,
    dp_metrics: dict[str, object],
    rl_metrics: dict[str, object],
    schedule_time_s_override: float | None,
) -> float:
    if schedule_time_s_override is not None and schedule_time_s_override > 0.0:
        return float(schedule_time_s_override)

    rl_target_time_s = _metric_as_float(rl_metrics.get("target_time_s"))
    if rl_target_time_s is not None and rl_target_time_s > 0.0:
        return rl_target_time_s

    dp_target_time_s = _metric_as_float(dp_metrics.get("target_time_s"))
    if dp_target_time_s is not None and dp_target_time_s > 0.0:
        return dp_target_time_s

    return DEFAULT_SCHEDULE_TIME_S


def _resolve_curve_artifacts(
    *,
    dp_curve_dir: str,
    rl_curve_dir: str,
    trajectory_source: str,
) -> tuple[OptimizedCurveArtifact, OptimizedCurveArtifact]:
    dp_artifact = resolve_dp_curve_artifact(curve_dir=dp_curve_dir)
    rl_artifact = resolve_rl_curve_artifact(
        curve_dir=rl_curve_dir,
        trajectory_source=trajectory_source,
    )
    return dp_artifact, rl_artifact


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
            f"{unknown_modes}. Choices: text, plot, json, text+plot"
        )

    return modes


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SPS compliance for selected DP and RL trajectories. "
            "Default output mode is text+plot."
        )
    )
    parser.add_argument(
        "--dp-curve-dir",
        default=DP_DEFAULT_SEARCH_DIR,
        help="Directory used to recursively search DP trajectory artifacts.",
    )
    parser.add_argument(
        "--rl-curve-dir",
        default=RL_DEFAULT_SEARCH_DIR,
        help="Directory used to recursively search RL trajectory artifacts.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=RL_TRAJECTORY_SOURCE_CHOICES,
        default="best",
        help="RL trajectory source: best, best_steps, best_episodes, final.",
    )
    parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=None,
        help="Optional schedule time override for scenario construction.",
    )
    parser.add_argument(
        "--step-delay-s",
        type=float,
        default=2.0,
        help="SPS average delay T_s used by replay, in seconds.",
    )
    parser.add_argument(
        "--boundary-eps",
        type=float,
        default=1e-6,
        help="Numerical tolerance for min/max boundary violation checks.",
    )
    parser.add_argument(
        "--output-mode",
        default="text+plot",
        help="Output mode: text, plot, json, text+plot, or comma/plus combinations.",
    )
    parser.add_argument(
        "--json-output-path",
        default=None,
        help="When json output is enabled, optionally save payload to this path.",
    )
    parser.add_argument(
        "--event-annotation",
        choices=("auto", "text", "marker-only"),
        default="auto",
        help="Event annotation mode on the main figure.",
    )
    parser.add_argument(
        "--max-text-annotations",
        type=int,
        default=12,
        help="Max event labels for auto annotation mode.",
    )
    parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="Do not render safeguard background on the main speed-position figure.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="Safeguard factor used for rendering and replay boundaries.",
    )
    return parser


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


def _compute_adaptive_delay_tolerance(time_arr: np.ndarray) -> float:
    if time_arr.size < 2:
        return 0.05

    dt_arr = np.diff(time_arr)
    positive_dt = dt_arr[dt_arr > 1e-9]
    if positive_dt.size == 0:
        return 0.05

    return max(0.05, 0.5 * float(np.median(positive_dt)))


def replay_sps_compliance(
    *,
    label: str,
    pos_arr: np.ndarray,
    speed_arr: np.ndarray,
    sgu: SafeGuardUtility,
    asa_ap_list: list[float],
    asa_dp_list: list[float],
    step_delay_s: float,
    boundary_eps: float = 1e-6,
    time_arr: np.ndarray | None = None,
) -> SPSComplianceResult:
    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)

    if pos.ndim != 1 or speed.ndim != 1:
        raise ValueError("pos_arr and speed_arr must be 1-D arrays")
    if pos.size != speed.size:
        raise ValueError("pos_arr and speed_arr must have equal length")
    if pos.size < 2:
        raise ValueError("trajectory must contain at least two points")
    if step_delay_s <= 0.0:
        raise ValueError("step_delay_s must be positive")
    if boundary_eps < 0.0:
        raise ValueError("boundary_eps must be >= 0")

    if time_arr is None:
        time_axis = recover_time_axis_from_trajectory(pos, speed)
    else:
        time_axis = np.asarray(time_arr, dtype=np.float64)

    if time_axis.ndim != 1:
        raise ValueError("time_arr must be a 1-D array")
    if time_axis.size != pos.size:
        raise ValueError("time_arr must have equal length with pos_arr")

    delay_tolerance_s = _compute_adaptive_delay_tolerance(time_axis)

    sps = SPS(
        sgu=sgu,
        ASA_ap_list=asa_ap_list,
        ASA_dp_list=asa_dp_list,
        T_s=step_delay_s,
    )

    current_sp = -1
    request_open = False
    request_target_sp: int | None = None
    request_timeout_deadline = float("inf")
    timeout_active = False
    timeout_start_time: float | None = None

    request_count = 0
    complete_count = 0
    unfinished_count = 0

    min_violation_count = 0
    max_violation_count = 0
    pre_timeout_min_violation_count = 0
    pre_timeout_max_violation_count = 0
    delay_related_min_violation_count = 0
    delay_related_max_violation_count = 0
    delay_window_total_s = 0.0

    events: list[SPSEventRecord] = []

    for idx in range(pos.size):
        t = float(time_axis[idx])
        x = float(pos[idx])
        v = float(speed[idx])

        if request_open and (not timeout_active) and t > request_timeout_deadline:
            timeout_active = True
            timeout_start_time = t

        current_min_speed, current_max_speed = sgu.get_min_and_max_speed(
            current_pos=x,
            current_sp=current_sp,
        )
        is_min_violation = v < (current_min_speed - boundary_eps)
        is_max_violation = v > (current_max_speed + boundary_eps)

        if is_min_violation:
            min_violation_count += 1
            events.append(
                SPSEventRecord(
                    kind=_EVENT_MIN_VIOLATION,
                    time_s=t,
                    pos_m=x,
                    speed_mps=v,
                    current_sp=current_sp,
                    boundary="min",
                )
            )
            if timeout_active and request_open:
                delay_related_min_violation_count += 1
                events.append(
                    SPSEventRecord(
                        kind=_EVENT_DELAY_RELATED_VIOLATION,
                        time_s=t,
                        pos_m=x,
                        speed_mps=v,
                        current_sp=current_sp,
                        request_target_sp=request_target_sp,
                        boundary="min",
                    )
                )
            else:
                pre_timeout_min_violation_count += 1

        if is_max_violation:
            max_violation_count += 1
            events.append(
                SPSEventRecord(
                    kind=_EVENT_MAX_VIOLATION,
                    time_s=t,
                    pos_m=x,
                    speed_mps=v,
                    current_sp=current_sp,
                    boundary="max",
                )
            )
            if timeout_active and request_open:
                delay_related_max_violation_count += 1
                events.append(
                    SPSEventRecord(
                        kind=_EVENT_DELAY_RELATED_VIOLATION,
                        time_s=t,
                        pos_m=x,
                        speed_mps=v,
                        current_sp=current_sp,
                        request_target_sp=request_target_sp,
                        boundary="max",
                    )
                )
            else:
                pre_timeout_max_violation_count += 1

        prev_done = bool(sps.IsPrevSPSReqDone)
        prev_sp = current_sp
        next_sp = int(
            sps.step_to_next_stopping_point(
                current_pos=x,
                current_speed=v,
                current_time=t,
                current_sp=current_sp,
            )
        )
        now_done = bool(sps.IsPrevSPSReqDone)

        if prev_done and (not now_done) and next_sp == prev_sp:
            request_count += 1
            request_open = True
            request_target_sp = prev_sp + 1
            request_timeout_deadline = (
                float(sps.SPSReqTimeStamp) + step_delay_s + delay_tolerance_s
            )
            timeout_active = False
            timeout_start_time = None
            events.append(
                SPSEventRecord(
                    kind=_EVENT_REQUEST_START,
                    time_s=t,
                    pos_m=x,
                    speed_mps=v,
                    current_sp=prev_sp,
                    request_target_sp=request_target_sp,
                )
            )
        elif (not prev_done) and now_done and next_sp == prev_sp + 1:
            complete_count += 1
            events.append(
                SPSEventRecord(
                    kind=_EVENT_STEP_COMPLETE,
                    time_s=t,
                    pos_m=x,
                    speed_mps=v,
                    current_sp=next_sp,
                    request_target_sp=next_sp,
                )
            )
            if timeout_active and timeout_start_time is not None:
                delay_window_total_s += max(0.0, t - timeout_start_time)

            request_open = False
            request_target_sp = None
            request_timeout_deadline = float("inf")
            timeout_active = False
            timeout_start_time = None

        current_sp = next_sp

    if request_open:
        unfinished_count += 1
        final_t = float(time_axis[-1])
        final_x = float(pos[-1])
        final_v = float(speed[-1])
        events.append(
            SPSEventRecord(
                kind=_EVENT_REQUEST_UNFINISHED,
                time_s=final_t,
                pos_m=final_x,
                speed_mps=final_v,
                current_sp=current_sp,
                request_target_sp=request_target_sp,
            )
        )
        if timeout_active and timeout_start_time is not None:
            delay_window_total_s += max(0.0, final_t - timeout_start_time)

    delay_related_total = (
        delay_related_min_violation_count + delay_related_max_violation_count
    )
    triggered_pass = request_count > 0
    delay_related_boundary_violation_pass = delay_related_total == 0

    first_failure_reason: str | None = None
    if not triggered_pass:
        first_failure_reason = "no_step_request_triggered"
    elif not delay_related_boundary_violation_pass:
        first_failure_reason = "delay_related_boundary_violation"

    return SPSComplianceResult(
        label=label,
        triggered_pass=triggered_pass,
        delay_related_boundary_violation_pass=delay_related_boundary_violation_pass,
        request_count=request_count,
        complete_count=complete_count,
        unfinished_count=unfinished_count,
        min_violation_count=min_violation_count,
        max_violation_count=max_violation_count,
        pre_timeout_min_violation_count=pre_timeout_min_violation_count,
        pre_timeout_max_violation_count=pre_timeout_max_violation_count,
        delay_related_min_violation_count=delay_related_min_violation_count,
        delay_related_max_violation_count=delay_related_max_violation_count,
        delay_window_total_s=delay_window_total_s,
        delay_tolerance_s=delay_tolerance_s,
        first_failure_reason=first_failure_reason,
        events=events,
    )


def _print_result_summary(result: SPSComplianceResult) -> None:
    trigger_status = "PASS" if result.triggered_pass else "FAIL"
    delay_status = "PASS" if result.delay_related_boundary_violation_pass else "FAIL"

    print(f"[{result.label}]")
    print(f"  triggered: {trigger_status}")
    print(f"  delay_related_boundary_violation: {delay_status}")
    print(f"  request_count: {result.request_count}")
    print(f"  complete_count: {result.complete_count}")
    print(f"  unfinished_count: {result.unfinished_count}")
    print(
        "  boundary_violations(min/max): "
        f"{result.min_violation_count}/{result.max_violation_count}"
    )
    print(
        "  delay_related_violations(min/max): "
        f"{result.delay_related_min_violation_count}/"
        f"{result.delay_related_max_violation_count}"
    )
    print(f"  delay_window_total_s: {result.delay_window_total_s:.6f}")
    print(f"  delay_tolerance_s: {result.delay_tolerance_s:.6f}")
    print(
        "  first_failure_reason: "
        f"{result.first_failure_reason if result.first_failure_reason else 'none'}"
    )


def _print_comparison_summary(
    *,
    dp_result: SPSComplianceResult,
    rl_result: SPSComplianceResult,
) -> None:
    print("\n[DP vs RL summary]")
    print(
        f"  request_count (dp/rl): {dp_result.request_count}/{rl_result.request_count}"
    )
    print(
        "  complete_count (dp/rl): "
        f"{dp_result.complete_count}/{rl_result.complete_count}"
    )
    print(
        "  delay_window_total_s (dp/rl): "
        f"{dp_result.delay_window_total_s:.6f}/{rl_result.delay_window_total_s:.6f}"
    )
    print(
        "  delay_related_violation_count (dp/rl): "
        f"{dp_result.delay_related_min_violation_count + dp_result.delay_related_max_violation_count}/"  # noqa: E501
        f"{rl_result.delay_related_min_violation_count + rl_result.delay_related_max_violation_count}"  # noqa: E501
    )


def _plot_event_markers(
    *,
    ax: Any,
    result: SPSComplianceResult,
    color: str,
    trajectory_label: str,
    annotation_mode: str,
    max_text_annotations: int,
) -> None:
    request_events = [e for e in result.events if e.kind == _EVENT_REQUEST_START]
    complete_events = [e for e in result.events if e.kind == _EVENT_STEP_COMPLETE]

    if request_events:
        ax.scatter(
            [event.pos_m for event in request_events],
            [event.speed_mps * 3.6 for event in request_events],
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
            [event.pos_m for event in complete_events],
            [event.speed_mps * 3.6 for event in complete_events],
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

    for idx, event in enumerate(request_events, start=1):
        ax.annotate(
            f"R{idx}",
            xy=(event.pos_m, event.speed_mps * 3.6),
            xytext=(3.0, 4.0),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    for idx, event in enumerate(complete_events, start=1):
        ax.annotate(
            f"C{idx}",
            xy=(event.pos_m, event.speed_mps * 3.6),
            xytext=(3.0, -9.0),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )


def _plot_sps_main_figure(
    *,
    dp_pos_arr: np.ndarray,
    dp_speed_arr: np.ndarray,
    rl_pos_arr: np.ndarray,
    rl_speed_arr: np.ndarray,
    dp_result: SPSComplianceResult,
    rl_result: SPSComplianceResult,
    no_safeguard: bool,
    factor: float,
    annotation_mode: str,
    max_text_annotations: int,
    safeguard: SafeGuardUtility | None,
) -> None:
    apply_rl_curve_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if not no_safeguard:
        resolved_safeguard = (
            safeguard if safeguard is not None else build_safeguard_utility(factor)
        )
        resolved_safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

    ax.plot(
        dp_pos_arr,
        dp_speed_arr * 3.6,
        color="tab:red",
        linewidth=1.5,
        alpha=0.9,
        label="DP trajectory",
        zorder=4,
    )
    ax.plot(
        rl_pos_arr,
        rl_speed_arr * 3.6,
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

    ax.set_title("DP/RL SPS compliance (speed-position)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    _deduplicate_legend(ax)

    plt.tight_layout()
    plt.show()


def _build_json_payload(
    *,
    schedule_time_s: float,
    step_delay_s: float,
    trajectory_source: str,
    dp_artifact: OptimizedCurveArtifact,
    rl_artifact: OptimizedCurveArtifact,
    dp_result: SPSComplianceResult,
    rl_result: SPSComplianceResult,
) -> dict[str, object]:
    return {
        "schedule_time_s": schedule_time_s,
        "step_delay_s": step_delay_s,
        "trajectory_source": trajectory_source,
        "artifacts": {
            "dp": {
                "npz_path": dp_artifact.npz_path,
                "metrics_path": dp_artifact.metrics_path,
            },
            "rl": {
                "npz_path": rl_artifact.npz_path,
                "metrics_path": rl_artifact.metrics_path,
            },
        },
        "results": {
            "dp": dp_result.to_dict(),
            "rl": rl_result.to_dict(),
        },
    }


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.step_delay_s <= 0.0:
        parser.error("--step-delay-s must be positive")
    if args.boundary_eps < 0.0:
        parser.error("--boundary-eps must be >= 0")
    if args.max_text_annotations < 0:
        parser.error("--max-text-annotations must be >= 0")

    try:
        output_modes = _parse_output_mode(args.output_mode)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        dp_artifact, rl_artifact = _resolve_curve_artifacts(
            dp_curve_dir=args.dp_curve_dir,
            rl_curve_dir=args.rl_curve_dir,
            trajectory_source=args.trajectory_source,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    dp_pos_arr, dp_speed_arr, dp_metrics = load_dp_curve_artifact(dp_artifact)
    rl_pos_arr, rl_speed_arr, rl_metrics = load_rl_curve_artifact(rl_artifact)

    schedule_time_s = _resolve_target_schedule_time(
        dp_metrics=dp_metrics,
        rl_metrics=rl_metrics,
        schedule_time_s_override=args.schedule_time_s,
    )

    _, track, _, _ = build_scenario(schedule_time_s=schedule_time_s)
    safeguard_utility = build_safeguard_utility(args.factor)

    dp_result = replay_sps_compliance(
        label="DP",
        pos_arr=dp_pos_arr,
        speed_arr=dp_speed_arr,
        sgu=safeguard_utility,
        asa_ap_list=track.ASA_aps,
        asa_dp_list=track.ASA_dps,
        step_delay_s=args.step_delay_s,
        boundary_eps=args.boundary_eps,
    )
    rl_result = replay_sps_compliance(
        label="RL",
        pos_arr=rl_pos_arr,
        speed_arr=rl_speed_arr,
        sgu=safeguard_utility,
        asa_ap_list=track.ASA_aps,
        asa_dp_list=track.ASA_dps,
        step_delay_s=args.step_delay_s,
        boundary_eps=args.boundary_eps,
    )

    if _OUTPUT_MODE_TEXT in output_modes:
        print(f"Using DP curve file: {dp_artifact.npz_path}")
        print(f"Using RL curve file: {rl_artifact.npz_path}")
        print(f"Resolved schedule_time_s: {schedule_time_s:.6f}")
        print(f"Replay step_delay_s (T_s): {args.step_delay_s:.6f}")
        print()
        _print_result_summary(dp_result)
        print()
        _print_result_summary(rl_result)
        _print_comparison_summary(dp_result=dp_result, rl_result=rl_result)

    if _OUTPUT_MODE_JSON in output_modes:
        payload = _build_json_payload(
            schedule_time_s=schedule_time_s,
            step_delay_s=args.step_delay_s,
            trajectory_source=args.trajectory_source,
            dp_artifact=dp_artifact,
            rl_artifact=rl_artifact,
            dp_result=dp_result,
            rl_result=rl_result,
        )
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json_output_path:
            output_path = Path(args.json_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload_text, encoding="utf-8")
            print(f"JSON report saved to: {output_path}")
        else:
            print(payload_text)

    if _OUTPUT_MODE_PLOT in output_modes:
        _plot_sps_main_figure(
            dp_pos_arr=dp_pos_arr,
            dp_speed_arr=dp_speed_arr,
            rl_pos_arr=rl_pos_arr,
            rl_speed_arr=rl_speed_arr,
            dp_result=dp_result,
            rl_result=rl_result,
            no_safeguard=args.no_safeguard,
            factor=args.factor,
            annotation_mode=args.event_annotation,
            max_text_annotations=args.max_text_annotations,
            safeguard=safeguard_utility,
        )


if __name__ == "__main__":
    main()
