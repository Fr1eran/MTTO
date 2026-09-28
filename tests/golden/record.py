"""Record golden regression snapshots to output/golden/ (not tracked by git).

Usage: uv run python -m tests.golden.record [--case NAME] [--force]
                                             [--regenerate-actions NAME]

Input action sequences (tests/golden/actions/*.npy) are tracked in git and
replayed as-is; pass --regenerate-actions to regenerate a specific RL case's
sequence from its controller instead (default: reuse the frozen sequence).
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np

import mtto
from paper.experiments.runner import git_state

from .cases import DP_CASES, REWARD_PRESETS, RL_CASES, SCHEDULE_CHANGE_CASES
from .drive import (
    ACTIONS_DIR,
    OUTPUT_DIR,
    Flags,
    Numerics,
    drive_dp,
    drive_rl,
    drive_rl_case,
    generate_actions,
    load_actions,
)

CASE_NAMES = tuple(case.name for case in (*RL_CASES, *SCHEDULE_CHANGE_CASES, *DP_CASES))
RL_CASE_NAMES = tuple(case.name for case in RL_CASES)


def _write(name: str, numerics: Numerics, flags: Flags, *, force: bool) -> None:
    npz_path = OUTPUT_DIR / f"{name}.npz"
    json_path = OUTPUT_DIR / f"{name}.json"
    if not force and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(f"{name}: golden data exists; pass --force to replace")
    np.savez_compressed(npz_path, **numerics)
    json_path.write_text(
        json.dumps(flags, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _check_expected_outcome(case, flags: Flags) -> None:
    actual = flags[REWARD_PRESETS[0]]["termination_reason"][-1]
    if actual != case.expected_outcome:
        print(
            f"{case.name}: expected outcome {case.expected_outcome!r}, got {actual!r}"
        )


def record(name: str, *, force: bool, regenerate_actions: frozenset[str]) -> None:
    for case in RL_CASES:
        if case.name == name:
            if name in regenerate_actions:
                np.save(ACTIONS_DIR / f"{name}.npy", generate_actions(case))
            numerics, flags = drive_rl_case(load_actions(name))
            _check_expected_outcome(case, flags)
            _write(name, numerics, flags, force=force)
            return
    for case in SCHEDULE_CHANGE_CASES:
        if case.name == name:
            _write(name, *drive_rl(load_actions(case.actions_from), case), force=force)
            return
    for case in DP_CASES:
        if case.name == name:
            _write(name, *drive_dp(case), force=force)
            return
    raise ValueError(f"unknown case: {name}")


def _write_manifest() -> None:
    commit, dirty = git_state()
    cases = sorted(p.stem for p in OUTPUT_DIR.glob("*.json"))
    manifest = {
        "mtto_version": mtto.__version__,
        "git_commit": commit,
        "dirty": dirty,
        "recorded_at": datetime.now(UTC).isoformat(),
        "cases": cases,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, action="append")
    parser.add_argument(
        "--force", action="store_true", help="replace existing golden data"
    )
    parser.add_argument(
        "--regenerate-actions",
        choices=RL_CASE_NAMES,
        action="append",
        default=[],
        help="regenerate the frozen action sequence for the named RL case",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regenerate_actions = frozenset(args.regenerate_actions)
    # RL cases first: schedule-change cases replay their frozen actions.
    for name in args.case or CASE_NAMES:
        record(name, force=args.force, regenerate_actions=regenerate_actions)
        print(f"recorded {name}")
    _write_manifest()
    print(f"wrote {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
