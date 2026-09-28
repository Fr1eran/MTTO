"""Compare the current implementation against the recorded regression snapshots.

Snapshots are the current version's regression baseline for computation logic:
a change to their values that is not explained by an intentional change to
computation logic means the computation was modified without bumping the
version number. Snapshots live in ``output/golden/`` (not tracked by git, see
README.md "Golden 回归快照"); record them with
``uv run python -m tests.golden.record`` before running these tests
(``uv run pytest -m golden``).

Tolerances: discrete values must match exactly; pure numerics use
rtol=1e-9, rewards and energies rtol=1e-7.
"""

import json

import numpy as np
import pytest

from .cases import DP_CASES, RL_CASES, SCHEDULE_CHANGE_CASES
from .drive import (
    OUTPUT_DIR,
    Numerics,
    drive_dp,
    drive_rl,
    drive_rl_case,
    load_actions,
    manifest_context,
    require_snapshot,
)

pytestmark = pytest.mark.golden

PURE_NUMERIC = {"rtol": 1e-9, "atol": 1e-12}
REWARD_OR_ENERGY = {"rtol": 1e-7, "atol": 1e-10}
OBSERVATION = {"rtol": 1e-6, "atol": 1e-7}


def _tolerance(key: str) -> dict[str, float]:
    field = key.rsplit(".", 1)[-1]
    if field == "observation":
        return OBSERVATION
    if field.startswith("reward") or "energy" in field:
        return REWARD_OR_ENERGY
    return PURE_NUMERIC


# DP accepts equivalent optima: only the objective, summary statistics and
# the quality-assessment snapshot are asserted; the stored path is kept for
# diagnosis.
DP_SUMMARY_KEYS = (
    "total_time_s",
    "total_energy_kj",
    "time_error_s",
    "min_upper_margin_mps",
)


def _assert_matches(
    name: str,
    numerics: Numerics,
    flags: dict,
    keys: tuple[str, ...] | None = None,
) -> None:
    require_snapshot(name)
    context = manifest_context()
    expected_flags = json.loads((OUTPUT_DIR / f"{name}.json").read_text("utf-8"))
    assert json.loads(json.dumps(flags)) == expected_flags, context
    with np.load(OUTPUT_DIR / f"{name}.npz") as expected:
        assert set(numerics) == set(expected.files), context
        for key in keys or expected.files:
            actual = numerics[key]
            assert actual.shape == expected[key].shape, f"{key}; {context}"
            np.testing.assert_allclose(
                actual,
                expected[key],
                equal_nan=True,
                err_msg=f"{key}; {context}",
                **_tolerance(key),
            )


@pytest.mark.parametrize("case", RL_CASES, ids=lambda case: case.name)
def test_rl_golden(case) -> None:
    _assert_matches(case.name, *drive_rl_case(load_actions(case.name)))


@pytest.mark.parametrize("case", SCHEDULE_CHANGE_CASES, ids=lambda case: case.name)
def test_schedule_change_golden(case) -> None:
    _assert_matches(case.name, *drive_rl(load_actions(case.actions_from), case))


@pytest.mark.parametrize("case", DP_CASES, ids=lambda case: case.name)
def test_dp_golden(case) -> None:
    numerics, flags = drive_dp(case)
    keys = DP_SUMMARY_KEYS + tuple(k for k in numerics if k.startswith("quality."))
    _assert_matches(case.name, numerics, flags, keys)
    with np.load(OUTPUT_DIR / f"{case.name}.npz") as expected:
        for key in ("propulsion_energy_kj", "levitation_energy_kj"):
            np.testing.assert_allclose(
                numerics[key][-1], expected[key][-1], err_msg=key, **REWARD_OR_ENERGY
            )
