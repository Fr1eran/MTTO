# Agent Instructions

## 1. Environment & Deterministic Commands
- Package and runner: `uv` (never call `python`, `pip`, or `poetry` directly)
- Format & Lint: `uv run ruff check --fix && uv run ruff format`
- Tests: `uv run pytest`
- Package management: Require approval before running `uv add`

## 2. Code Style & Defensiveness Boundaries
- Favor standard library idioms, explicit type hints, and flat structure over indirection.
- Trust internal contracts: Do NOT write redundant `isinstance` type checks, paranoid `None` guards, or `try-except` blocks around trusted internal calls.
- Let exceptions propagate naturally (EAFP). Handle errors only at genuine boundaries (external I/O, user input, network payloads).
- Do not introduce single-use helper functions, abstract base classes, or factory patterns for trivial logic.

## 3. Testing Philosophy (pytest)
- Write tests only for non-trivial domain logic, stateful behaviors, and regressions.
- Prohibited tests:
  - Trivial getters, setters, dataclass initialization, or standard library pass-throughs.
  - Mock-only tests that merely verify an internal function was invoked with arguments without verifying outcome state.
  - Tests written solely to inflate code coverage metrics.
- Group tests logically; prefer parameterized tests (`@pytest.mark.parametrize`) over repetitive test functions.

## 4. Verification Gate
- Run `uv run ruff check` and `uv run ruff format --check` on touched files.
- Run `uv run pytest` on relevant test suites before marking the task complete.
