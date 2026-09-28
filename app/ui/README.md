# `app/ui` (placeholder)

This directory is a placeholder for a future UI aimed at non-expert users.
No code exists here yet.

## Why this is easy to add later

- The UI is only allowed to call `mtto.workflows` (the same entry points
  used by `cli.py` and `paper/`): `train`, `evaluate`, `solve_dp`,
  `analyze_training`.
- `workflows` functions accept a config object, return a result object
  containing `SpeedProfile`, and report progress through an optional
  callback; they do not depend on argparse, printing, or matplotlib.
- `Scenario` and `Task` are frozen dataclasses; line data is loaded by path
  (`mtto.io.scenario`).
- Each run's `run.json` already records the full configuration and inputs,
  so a database schema can be designed later from real query needs
  (see `app/api/README.md`).

## What implementing this will require

- Build the UI here, calling only `workflows`.
- `workflows` will need a cancellation mechanism for long-running tasks.
- Long tasks (training, DP) should run in a separate process.
- Choices of service framework, front-end charting, real-operation-data
  import, and authentication are deferred until this is built.

See the architecture section of the project `README.md` for the design these seams rely on.
