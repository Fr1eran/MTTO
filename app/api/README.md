# `app/api` (placeholder)

This directory is a placeholder for a future API layer backing `app/ui`.
No code exists here yet.

## Why this is easy to add later

- The API is only allowed to call `mtto.workflows` (`train`, `evaluate`,
  `solve_dp`, `analyze_training`); it does not implement business logic
  itself.
- `workflows` functions accept a config object, return a result object
  containing `SpeedProfile`, and report progress through an optional
  callback; they do not depend on argparse, printing, or matplotlib.
- Every run's `run.json` (written by `mtto.io.artifacts`) already records
  the full configuration, inputs, and version provenance for that run, so a
  database is not required just to reproduce or describe a result.

## What implementing this will require

- Build the API here, calling only `workflows`.
- `workflows` will need a cancellation mechanism for long-running tasks.
- Long tasks (training, DP) should run in a separate process.
- A database schema, if needed for real query patterns (listing, search,
  status tracking across runs), should be designed once this layer exists,
  using `run.json` as the source of truth for each run's inputs.

See the architecture section of the project `README.md` for the design these seams rely on.
