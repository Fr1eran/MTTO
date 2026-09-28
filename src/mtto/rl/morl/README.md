# `rl/morl` (placeholder)

This package is a placeholder for multi-objective reinforcement learning (MORL)
and other Pareto-set algorithms. No code exists here yet.

## Why this is easy to add later

- `Task.schedule_time_s` may be `None`; when it is, quality assessment
  (`evaluation/quality.py`) does not include arrival-time judgment, so a
  schedule-free multi-objective task is already representable.
- `rl/rewards.py` already returns rewards broken down by component
  (`RewardBreakdown`) instead of a single scalar.
- `evaluation/quality.py` has no dependency on rewards.
- `domain/` functions (kinematics, energy, safeguard, SRTSP, ...) can be
  reused directly by any new algorithm.

## What implementing this will require

- Implement the environment state transition and the objective definition
  for the chosen MORL or Pareto-set algorithm inside `rl/morl/` (or a new
  sibling algorithm package).
- Add Pareto-front analysis based on speed-profile quality metrics under
  `evaluation/`.

See the architecture section of the project `README.md` for the design these seams rely on.
