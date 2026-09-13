"""Load the canonical representative-policy selection artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

PAPER_POLICY_SELECTION_PROTOCOL_VERSION = 5
SUPPORTED_POLICY_SELECTION_PROTOCOL_VERSIONS = (5,)


def load_selected_policy_dir(selection_file: str | Path) -> Path:
    """Return the selected policy directory from a validated JSON artifact."""
    path = Path(selection_file)
    if not path.is_file():
        raise FileNotFoundError(f"Policy selection file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid policy selection JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Policy selection payload must be an object")
    if payload.get("artifact_type") != "paper_policy_selection":
        raise ValueError("Unsupported policy selection artifact_type")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported policy selection schema_version")
    if (
        payload.get("protocol_version")
        not in SUPPORTED_POLICY_SELECTION_PROTOCOL_VERSIONS
    ):
        raise ValueError("Unsupported policy selection protocol_version")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Policy selection is missing selected candidate")
    raw_dir = selected.get("model_dir")
    if not isinstance(raw_dir, str) or not raw_dir:
        raise ValueError("Selected candidate is missing model_dir")
    policy_dir = Path(raw_dir)
    if not policy_dir.is_absolute():
        policy_dir = (path.parent / policy_dir).resolve()
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Selected policy directory not found: {policy_dir}")
    return policy_dir
