"""Coordinator.

Selects channels from the catalog that apply to a given policy. Rule
based first. Only falls back to an LLM call if the rule returns nothing,
which would be a catalog gap worth flagging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.schemas import StructuredPolicy

from ._client import call_tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHANNEL_CATALOG = PROJECT_ROOT / "configs" / "channel_catalog.yaml"


def _load_catalog() -> list[dict]:
    with CHANNEL_CATALOG.open() as fh:
        return yaml.safe_load(fh)["channels"]


def select_channels(policy: StructuredPolicy, run_id: Optional[str] = None) -> list[str]:
    """Return channel_ids whose applicable_policy_types include this policy type."""
    catalog = _load_catalog()
    policy_type = policy.policy_type.value
    matched = [
        c["id"] for c in catalog
        if policy_type in c.get("applicable_policy_types", [])
    ]
    if matched:
        return matched

    # Fallback: no rule match. Ask the LLM to propose channels out of the
    # full catalog. This covers edge cases where the catalog is incomplete.
    return _llm_channel_selection(policy, catalog, run_id=run_id)


def _llm_channel_selection(
    policy: StructuredPolicy,
    catalog: list[dict],
    run_id: Optional[str] = None,
) -> list[str]:
    catalog_summary = "\n".join(
        f"- {c['id']}: {c['description']} (applicable: {c.get('applicable_policy_types', [])})"
        for c in catalog
    )
    tool_schema = {
        "type": "object",
        "properties": {
            "channel_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact channel_ids from the provided catalog.",
            },
            "rationale": {"type": "string"},
        },
        "required": ["channel_ids", "rationale"],
    }
    data = call_tool(
        system=(
            "You select transmission channels from a catalog that are relevant to a given "
            "scenario. Return channel_ids exactly as they appear. No new ids."
        ),
        cacheable_context=f"CHANNEL CATALOG\n{catalog_summary}",
        user=(
            f"Scenario: {policy.raw_input}\npolicy_type={policy.policy_type.value}\n"
            f"subject={policy.subject}\nmagnitude={policy.magnitude} {policy.magnitude_unit}\n"
            "Return the relevant channel ids."
        ),
        tool_name="submit_channels",
        tool_description="Submit the list of relevant channel ids.",
        tool_schema=tool_schema,
        run_id=run_id,
        caller="coordinator",
    )
    # Trust but verify: strip to known ids
    known = {c["id"] for c in catalog}
    return [cid for cid in data.get("channel_ids", []) if cid in known]
