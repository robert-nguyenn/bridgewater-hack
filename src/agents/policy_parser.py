"""Policy parser.

Uses Claude to convert a plain language policy description into a
StructuredPolicy. Uses forced tool call for structured output.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from src.schemas import PolicyType, StructuredPolicy

from ._client import call_tool


SYSTEM_PROMPT = """You are a macro policy parser. Your job is to extract a structured
representation of a user's plain language policy or scenario description.

Rules:
- Choose the best matching policy_type from: monetary, fiscal, trade, regulatory, geopolitical.
- Extract the magnitude as a number. If the user describes a probability, magnitude
  is 0 to 1. If the user describes basis points, magnitude is the bp number (e.g. 25 or -50).
  If percent change, use percent (e.g. 0.25 for 25 percent).
- Set magnitude_unit to one of: percent, basis_points, probability, percentage_points,
  usd_trillion, ratio. Pick the most natural for the described policy.
- direction should be positive if the magnitude is a hike/increase/escalation,
  negative if a cut/decrease/de-escalation, bidirectional if the user is asking
  about a two sided scenario.
- horizon_days is the time window the user is asking about. Default to 90 if not
  clear.
- subject is a short noun phrase identifying what is being changed or what event is
  being evaluated, e.g. "fed_funds_rate", "chinese_semiconductors", "strait_of_hormuz_closure".
- If the input is ambiguous, make reasonable defaults and explain them in
  additional_context.
"""


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "policy_type": {
            "type": "string",
            "enum": [pt.value for pt in PolicyType],
        },
        "subject": {"type": "string"},
        "magnitude": {"type": "number"},
        "magnitude_unit": {"type": "string"},
        "direction": {"type": "string", "enum": ["positive", "negative", "bidirectional"]},
        "horizon_days": {"type": "integer", "minimum": 1},
        "effective_date": {
            "type": "string",
            "description": "Optional ISO date YYYY-MM-DD if user specifies a date.",
        },
        "additional_context": {
            "type": "string",
            "description": "Any assumptions the parser made to resolve ambiguity.",
        },
    },
    "required": ["policy_type", "subject", "magnitude", "magnitude_unit",
                 "direction", "horizon_days"],
}


def parse_policy(raw_input: str, run_id: Optional[str] = None) -> StructuredPolicy:
    data = call_tool(
        system=SYSTEM_PROMPT,
        cacheable_context=None,
        user=f"Parse this policy or scenario description:\n\n{raw_input}",
        tool_name="submit_structured_policy",
        tool_description="Submit the parsed structured policy representation.",
        tool_schema=TOOL_SCHEMA,
        run_id=run_id,
        caller="policy_parser",
    )
    # Dates come as strings
    if data.get("effective_date"):
        try:
            data["effective_date"] = date.fromisoformat(data["effective_date"])
        except (TypeError, ValueError):
            data["effective_date"] = None

    return StructuredPolicy(
        raw_input=raw_input,
        policy_type=PolicyType(data["policy_type"]),
        subject=data["subject"],
        magnitude=float(data["magnitude"]),
        magnitude_unit=data["magnitude_unit"],
        direction=data["direction"],
        horizon_days=int(data["horizon_days"]),
        effective_date=data.get("effective_date"),
        additional_context=data.get("additional_context"),
    )
