"""Per scenario event researcher.

Given a StructuredPolicy, ask Claude (with web search enabled) to
surface 30-80 historical events that could inform the analysis. The
events drive downstream analog retrieval: the similarity engine picks
the top 20 and attaches their realized responses.

This replaces the static event_catalog.csv as the PRIMARY source of
analog events. The static catalog still exists as a seed for grounding
and for teammate reference, but the researcher can go much further:

  - events specific to THIS scenario (Hormuz tanker attacks, rate
    decision surprises, tariff retaliation announcements)
  - recent events from the last year or two that may be absent from the
    static catalog
  - international events for cross country hypotheses

Web search is Anthropic's server side web_search_20250305 tool. Claude
drives the search transparently, synthesizes results, and returns a
structured list via the submit_events custom tool.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

from src.schemas import HistoricalEpisode, PolicyType, StructuredPolicy

from ._client import call_tool_async, DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_CATALOG = PROJECT_ROOT / "configs" / "event_catalog.csv"


SYSTEM = """You are the macro events researcher. Given a policy scenario, your
job is to produce a list of 30 to 80 historical events that could inform the
analysis. You have access to web_search. Use it liberally.

Prioritize in this order:

1. DIRECT ANALOGS (the most important). Events whose MECHANISM most closely
   matches the scenario. Examples:
   - Scenario: Strait of Hormuz closure -> Gulf tanker attacks (1987, 2019),
     Iran-Iraq tanker war, Libya supply outages, Venezuela sanctions, Nord
     Stream incidents, Suez blockage, refinery outages.
   - Scenario: Fed 50bp cut -> other -50bp cuts (2020-03-03, 2024-09-18, SVB
     response, emergency cuts, intermeeting cuts).
   - Scenario: China tariff -> Section 301 tranches, Section 232 steel,
     Smoot-Hawley, 1960s textile tariffs.

2. REGIME ANALOGS. Events with a similar macro backdrop (high inflation vs
   low, hiking vs easing cycle, strong vs weak dollar). These help condition
   the downstream response estimation.

3. CONTRASTING EVENTS. Events with a similar trigger but a DIFFERENT outcome.
   These let the adversary test robustness. E.g. geopolitical shocks that
   did NOT push oil, Fed cuts that did NOT rally equities.

4. RECENT EVENTS. 2023-present should be well represented. Web search is
   how you get these; your training data is stale.

You DO NOT need to make up or invent events. Use web search to verify dates
and magnitudes, especially for events after your training cutoff. If you are
unsure about a date, search to confirm.

For each event emit:
  date (YYYY-MM-DD) - real historical date, verified where possible
  description (one sentence) - what happened
  subject (snake_case noun) - the thing changed or triggered
  policy_type (monetary, fiscal, trade, regulatory, geopolitical)
  magnitude (number) - 0 when irrelevant
  magnitude_unit (percent, basis_points, probability, usd_billion, etc.)
  relevance_notes (one sentence) - why this matches the scenario

Aim for 50 events. Cover at least 3 decades where relevant. Do NOT dedupe
aggressively; near duplicates are fine if they are different episodes.

Output via the submit_events tool.
"""


EVENTS_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "minItems": 30,
            "maxItems": 100,
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "description": {"type": "string"},
                    "subject": {"type": "string"},
                    "policy_type": {
                        "type": "string",
                        "enum": [pt.value for pt in PolicyType],
                    },
                    "magnitude": {"type": "number"},
                    "magnitude_unit": {"type": "string"},
                    "relevance_notes": {"type": "string"},
                },
                "required": [
                    "date", "description", "subject", "policy_type",
                    "magnitude", "magnitude_unit", "relevance_notes",
                ],
            },
        },
    },
    "required": ["events"],
}


def _seed_catalog_text(max_rows: int = 30) -> str:
    """Provide the static catalog as a seed so the researcher has a starting
    point and can cite / extend it. Truncate to keep the prompt bounded.
    """
    if not SEED_CATALOG.exists():
        return ""
    lines = SEED_CATALOG.read_text().splitlines()
    header, rows = lines[0], lines[1:]
    sample = rows[:max_rows] + (["... (and more, use web search to extend)"] if len(rows) > max_rows else [])
    return "=== SEED CATALOG (extend, do not duplicate) ===\n" + "\n".join([header] + sample)


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce_event(raw: dict) -> Optional[dict]:
    """Validate and clean one event dict. Returns None if unusable."""
    d = raw.get("date", "").strip()
    if not ISO_DATE_RE.match(d):
        return None
    try:
        event_date = date.fromisoformat(d)
    except ValueError:
        return None
    desc = (raw.get("description") or "").strip()
    subj = (raw.get("subject") or "").strip()
    if not desc or not subj:
        return None
    pt = raw.get("policy_type", "").strip().lower()
    if pt not in {p.value for p in PolicyType}:
        return None
    try:
        mag = float(raw.get("magnitude") or 0.0)
    except (TypeError, ValueError):
        mag = 0.0
    unit = (raw.get("magnitude_unit") or "").strip()
    notes = (raw.get("relevance_notes") or "").strip()
    return {
        "date": event_date,
        "description": desc,
        "subject": subj,
        "policy_type": pt,
        "magnitude": mag,
        "magnitude_unit": unit,
        "relevance_notes": notes,
    }


async def research_events_async(
    policy: StructuredPolicy,
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    web_search_max_uses: int = 6,
) -> list[dict]:
    """Run the event researcher with web search. Returns a list of dict
    events suitable for analog retrieval and specialist grounding.
    """
    user = (
        f"Scenario to research:\n"
        f"  raw_input: {policy.raw_input}\n"
        f"  policy_type: {policy.policy_type.value}\n"
        f"  subject: {policy.subject}\n"
        f"  magnitude: {policy.magnitude} {policy.magnitude_unit}\n"
        f"  horizon_days: {policy.horizon_days}\n\n"
        "Use web_search to build the event list. Submit via submit_events.\n"
        "Include recent events (last 24 months) found via search."
    )
    try:
        data = await call_tool_async(
            system=SYSTEM,
            cacheable_context=_seed_catalog_text(),
            user=user,
            tool_name="submit_events",
            tool_description="Return the list of historical events relevant to the scenario.",
            tool_schema=EVENTS_TOOL_SCHEMA,
            model=model,
            max_tokens=max_tokens,
            run_id=run_id,
            caller="event_researcher",
            enable_web_search=True,
            web_search_max_uses=web_search_max_uses,
        )
    except Exception as exc:
        print(f"  [event_researcher] FAILED: {type(exc).__name__}: {exc}")
        return []

    raw_events = data.get("events") or []
    clean: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for ev in raw_events:
        c = _coerce_event(ev)
        if c is None:
            continue
        key = (c["description"][:40].lower(), str(c["date"]))
        if key in seen:
            continue
        seen.add(key)
        clean.append(c)
    return clean


def research_events(
    policy: StructuredPolicy,
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    import asyncio
    return asyncio.run(research_events_async(policy, run_id=run_id, model=model))
