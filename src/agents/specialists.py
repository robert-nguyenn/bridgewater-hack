"""Specialist agents.

One module, five specialists. Each specialist has a prompt file at
configs/specialist_prompts/{id}.md. The shared runner loads the prompt,
injects the relevant subset of the channel catalog and event catalog,
and forces a tool call that returns a list of proposed hypotheses.

Critical behavior per team feedback:
  - Specialist prompts must push for hypothesis DIVERSITY: cross
    country perspectives, different transmission mechanisms, different
    regimes. Each specialist enumerates explicit angles in its prompt
    under a DIVERSITY MANDATE section.
  - Every hypothesis must carry a `perspective` tag so the coordinator
    and adversary can see what angle it represents.

Hypotheses from the LLM are ENRICHED with channel catalog defaults
(estimator, source hints, typical elasticity, default covariates) so
the LLM only has to do the conceptual work, not the bookkeeping.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from src.schemas import (
    Confounder,
    EstimatorType,
    HistoricalEpisode,
    Hypothesis,
    StructuredPolicy,
    VariableType,
)

from ._client import call_tool, call_tool_async, DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / "configs" / "specialist_prompts"
CHANNEL_CATALOG = PROJECT_ROOT / "configs" / "channel_catalog.yaml"
EVENT_CATALOG = PROJECT_ROOT / "configs" / "event_catalog.csv"


SPECIALIST_IDS = ["monetary", "supply_chain", "financial_conditions", "international", "behavioral"]


# ---------------------------------------------------------------------------
# Tool schema the LLM must produce
# ---------------------------------------------------------------------------
HYPOTHESIS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Exact id from the provided channel catalog.",
                    },
                    "perspective": {
                        "type": "string",
                        "description": "Which diversity angle this hypothesis represents. Required.",
                    },
                    "shock_variable": {
                        "type": "string",
                        "description": "Variable being shocked. Prefer FRED series IDs or documented HF paths.",
                    },
                    "response_variable": {
                        "type": "string",
                        "description": "Response variable. FRED ID or documented series.",
                    },
                    "historical_episodes": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                                "magnitude": {"type": "number"},
                                "notes": {"type": "string"},
                            },
                            "required": ["name", "date", "magnitude"],
                        },
                    },
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confounders": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "mechanism": {"type": "string"},
                                "proxy_variable": {"type": "string"},
                                "handling": {
                                    "type": "string",
                                    "enum": ["include_covariate", "sample_restriction", "identification"],
                                },
                                "expected_direction": {"type": "string"},
                            },
                            "required": ["name", "mechanism", "proxy_variable", "handling"],
                        },
                    },
                    "expected_sign": {
                        "type": "string",
                        "enum": ["positive", "negative", "ambiguous"],
                    },
                    "economic_rationale": {
                        "type": "string",
                        "description": "2 to 4 sentences explaining the mechanism.",
                    },
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "channel_id", "perspective", "shock_variable", "response_variable",
                    "historical_episodes", "confounders", "expected_sign", "economic_rationale",
                ],
            },
        },
    },
    "required": ["hypotheses"],
}


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------
def _load_channel_catalog() -> list[dict]:
    with CHANNEL_CATALOG.open() as fh:
        return yaml.safe_load(fh)["channels"]


def _channels_for_policy_type(policy_type: str) -> list[dict]:
    return [c for c in _load_channel_catalog()
            if policy_type in c.get("applicable_policy_types", [])]


def _load_event_catalog_text() -> str:
    # Inject the event catalog as raw CSV text so the LLM can reference it.
    return EVENT_CATALOG.read_text()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
def _load_specialist_prompt(specialist_id: str) -> str:
    path = PROMPT_DIR / f"{specialist_id}.md"
    if not path.exists():
        raise FileNotFoundError(f"Specialist prompt missing: {path}")
    return path.read_text()


def _build_available_variables_block() -> str:
    """Enumerate every variable the loader can resolve. Injected into the
    specialist prompt so the LLM stops inventing series names.
    """
    from src.loaders.core import FRED_SERIES, HF_FILES
    from src.loaders import ALIASES

    fred_ids = sorted({sid for group in FRED_SERIES.values() for sid in group})
    hf_paths = sorted(HF_FILES.keys())
    alias_pairs = sorted(ALIASES.items())

    lines = ["=== AVAILABLE VARIABLES ==="]
    lines.append("")
    lines.append("FRED series ids (Tier 1 preloaded, use these EXACTLY as shock_variable or response_variable):")
    for sid in fred_ids:
        lines.append(f"  - {sid}")
    lines.append("")
    lines.append("Friendly aliases (also accepted, they resolve to FRED ids):")
    for alias, target in alias_pairs:
        lines.append(f"  - {alias}  ->  {target}")
    lines.append("")
    lines.append("HF dataset files (reference by exact path when naming constructed series):")
    for p in hf_paths:
        lines.append(f"  - {p}")
    lines.append("")
    lines.append("Yahoo Finance tickers are also accepted as shock or response if no FRED series fits.")
    lines.append("Use short uppercase symbols like 'EEM', 'GLD', 'USDBRL=X'. The loader will live fetch.")
    lines.append("")
    lines.append("For scenario trigger variables that do NOT have historical data (e.g., the probability")
    lines.append("of a rare event), use a descriptive snake_case name. The system will mark these as")
    lines.append("first link edges and rely on analog retrieval for their effect.")
    return "\n".join(lines)


def _build_cacheable_context(channels: list[dict], event_catalog_csv: str) -> str:
    """Content that is stable per scenario and should be cached across calls."""
    channel_block = yaml.safe_dump({"channels": channels}, sort_keys=False)
    return (
        _build_available_variables_block() + "\n\n"
        "=== CHANNEL CATALOG (filtered to this policy type) ===\n"
        f"{channel_block}\n"
        "=== EVENT CATALOG ===\n"
        f"{event_catalog_csv}\n"
    )


def _build_user_prompt(policy: StructuredPolicy, channel_ids: list[str]) -> str:
    return (
        "Scenario:\n"
        f"  raw_input: {policy.raw_input}\n"
        f"  policy_type: {policy.policy_type.value}\n"
        f"  subject: {policy.subject}\n"
        f"  magnitude: {policy.magnitude} {policy.magnitude_unit}\n"
        f"  direction: {policy.direction}\n"
        f"  horizon_days: {policy.horizon_days}\n"
        f"  additional_context: {policy.additional_context or 'none'}\n\n"
        f"Candidate channels (use these IDs exactly, or explain why none apply): "
        f"{channel_ids}\n\n"
        "STRICT RULES for shock_variable and response_variable:\n"
        "  - Must be a FRED id, a friendly alias, or a yfinance ticker FROM THE "
        "AVAILABLE VARIABLES list.\n"
        "  - Do NOT wrap a description around the name. Just the bare identifier.\n"
        "  - Do NOT invent series ids. If no exact match exists, pick the closest "
        "approximation from the list.\n"
        "  - For the scenario trigger itself (a variable with no historical data) "
        "use a short snake_case name and the system will treat the edge as first link.\n"
        "  - Proxy variables in confounders must also follow these rules.\n\n"
        "Propose 3 to 5 hypotheses. Each must reference a channel_id from the list, "
        "specify a distinct perspective from your diversity mandate, name 2 to 6 "
        "historical episodes drawn from the event catalog or well documented events, "
        "and list at least 2 confounders with proxy variables.\n"
        "Return strictly via the submit_hypotheses tool call."
    )


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def _enrich(
    item: dict,
    specialist_id: str,
    catalog: list[dict],
    seq: int,
) -> Optional[Hypothesis]:
    """Build a full Hypothesis from an LLM item by merging with catalog defaults."""
    channel_id = item.get("channel_id", "")
    channel = next((c for c in catalog if c["id"] == channel_id), None)
    if channel is None:
        # Silently drop unknown channel ids. The coordinator will log the miss.
        return None

    # Episodes
    episodes: list[HistoricalEpisode] = []
    for ep in item.get("historical_episodes", []):
        try:
            episodes.append(HistoricalEpisode(
                name=str(ep["name"]),
                date=ep["date"],
                magnitude=float(ep["magnitude"]),
                notes=ep.get("notes"),
            ))
        except Exception:
            continue
    if not episodes:
        return None

    # Confounders
    confounders: list[Confounder] = []
    for cf in item.get("confounders", []):
        try:
            confounders.append(Confounder(
                name=cf["name"],
                mechanism=cf["mechanism"],
                proxy_variable=cf["proxy_variable"],
                handling=cf["handling"],
                expected_direction=cf.get("expected_direction"),
            ))
        except Exception:
            continue

    # Catalog defaults for machinery
    shock_type = VariableType(channel["shock_type"])
    response_type = VariableType(channel["response_type"])
    estimator = EstimatorType(channel["default_estimator"])
    default_covs: list[str] = channel.get("default_covariates", []) or []
    user_covs: list[str] = item.get("covariates", []) or []
    combined_covs = list(dict.fromkeys(default_covs + user_covs))

    perspective = item.get("perspective", "unspecified")
    rationale = item["economic_rationale"].strip()
    if perspective:
        rationale = f"[perspective: {perspective}] {rationale}"

    return Hypothesis(
        hypothesis_id=f"{specialist_id}_{seq}_{uuid.uuid4().hex[:6]}",
        proposed_by=specialist_id,
        channel_id=channel_id,
        shock_variable=item["shock_variable"],
        shock_type=shock_type,
        shock_source_hints=[channel["shock_variable_template"]],
        response_variable=item["response_variable"],
        response_type=response_type,
        response_source_hints=[channel["response_variable_template"]],
        estimator=estimator,
        specification_params={},
        historical_episodes=episodes,
        covariates=combined_covs,
        confounders=confounders,
        expected_sign=item["expected_sign"],
        economic_rationale=rationale,
        citations=item.get("citations", []) or [],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_specialist(
    specialist_id: str,
    policy: StructuredPolicy,
    channel_ids: list[str],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[Hypothesis]:
    if specialist_id not in SPECIALIST_IDS:
        raise ValueError(f"Unknown specialist: {specialist_id}. Valid: {SPECIALIST_IDS}")

    prompt = _load_specialist_prompt(specialist_id)
    catalog = _load_channel_catalog()
    relevant = [c for c in catalog if c["id"] in channel_ids] or _channels_for_policy_type(policy.policy_type.value)

    cacheable = _build_cacheable_context(relevant, _load_event_catalog_text())
    user_msg = _build_user_prompt(policy, channel_ids)

    data = call_tool(
        system=prompt,
        cacheable_context=cacheable,
        user=user_msg,
        tool_name="submit_hypotheses",
        tool_description="Submit the specialist's proposed hypotheses.",
        tool_schema=HYPOTHESIS_TOOL_SCHEMA,
        model=model,
        run_id=run_id,
        caller=f"specialist:{specialist_id}",
        max_tokens=6000,
    )

    results: list[Hypothesis] = []
    for i, item in enumerate(data.get("hypotheses", [])):
        hyp = _enrich(item, specialist_id, catalog, i)
        if hyp is not None:
            results.append(hyp)
    return results


def run_all_specialists(
    policy: StructuredPolicy,
    channel_ids: list[str],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, list[Hypothesis]]:
    """Sequential runner. Prefer run_all_specialists_parallel for production."""
    out: dict[str, list[Hypothesis]] = {}
    for sid in SPECIALIST_IDS:
        try:
            out[sid] = run_specialist(sid, policy, channel_ids, run_id=run_id, model=model)
        except Exception as exc:
            out[sid] = []
            print(f"  [specialist:{sid}] FAILED  {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Async parallel runner
# ---------------------------------------------------------------------------
async def run_specialist_async(
    specialist_id: str,
    policy: StructuredPolicy,
    channel_ids: list[str],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[Hypothesis]:
    if specialist_id not in SPECIALIST_IDS:
        raise ValueError(f"Unknown specialist: {specialist_id}")

    prompt = _load_specialist_prompt(specialist_id)
    catalog = _load_channel_catalog()
    relevant = [c for c in catalog if c["id"] in channel_ids] or _channels_for_policy_type(policy.policy_type.value)

    cacheable = _build_cacheable_context(relevant, _load_event_catalog_text())
    user_msg = _build_user_prompt(policy, channel_ids)

    data = await call_tool_async(
        system=prompt,
        cacheable_context=cacheable,
        user=user_msg,
        tool_name="submit_hypotheses",
        tool_description="Submit the specialist's proposed hypotheses.",
        tool_schema=HYPOTHESIS_TOOL_SCHEMA,
        model=model,
        run_id=run_id,
        caller=f"specialist:{specialist_id}",
        max_tokens=6000,
    )

    results: list[Hypothesis] = []
    for i, item in enumerate(data.get("hypotheses", [])):
        hyp = _enrich(item, specialist_id, catalog, i)
        if hyp is not None:
            results.append(hyp)
    return results


async def run_all_specialists_parallel(
    policy: StructuredPolicy,
    channel_ids: list[str],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    specialist_ids: Optional[list[str]] = None,
) -> dict[str, list[Hypothesis]]:
    """Run specialists concurrently via asyncio.gather. Each specialist's
    exceptions are caught and returned as an empty list.
    """
    import asyncio

    ids = specialist_ids or SPECIALIST_IDS

    async def _one(sid: str) -> tuple[str, list[Hypothesis]]:
        try:
            r = await run_specialist_async(sid, policy, channel_ids, run_id=run_id, model=model)
            return sid, r
        except Exception as exc:
            print(f"  [specialist:{sid}] FAILED  {type(exc).__name__}: {exc}")
            return sid, []

    results = await asyncio.gather(*(_one(sid) for sid in ids))
    return dict(results)
