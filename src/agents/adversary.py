"""Adversarial reviewer.

Two pass review:

1. Hypothesis level: for each hypothesis, add one or two additional
   confounders and tighten the identification strategy.

2. **Analog level** (per team feedback): for EACH historical_episode on
   EACH hypothesis, produce an analog specific critique. What was
   different about that episode's regime, what concurrent policy or
   shock was also active, what alternative mechanism could explain the
   observed response. This sits on HistoricalEpisode.adversarial_critique.

The analog critique is the single most important output of this agent,
because it prevents the user from mistaking "this looked like 2019 so
this time oil goes up" type reasoning when 2019 had confounders that
may not apply today.
"""
from __future__ import annotations

from typing import Any, Optional

from src.schemas import Confounder, Hypothesis

from ._client import call_tool, call_tool_async


HYPOTHESIS_REVIEW_SYSTEM = """You are the adversarial reviewer. Your job is
TWO things:

1. Strengthen salvageable hypotheses by identifying the strongest alternative
   explanations that could confound the result. Return additional confounders
   that are not already listed, each with a named proxy_variable.

2. KILL fundamentally broken hypotheses by including them in the
   rejected list with a specific reason. Reject when:
   - The claimed shock has no chance of identifying the response (e.g.,
     reverse causality dominates and cannot be proxied).
   - The claimed historical analogs span incompatible regimes and no
     subset is usable.
   - The economic mechanism as stated contradicts basic accounting
     identities or well established priors.
   - The response variable is not causally downstream of the shock on
     any plausible transmission path.

Do NOT reject merely because a hypothesis is imperfect. Everything has
confounders. Reject only when the hypothesis is not salvageable.

Confounders should be:
  - Specific and proxyable (include a proxy_variable name)
  - Not generic. "Omitted variable bias" is not acceptable.
  - Drawn from the standard critiques: Lucas critique for regime changes,
    simultaneity, reverse causality, sample selection, omitted third driver,
    pegged vs floating regime differences.

Return ONE tool call with both added_confounders (keyed by hypothesis_id)
and rejected (a list). Most hypotheses should have an empty rejected entry.
"""


ANALOG_REVIEW_SYSTEM = """You are the adversarial analog reviewer.

You receive a hypothesis and the historical episodes it claims as analogs.
For EVERY single episode listed, write a short critique that flags:
  - What was structurally different about that episode's regime
    (central bank rate level, dollar cycle phase, cycle position, global
    growth)
  - What concurrent policy or shock was also active at that date
  - What alternative mechanism could have driven the response that does
    NOT apply to the current scenario

Be specific, not generic. "Different macro conditions" is not useful.
"In June 2019 oil was also being supported by OPEC supply cuts
unrelated to Gulf tensions, inflating the price response" IS useful.

If an episode is a clean analog with no major concerns, say so briefly.
Do not manufacture concerns where none exist.

CRITICAL OUTPUT RULES:
- Return critiques for EVERY episode on the hypothesis, not a subset.
- Use the EXACT episode name string as the key. Copy it character for character
  from the "episodes:" block in the prompt. Do not paraphrase, summarize, or
  combine names. No bullets, no dates appended, no rewrites.
- Outer key is the hypothesis_id. Inner keys are the episode names.
"""


HYPOTHESIS_REVIEW_TOOL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "added_confounders": {
            "type": "object",
            "description": "Map hypothesis_id to list of confounders to add.",
            "additionalProperties": {
                "type": "array",
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
        },
        "rejected": {
            "type": "array",
            "description": "Hypotheses that cannot be salvaged. Usually empty.",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["hypothesis_id", "reason"],
            },
        },
    },
    "required": ["added_confounders"],
}


ANALOG_REVIEW_TOOL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "critiques": {
            "type": "object",
            "description": "Map hypothesis_id -> {episode_name -> critique string}.",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
    },
    "required": ["critiques"],
}


def _serialize_hypotheses_for_review(hypotheses: list[Hypothesis]) -> str:
    """Serialize to a compact JSON-like text block for LLM consumption."""
    blocks = []
    for h in hypotheses:
        ep_lines = [
            f"    - {ep.name} | {ep.date} | mag={ep.magnitude} | {ep.notes or ''}"
            for ep in h.historical_episodes
        ]
        conf_lines = [
            f"    - {c.name}: {c.mechanism} (proxy={c.proxy_variable})"
            for c in h.confounders
        ]
        blocks.append(
            f"Hypothesis id={h.hypothesis_id}\n"
            f"  proposed_by: {h.proposed_by}\n"
            f"  channel_id: {h.channel_id}\n"
            f"  shock: {h.shock_variable}  response: {h.response_variable}\n"
            f"  expected_sign: {h.expected_sign}\n"
            f"  rationale: {h.economic_rationale}\n"
            f"  episodes:\n" + "\n".join(ep_lines) + "\n"
            f"  existing confounders:\n" + ("\n".join(conf_lines) if conf_lines else "    (none)")
        )
    return "\n\n".join(blocks)


def _review_hypotheses(
    hypotheses: list[Hypothesis],
    run_id: Optional[str] = None,
) -> dict[str, list[Confounder]]:
    if not hypotheses:
        return {}
    data = call_tool(
        system=HYPOTHESIS_REVIEW_SYSTEM,
        cacheable_context=None,
        user=(
            "Review the following hypotheses and add confounders. Do not duplicate "
            "existing ones.\n\n" + _serialize_hypotheses_for_review(hypotheses)
        ),
        tool_name="submit_additional_confounders",
        tool_description="Return additional confounders per hypothesis id.",
        tool_schema=HYPOTHESIS_REVIEW_TOOL,
        run_id=run_id,
        caller="adversary:hypotheses",
        max_tokens=6000,
    )

    added: dict[str, list[Confounder]] = {}
    for hid, items in (data.get("added_confounders") or {}).items():
        for it in items:
            try:
                added.setdefault(hid, []).append(Confounder(
                    name=it["name"],
                    mechanism=it["mechanism"],
                    proxy_variable=it["proxy_variable"],
                    handling=it["handling"],
                    expected_direction=it.get("expected_direction"),
                ))
            except Exception:
                continue
    return added


def _review_analogs(
    hypotheses: list[Hypothesis],
    policy_context: str,
    run_id: Optional[str] = None,
) -> dict[str, dict[str, str]]:
    if not hypotheses:
        return {}
    data = call_tool(
        system=ANALOG_REVIEW_SYSTEM,
        cacheable_context=None,
        user=(
            f"Current scenario: {policy_context}\n\n"
            "Review the historical episodes claimed by each hypothesis. For every "
            "episode, produce an analog specific critique.\n\n"
            + _serialize_hypotheses_for_review(hypotheses)
        ),
        tool_name="submit_analog_critiques",
        tool_description=(
            "Return per hypothesis, per episode, an adversarial critique string."
        ),
        tool_schema=ANALOG_REVIEW_TOOL,
        run_id=run_id,
        caller="adversary:analogs",
        max_tokens=8000,
    )

    critiques = data.get("critiques") or {}
    # Normalize to a flat {hid: {episode_name: critique}}
    out: dict[str, dict[str, str]] = {}
    for hid, by_ep in critiques.items():
        if isinstance(by_ep, dict):
            out[hid] = {str(k): str(v) for k, v in by_ep.items() if isinstance(v, str)}
    return out


def adversarial_review(
    hypotheses: list[Hypothesis],
    policy_context: str = "",
    run_id: Optional[str] = None,
) -> tuple[list[Hypothesis], dict[str, str]]:
    """Sync wrapper. Returns (enriched_hypotheses, rejected_map)."""
    import asyncio
    return asyncio.run(adversarial_review_async(hypotheses, policy_context, run_id))


# ---------------------------------------------------------------------------
# Async parallel version. The two passes (hypothesis confounders, analog
# critiques) are independent and run concurrently.
# ---------------------------------------------------------------------------
async def _review_one_hypothesis(
    h: Hypothesis,
    run_id: Optional[str] = None,
) -> tuple[str, list[Confounder], Optional[str]]:
    """Adversarial confounder review for a SINGLE hypothesis.

    Returns (hypothesis_id, added_confounders, rejection_reason_or_None).
    Per hypothesis calls keep the output bounded so we never hit max_tokens.
    """
    serialized = _serialize_hypotheses_for_review([h])
    try:
        data = await call_tool_async(
            system=HYPOTHESIS_REVIEW_SYSTEM,
            cacheable_context=None,
            user=(
                "Review this hypothesis. EITHER add 2 to 4 confounders not already "
                "listed, OR reject it if it is fundamentally unsalvageable. Most "
                "hypotheses should be kept with added confounders; only reject when "
                "no fix would rescue the test.\n\n" + serialized
            ),
            tool_name="submit_additional_confounders",
            tool_description="Return additional confounders keyed by hypothesis_id, "
                             "plus any hypotheses to reject.",
            tool_schema=HYPOTHESIS_REVIEW_TOOL,
            run_id=run_id,
            caller=f"adversary:hypothesis:{h.hypothesis_id}",
            max_tokens=1500,
        )
    except Exception as exc:
        print(f"  [adversary hypothesis {h.hypothesis_id}] FAILED: {type(exc).__name__}")
        return h.hypothesis_id, [], None

    confounders: list[Confounder] = []
    items = (data.get("added_confounders") or {}).get(h.hypothesis_id, [])
    if not items and data.get("added_confounders"):
        for v in data["added_confounders"].values():
            if isinstance(v, list):
                items = v
                break
    for it in items:
        try:
            confounders.append(Confounder(
                name=it["name"],
                mechanism=it["mechanism"],
                proxy_variable=it["proxy_variable"],
                handling=it["handling"],
                expected_direction=it.get("expected_direction"),
            ))
        except Exception:
            continue

    # Rejection pass
    rejection_reason: Optional[str] = None
    for rej in data.get("rejected") or []:
        hid = rej.get("hypothesis_id") or ""
        if hid == h.hypothesis_id or hid in ("", "this", "self"):
            rejection_reason = rej.get("reason", "rejected by adversary")
            break

    return h.hypothesis_id, confounders, rejection_reason


async def _review_analogs_one_hypothesis(
    h: Hypothesis,
    policy_context: str,
    run_id: Optional[str] = None,
) -> tuple[str, dict[str, str]]:
    """Analog critiques for a SINGLE hypothesis. All episodes at once but
    scoped to one hypothesis so output stays under 2000 tokens.
    """
    serialized = _serialize_hypotheses_for_review([h])
    try:
        data = await call_tool_async(
            system=ANALOG_REVIEW_SYSTEM,
            cacheable_context=None,
            user=(
                f"Current scenario: {policy_context}\n\n"
                "Review the historical episodes claimed by this hypothesis. For EACH "
                "episode, produce an episode specific adversarial critique "
                "(1 to 2 sentences).\n\n" + serialized
            ),
            tool_name="submit_analog_critiques",
            tool_description="Return critiques keyed by hypothesis_id then by episode name.",
            tool_schema=ANALOG_REVIEW_TOOL,
            run_id=run_id,
            caller=f"adversary:analogs:{h.hypothesis_id}",
            max_tokens=2500,
        )
    except Exception as exc:
        print(f"  [adversary analogs {h.hypothesis_id}] FAILED: {type(exc).__name__}")
        return h.hypothesis_id, {}

    critiques = data.get("critiques") or {}
    # LLM may key by hypothesis_id or skip straight to episode map
    per_ep = critiques.get(h.hypothesis_id)
    if per_ep is None and critiques:
        # Fallback: first value if the LLM keyed by a different id
        for v in critiques.values():
            if isinstance(v, dict):
                per_ep = v
                break
    # Or the LLM may have returned episode names at the top level
    if per_ep is None and critiques and all(isinstance(v, str) for v in critiques.values()):
        per_ep = critiques

    if not isinstance(per_ep, dict):
        return h.hypothesis_id, {}
    return h.hypothesis_id, {str(k): str(v) for k, v in per_ep.items() if isinstance(v, str)}


async def adversarial_review_async(
    hypotheses: list[Hypothesis],
    policy_context: str = "",
    run_id: Optional[str] = None,
) -> tuple[list[Hypothesis], dict[str, str]]:
    """Per hypothesis parallel adversarial review.

    Returns (enriched_hypotheses, rejected_map) where rejected_map is
    hypothesis_id -> rejection reason. Rejected hypotheses are STILL
    returned in the enriched list (with their critiques attached) so
    they remain auditable, and the orchestrator filters them out via
    the rejected_map before running empirics.
    """
    import asyncio
    if not hypotheses:
        return [], {}

    hyp_tasks = [_review_one_hypothesis(h, run_id=run_id) for h in hypotheses]
    analog_tasks = [_review_analogs_one_hypothesis(h, policy_context=policy_context, run_id=run_id)
                    for h in hypotheses]
    results = await asyncio.gather(*hyp_tasks, *analog_tasks)

    n = len(hypotheses)
    hyp_results = results[:n]
    analog_results = results[n:]

    added: dict[str, list[Confounder]] = {hid: conf for hid, conf, _ in hyp_results}
    rejected: dict[str, str] = {
        hid: reason for hid, _, reason in hyp_results if reason is not None
    }
    analog: dict[str, dict[str, str]] = dict(analog_results)

    enriched: list[Hypothesis] = []
    for h in hypotheses:
        data = h.model_dump()
        existing_names = {c["name"] for c in data["confounders"]}
        for new_c in added.get(h.hypothesis_id, []):
            if new_c.name not in existing_names:
                data["confounders"].append(new_c.model_dump())
                existing_names.add(new_c.name)
        critique_map = analog.get(h.hypothesis_id, {})
        _attach_critiques_fuzzy(data["historical_episodes"], critique_map)
        enriched.append(Hypothesis.model_validate(data))
    return enriched, rejected


def _attach_critiques_fuzzy(episodes: list[dict], critique_map: dict[str, str]) -> None:
    """Match adversary critiques to episodes by name, with fuzzy fallback.

    Tries exact match first, then substring/overlap, then date proximity.
    """
    if not critique_map:
        return

    def _tokens(s: str) -> set[str]:
        return {t for t in s.lower().replace("-", " ").replace("/", " ").split() if len(t) > 2}

    # Normalize critique keys
    crit_items = [(k, _tokens(k), v) for k, v in critique_map.items() if v]
    used = [False] * len(crit_items)

    # Pass 1: exact case-insensitive match
    for ep in episodes:
        ep_name_norm = str(ep["name"]).strip().lower()
        for i, (ck, ctoks, cv) in enumerate(crit_items):
            if used[i]:
                continue
            if ck.strip().lower() == ep_name_norm:
                ep["adversarial_critique"] = cv
                used[i] = True
                break

    # Pass 2: substring or strong token overlap (>=60% of shorter side)
    for ep in episodes:
        if ep.get("adversarial_critique"):
            continue
        ep_toks = _tokens(ep["name"])
        best_i, best_score = -1, 0.0
        for i, (ck, ctoks, cv) in enumerate(crit_items):
            if used[i] or not ctoks or not ep_toks:
                continue
            overlap = len(ep_toks & ctoks)
            score = overlap / min(len(ep_toks), len(ctoks))
            # Also boost if one is a substring of the other
            if str(ep["name"]).lower() in ck.lower() or ck.lower() in str(ep["name"]).lower():
                score = max(score, 0.8)
            if score > best_score:
                best_score = score
                best_i = i
        if best_score >= 0.6 and best_i >= 0:
            ep["adversarial_critique"] = crit_items[best_i][2]
            used[best_i] = True

    # Pass 3: positional fallback, for episodes without a match, take any
    # remaining critique in order (ordered matching).
    remaining = [i for i, u in enumerate(used) if not u]
    uncritiqued = [ep for ep in episodes if not ep.get("adversarial_critique")]
    for ep, ri in zip(uncritiqued, remaining):
        ep["adversarial_critique"] = crit_items[ri][2]
        used[ri] = True
