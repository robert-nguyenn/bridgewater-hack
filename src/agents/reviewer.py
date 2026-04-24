"""Review agent.

Runs AFTER the synthesizer and BEFORE final ImpactMap assembly. Validates
the full chain end to end by taking each edge, its method estimates,
plots, and caveats, and asking Claude to flag issues in four categories:

  1. sign_mismatch: coefficient sign contradicts expected_sign on hypotheses
  2. sample_size: n too small for the claim being made
  3. r2: R^2 implausible (too high or too low for the channel)
  4. plot_vs_claim: the saved plot does not visually support the numbers
  5. invented_number: a number in caveats that does not trace to any method_estimate

Uses Claude's vision capability to actually LOOK at the plots.

Output: list of ReviewFlag objects attached to the ImpactMap.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from src.schemas import EdgeObject, Hypothesis, ReviewFlag

from ._client import call_with_images_async, DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]


REVIEWER_SYSTEM = """You are the review agent for a macro impact map.

You receive ONE edge of the graph with its method estimates, plots, and
caveats. You must decide whether the edge has any of the following
problems and emit zero or more review flags:

  - sign_mismatch: does the coefficient sign match what the hypothesis
    expected (positive, negative, ambiguous)? A mismatch is an ERROR.
  - sample_size: is the sample size plausibly enough for the claim? For
    daily regressions, n < 30 is a WARNING. For monthly, n < 24 is a
    WARNING. For event studies, n < 10 events is a WARNING.
  - r2: is the R^2 plausible? R^2 > 0.95 on economic data is almost
    always a WARNING (possibly data leakage or collinearity). R^2 = 0
    is a WARNING for supposedly strong channels.
  - plot_vs_claim: look at the attached plot. Does the scatter slope
    match the claimed coefficient sign? Do the point counts match n?
    Are there obvious outliers driving the fit? A mismatch is an ERROR.
  - invented_number: scan the caveats text. Does it reference any number
    that is NOT in the method_estimates block provided? If so, flag as
    ERROR.

For first link edges (where the source is the scenario itself, not a
market series), statistical critiques are less applicable. Do NOT flag
sample_size or r2 issues on first_link edges. DO flag plot_vs_claim
and sign_mismatch for first link edges.

Be specific in messages. Instead of "small sample", say
"sample_size=12 is below the 30 event minimum for this channel".
"""


REVIEWER_TOOL: dict = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                    "category": {
                        "type": "string",
                        "enum": [
                            "sign_mismatch", "sample_size", "r2",
                            "plot_vs_claim", "invented_number", "other",
                        ],
                    },
                    "message": {"type": "string"},
                },
                "required": ["severity", "category", "message"],
            },
        },
        "overall_verdict": {
            "type": "string",
            "enum": ["ok", "minor_issues", "serious_issues"],
        },
    },
    "required": ["flags", "overall_verdict"],
}


def _load_plot_bytes(plot_path: Optional[str]) -> Optional[tuple[str, bytes]]:
    if not plot_path:
        return None
    full = PROJECT_ROOT / plot_path
    if not full.exists():
        return None
    return ("image/png", full.read_bytes())


def _edge_summary(edge: EdgeObject, hypotheses_by_id: dict[str, Hypothesis]) -> str:
    method_lines = []
    for m in edge.method_estimates:
        coef = "n/a" if m.coefficient is None else f"{m.coefficient:+.4f}"
        se = "n/a" if m.standard_error is None else f"{m.standard_error:.4f}"
        n = m.sample_size or 0
        r2 = "n/a" if m.r_squared is None else f"{m.r_squared:.2f}"
        method_lines.append(
            f"  - {m.method.value}: coef={coef} SE={se} n={n} R^2={r2} "
            f"passed={m.passed}  plot={m.plot_path}\n"
            f"    notes: {m.notes}"
        )

    expected_signs = []
    for hid in edge.hypothesis_ids:
        h = hypotheses_by_id.get(hid)
        if h:
            expected_signs.append(f"{hid}: {h.expected_sign}")

    return (
        f"edge: {edge.source_node} -> {edge.target_node}\n"
        f"wave: {edge.wave}  is_first_link: {edge.is_first_link}\n"
        f"elasticity: point={edge.elasticity.point} "
        f"[{edge.elasticity.low}, {edge.elasticity.high}] {edge.elasticity.unit}\n"
        f"confidence: overall={edge.confidence.overall} "
        f"stat={edge.confidence.statistical} samp={edge.confidence.sample} "
        f"cross={edge.confidence.cross_method} regime={edge.confidence.regime}\n\n"
        f"method estimates:\n" + "\n".join(method_lines) + "\n\n"
        f"expected signs per hypothesis:\n  " + "\n  ".join(expected_signs) + "\n\n"
        f"caveats:\n" + "\n".join(f"  - {c}" for c in edge.caveats)
    )


async def _review_one_edge(
    edge: EdgeObject,
    hypotheses_by_id: dict[str, Hypothesis],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[ReviewFlag]:
    # Gather all plots for this edge
    images: list[tuple[str, bytes]] = []
    for m in edge.method_estimates:
        blob = _load_plot_bytes(m.plot_path)
        if blob:
            images.append(blob)

    text = _edge_summary(edge, hypotheses_by_id)
    target_id = f"{edge.source_node}->{edge.target_node}"

    # If no images, still review but note it.
    if not images:
        text += "\n\n(no plots were saved for this edge)"

    try:
        data = await call_with_images_async(
            system=REVIEWER_SYSTEM,
            text=text,
            images=images,
            tool_name="submit_review_flags",
            tool_description="Return review flags for this edge.",
            tool_schema=REVIEWER_TOOL,
            model=model,
            run_id=run_id,
            caller=f"reviewer:{target_id}",
            max_tokens=1500,
        )
    except Exception as exc:
        return [ReviewFlag(
            severity="warning",
            target=target_id,
            category="other",
            message=f"reviewer call failed: {type(exc).__name__}: {exc}",
        )]

    flags: list[ReviewFlag] = []
    for f in data.get("flags", []):
        try:
            flags.append(ReviewFlag(
                severity=f["severity"],
                target=target_id,
                category=f["category"],
                message=f["message"],
            ))
        except Exception:
            continue
    return flags


async def review_all_edges_async(
    edges: list[EdgeObject],
    hypotheses_by_id: dict[str, Hypothesis],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[ReviewFlag]:
    """Run the reviewer on every edge in parallel."""
    tasks = [_review_one_edge(e, hypotheses_by_id, run_id=run_id, model=model) for e in edges]
    results = await asyncio.gather(*tasks)
    flat: list[ReviewFlag] = []
    for flags in results:
        flat.extend(flags)
    return flat


def review_all_edges(
    edges: list[EdgeObject],
    hypotheses_by_id: dict[str, Hypothesis],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[ReviewFlag]:
    """Sync wrapper around the async reviewer."""
    return asyncio.run(review_all_edges_async(edges, hypotheses_by_id, run_id=run_id, model=model))
