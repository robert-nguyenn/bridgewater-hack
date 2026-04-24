"""Synthesizer.

Rolls up all method estimates for a single edge into a single
EstimateRange + ConfidenceBreakdown. Deterministic math for the
numerical aggregation. One LLM call per edge to generate human
readable caveats that surface the weak points.

First link vs downstream:
  First link edges have no true regression support (shock doesn't
  resolve to Tier 1 data). For these, the elasticity range comes from
  analog retrieval coefficients and the confidence is capped by the
  regime and cross method scores. Statistical confidence is hard coded
  to a low value because there is no valid regression.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Optional

from src.schemas import (
    ConfidenceBreakdown,
    EdgeObject,
    EstimateRange,
    EstimatorType,
    Hypothesis,
    MethodEstimate,
)

from ._client import call_tool, DEFAULT_MODEL


def synthesize_edge(
    edge: EdgeObject,
    hypotheses_by_id: dict[str, Hypothesis],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> EdgeObject:
    """Return a new EdgeObject with numerical rollup + LLM generated caveats."""
    valid = [m for m in edge.method_estimates if m.passed and m.coefficient is not None]

    elasticity, conf_stat, conf_sample, conf_cross = _numerical_rollup(valid, edge.is_first_link)

    # Regime confidence: use the max analog similarity if an analog estimate is present
    analog_ests = [m for m in edge.method_estimates if m.method == EstimatorType.ANALOG_RETRIEVAL and m.passed]
    if analog_ests:
        # We do not carry similarity scores on MethodEstimate, default to 0.5 when present
        conf_regime = 0.5
    else:
        conf_regime = 0.3

    # For first link edges, down weight statistical confidence
    if edge.is_first_link:
        conf_stat = min(conf_stat, 0.25)

    # Overall: weighted average. Heavier weight on cross method and regime.
    overall = _weighted_overall(conf_stat, conf_sample, conf_cross, conf_regime, edge.is_first_link)

    confidence = ConfidenceBreakdown(
        statistical=round(conf_stat, 3),
        sample=round(conf_sample, 3),
        cross_method=round(conf_cross, 3),
        regime=round(conf_regime, 3),
        overall=round(overall, 3),
    )

    # LLM caveats
    caveats = _generate_caveats(edge, hypotheses_by_id, valid, run_id=run_id, model=model)

    return EdgeObject(
        source_node=edge.source_node,
        target_node=edge.target_node,
        wave=edge.wave,
        elasticity=elasticity,
        confidence=confidence,
        lag_days=edge.lag_days,
        causal_share=edge.causal_share,
        method_estimates=edge.method_estimates,
        confounders_tested=edge.confounders_tested,
        caveats=caveats,
        hypothesis_ids=edge.hypothesis_ids,
        is_first_link=edge.is_first_link,
    )


def _numerical_rollup(
    valid: list[MethodEstimate],
    is_first_link: bool,
) -> tuple[EstimateRange, float, float, float]:
    """Compute elasticity range + statistical/sample/cross-method confidences."""
    if not valid:
        return (
            EstimateRange(point=0.0, low=0.0, high=0.0, unit="unknown"),
            0.0, 0.0, 0.0,
        )

    coefs = [float(m.coefficient) for m in valid]
    ses = [float(m.standard_error) if m.standard_error else 0.0 for m in valid]

    # Elasticity point: median
    point = median(coefs)

    # Low/high: 25th/75th percentile of (coef +/- 1 SE)
    lo_samples = sorted(c - s for c, s in zip(coefs, ses))
    hi_samples = sorted(c + s for c, s in zip(coefs, ses))
    if len(lo_samples) >= 4:
        lo = _percentile(lo_samples, 25)
        hi = _percentile(hi_samples, 75)
    elif len(lo_samples) > 1:
        lo = min(lo_samples)
        hi = max(hi_samples)
    else:
        lo = lo_samples[0]
        hi = hi_samples[0]
    # Ensure low <= point <= high
    lo = min(lo, point)
    hi = max(hi, point)

    # Statistical confidence from avg |t-stat|
    tstats = [abs(c) / s for c, s in zip(coefs, ses) if s and s > 0]
    avg_t = sum(tstats) / len(tstats) if tstats else 0.0
    # t=2 -> conf ~ 0.75, t>=3 -> conf ~ 0.9, t<1 -> low
    conf_stat = min(avg_t / 3.0, 0.95)

    # Sample confidence from max sample size
    n_max = max((m.sample_size or 0) for m in valid)
    # n >= 300 full conf, n >= 50 moderate, n < 20 low
    conf_sample = min(math.log10(n_max + 1) / 2.5, 0.95) if n_max > 0 else 0.1

    # Cross method confidence: sign agreement across methods + low CoV
    if len(coefs) >= 2:
        pos = sum(1 for c in coefs if c > 0)
        neg = sum(1 for c in coefs if c < 0)
        agreement = max(pos, neg) / len(coefs)
        cov = (max(coefs) - min(coefs)) / (abs(point) + 1e-9)
        cov_score = max(0.0, 1.0 - min(cov / 3.0, 1.0))
        conf_cross = 0.5 * agreement + 0.5 * cov_score
    else:
        conf_cross = 0.3  # single method, unknown cross agreement

    unit = valid[0].notes or ""
    if "beta" in unit.lower():
        unit_label = "beta"
    else:
        unit_label = "coefficient"

    return (
        EstimateRange(point=round(point, 4), low=round(lo, 4), high=round(hi, 4), unit=unit_label),
        conf_stat, conf_sample, conf_cross,
    )


def _percentile(sorted_xs: list[float], p: float) -> float:
    k = (len(sorted_xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_xs[int(k)]
    return sorted_xs[f] * (c - k) + sorted_xs[c] * (k - f)


def _weighted_overall(
    stat: float, samp: float, cross: float, regime: float, is_first_link: bool,
) -> float:
    if is_first_link:
        # First link: weight regime and cross_method heavily, statistical barely counts
        return 0.10 * stat + 0.15 * samp + 0.35 * cross + 0.40 * regime
    return 0.35 * stat + 0.20 * samp + 0.25 * cross + 0.20 * regime


# ---------------------------------------------------------------------------
# LLM caveats
# ---------------------------------------------------------------------------
CAVEATS_SYSTEM = """You are the synthesizer for a macro impact map. You receive one edge of the
impact graph: its source, target, the list of method estimates that produced
coefficients, and the list of confounders tested.

Write 2 to 4 short caveats (one sentence each) flagging the WEAKEST points a
reader should know about this edge. Focus on: small sample size, disagreement
across methods, regime concerns, absorbed vs unabsorbed confounders, and the
first link problem (the edge is LLM reasoning rather than regression).

Do NOT invent numbers. Only refer to numbers in the provided data.
Do NOT repeat the numerical range or confidence values.
Do NOT generic caveats like "correlation is not causation". Be specific.
"""

CAVEATS_TOOL: dict = {
    "type": "object",
    "properties": {
        "caveats": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string"},
        },
    },
    "required": ["caveats"],
}


def _generate_caveats(
    edge: EdgeObject,
    hypotheses_by_id: dict[str, Hypothesis],
    valid: list[MethodEstimate],
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    rationales = []
    for hid in edge.hypothesis_ids:
        h = hypotheses_by_id.get(hid)
        if h:
            rationales.append(h.economic_rationale)

    method_lines = []
    for m in edge.method_estimates:
        coef = "n/a" if m.coefficient is None else f"{m.coefficient:+.4f}"
        se = "n/a" if m.standard_error is None else f"{m.standard_error:.4f}"
        n = m.sample_size or 0
        r2 = "n/a" if m.r_squared is None else f"{m.r_squared:.2f}"
        method_lines.append(
            f"  - {m.method.value}: coef={coef} SE={se} n={n} R2={r2} passed={m.passed} notes={m.notes}"
        )

    conf_lines = [
        f"  - {c.name}: {c.mechanism} (handling={c.handling}, proxy={c.proxy_variable})"
        for c in edge.confounders_tested
    ]

    first_link_flag = "YES (LLM reasoning based, no direct regression)" if edge.is_first_link else "no"

    user = (
        f"Edge: {edge.source_node} -> {edge.target_node}\n"
        f"Wave: {edge.wave}\n"
        f"First link: {first_link_flag}\n\n"
        f"Method estimates:\n" + "\n".join(method_lines) + "\n\n"
        f"Confounders tested:\n" + ("\n".join(conf_lines) if conf_lines else "  (none)") + "\n\n"
        f"Hypothesis rationales:\n" + "\n\n".join(rationales)
    )

    try:
        data = call_tool(
            system=CAVEATS_SYSTEM,
            cacheable_context=None,
            user=user,
            tool_name="submit_caveats",
            tool_description="Submit 2 to 4 short caveats for this edge.",
            tool_schema=CAVEATS_TOOL,
            model=model,
            run_id=run_id,
            caller=f"synthesizer:{edge.source_node}->{edge.target_node}",
            max_tokens=800,
        )
        return [c.strip() for c in data.get("caveats", []) if c.strip()]
    except Exception as exc:
        return [f"(caveat generation failed: {type(exc).__name__})"]
