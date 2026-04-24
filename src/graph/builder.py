"""Graph builder.

Takes the full list of hypotheses and their method estimates, groups
them by (source, target), and produces deduped NodeObjects and
EdgeObjects. The synthesizer is responsible for rolling up multiple
method estimates into a single EstimateRange and ConfidenceBreakdown
on each edge. This module only assembles the structure.

First link detection:
  An edge is `is_first_link` if its source is the scenario itself, or
  if the shock variable does not resolve to a Tier 1 data series. Those
  edges are LLM / analog driven, not empirically estimated, and must
  be treated differently in confidence and UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.schemas import (
    EstimateRange,
    ConfidenceBreakdown,
    EdgeObject,
    Hypothesis,
    MethodEstimate,
    NodeObject,
    StructuredPolicy,
    VariableType,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHANNEL_CATALOG = PROJECT_ROOT / "configs" / "channel_catalog.yaml"


def _load_channel_catalog() -> dict[str, dict]:
    with CHANNEL_CATALOG.open() as fh:
        return {c["id"]: c for c in yaml.safe_load(fh)["channels"]}


def _is_first_link_shock(shock_variable: str) -> bool:
    """An edge is first link when its shock is not loadable from Tier 1 data.

    Practically: scenario specific variables like 'supply_disruption_pct_global',
    'strait_of_hormuz_closure_probability', or 'effective_tariff_rate_china_semis'
    do not resolve, so those edges are marked is_first_link.
    """
    from src.loaders import get_data

    try:
        df = get_data(shock_variable)
        return df is None or df.is_empty()
    except Exception:
        return True


def build_graph(
    policy: StructuredPolicy,
    hypotheses: list[Hypothesis],
    estimates_by_hid: dict[str, list[MethodEstimate]],
) -> tuple[list[NodeObject], list[EdgeObject]]:
    """Assemble nodes and edges from hypotheses and their estimates."""
    catalog = _load_channel_catalog()

    # Group hypotheses by (source_node, target_node) == (shock_variable, response_variable)
    groups: dict[tuple[str, str], list[Hypothesis]] = {}
    for h in hypotheses:
        key = (h.shock_variable, h.response_variable)
        groups.setdefault(key, []).append(h)

    edges: list[EdgeObject] = []
    for (src, tgt), hyps in groups.items():
        # Collect all estimates across hypotheses in the group
        method_ests: list[MethodEstimate] = []
        for h in hyps:
            method_ests.extend(estimates_by_hid.get(h.hypothesis_id, []))

        # Wave from catalog: take the min across hypotheses
        wave_hints: list[int] = []
        for h in hyps:
            ch = catalog.get(h.channel_id)
            if ch and isinstance(ch.get("wave_hint"), int):
                wave_hints.append(int(ch["wave_hint"]))
        wave = min(wave_hints) if wave_hints else 1

        first_link = _is_first_link_shock(src)

        # Typical lag: take median from catalog
        lags: list[int] = []
        for h in hyps:
            ch = catalog.get(h.channel_id)
            if ch and isinstance(ch.get("typical_lag_days"), (int, float)):
                lags.append(int(ch["typical_lag_days"]))
        lag_days = int(sum(lags) / len(lags)) if lags else 1

        # Gather all confounders across group hypotheses, deduped by name
        confounders_tested = []
        seen_cf = set()
        for h in hyps:
            for c in h.confounders:
                if c.name not in seen_cf:
                    confounders_tested.append(c)
                    seen_cf.add(c.name)

        # Placeholder elasticity and confidence. Synthesizer fills these in.
        placeholder_range = EstimateRange(point=0.0, low=0.0, high=0.0, unit="unknown")
        placeholder_conf = ConfidenceBreakdown(
            statistical=0.0, sample=0.0, cross_method=0.0, regime=0.0, overall=0.0,
        )

        edges.append(EdgeObject(
            source_node=src,
            target_node=tgt,
            wave=wave,
            elasticity=placeholder_range,
            confidence=placeholder_conf,
            lag_days=lag_days,
            causal_share=None,
            method_estimates=method_ests,
            confounders_tested=confounders_tested,
            caveats=[],
            hypothesis_ids=[h.hypothesis_id for h in hyps],
            is_first_link=first_link,
        ))

    # Build nodes. Every unique node id gets one NodeObject.
    type_hints = _infer_variable_types(hypotheses)
    all_ids = {e.source_node for e in edges} | {e.target_node for e in edges}
    # Wave assignment: source inherits (edge.wave - 1) min, target inherits edge.wave max
    node_wave: dict[str, int] = {}
    for e in edges:
        node_wave[e.target_node] = max(node_wave.get(e.target_node, 0), e.wave)
        node_wave.setdefault(e.source_node, max(e.wave - 1, 1))

    nodes = [
        NodeObject(
            node_id=nid,
            label=_humanize(nid),
            variable_type=type_hints.get(nid, VariableType.PRICE),
            wave=node_wave.get(nid, 1),
            current_level=None,
            projected_level=None,
            projected_range=None,
            data_source=None,
        )
        for nid in sorted(all_ids)
    ]

    return nodes, edges


def _infer_variable_types(hypotheses: list[Hypothesis]) -> dict[str, VariableType]:
    out: dict[str, VariableType] = {}
    for h in hypotheses:
        out.setdefault(h.shock_variable, h.shock_type)
        out.setdefault(h.response_variable, h.response_type)
    return out


def _humanize(nid: str) -> str:
    return nid.replace("_", " ").title()
