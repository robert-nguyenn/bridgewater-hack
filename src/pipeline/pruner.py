"""Edge pruner.

Generates broad, prunes narrow. Runs AFTER empirics and AFTER the
reviewer, removes edges that are:

  - empirically dead        : no MethodEstimate passed, or all coefficients 0
  - low confidence          : overall confidence below threshold
  - duplicates              : same (source, target), keep the one with the
                              highest overall confidence and more passing methods
  - reviewer rejected       : reviewer emitted an error-severity flag AND
                              marked the edge as should_reject
  - adversary rejected      : hypothesis_ids are all in the adversary's
                              reject list

First link edges are treated more leniently because they are LLM / analog
reasoning rather than regression estimated; we do not drop them for low
statistical confidence alone.

Pruned edges are kept in the return value so they remain auditable in
pruned_edges.json on disk.
"""
from __future__ import annotations

from typing import Optional

from src.schemas import EdgeObject, ReviewFlag


DEFAULT_MIN_CONFIDENCE = 0.20
DEFAULT_MIN_CONFIDENCE_FIRST_LINK = 0.10   # first-link edges get a pass


def prune_edges(
    edges: list[EdgeObject],
    review_flags: Optional[list[ReviewFlag]] = None,
    rejected_hypothesis_ids: Optional[set[str]] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_confidence_first_link: float = DEFAULT_MIN_CONFIDENCE_FIRST_LINK,
) -> tuple[list[EdgeObject], list[dict]]:
    """Return (kept_edges, prune_log).

    prune_log entries look like:
        {"edge": "src->tgt", "reasons": ["no_passing_method", "low_confidence"]}
    """
    review_flags = review_flags or []
    rejected = rejected_hypothesis_ids or set()

    # Index review flags per edge target, collect only errors
    flags_by_target: dict[str, list[ReviewFlag]] = {}
    for f in review_flags:
        if f.severity == "error":
            flags_by_target.setdefault(f.target, []).append(f)

    kept: list[EdgeObject] = []
    prune_log: list[dict] = []

    for edge in edges:
        reasons = _reasons_to_drop(
            edge,
            flags_by_target.get(f"{edge.source_node}->{edge.target_node}", []),
            rejected,
            min_confidence,
            min_confidence_first_link,
        )
        if reasons:
            prune_log.append({
                "edge": f"{edge.source_node}->{edge.target_node}",
                "is_first_link": edge.is_first_link,
                "reasons": reasons,
                "overall_confidence": edge.confidence.overall,
                "n_methods": len(edge.method_estimates),
                "hypothesis_ids": edge.hypothesis_ids,
            })
            continue
        kept.append(edge)

    # Second pass: dedupe on (source, target), keep best
    kept = _dedupe_best(kept, prune_log)

    return kept, prune_log


def _reasons_to_drop(
    edge: EdgeObject,
    error_flags: list[ReviewFlag],
    rejected_hyps: set[str],
    min_confidence: float,
    min_confidence_first_link: float,
) -> list[str]:
    reasons: list[str] = []

    # Adversary rejected all supporting hypotheses
    if edge.hypothesis_ids and all(h in rejected_hyps for h in edge.hypothesis_ids):
        reasons.append("adversary_rejected_all_hypotheses")

    # No method passed
    any_passing = any(m.passed for m in edge.method_estimates)
    if not any_passing:
        reasons.append("no_passing_method")
        # Strong enough on its own: no point checking other criteria
        return reasons

    # All coefficients 0: empirically dead even if "passed"
    non_zero = [m for m in edge.method_estimates if m.passed and m.coefficient is not None and m.coefficient != 0.0]
    if not non_zero:
        reasons.append("all_coefficients_zero")

    # Low confidence
    threshold = min_confidence_first_link if edge.is_first_link else min_confidence
    if edge.confidence.overall < threshold:
        reasons.append(f"low_confidence_below_{threshold:.2f}")

    # Reviewer fired sign_mismatch + plot_vs_claim at error severity
    cats = {f.category for f in error_flags}
    if "sign_mismatch" in cats and "plot_vs_claim" in cats:
        reasons.append("reviewer_sign_and_plot_both_wrong")

    # Reviewer fired invented_number AND sign_mismatch
    if "invented_number" in cats and "sign_mismatch" in cats:
        reasons.append("reviewer_invented_and_sign_wrong")

    return reasons


def _dedupe_best(edges: list[EdgeObject], prune_log: list[dict]) -> list[EdgeObject]:
    """For duplicate (source, target) pairs, keep the one with highest overall
    confidence. Ties broken by more passing methods, then by |coef|.
    """
    groups: dict[tuple[str, str], list[EdgeObject]] = {}
    for e in edges:
        groups.setdefault((e.source_node, e.target_node), []).append(e)

    winners: list[EdgeObject] = []
    for key, group in groups.items():
        if len(group) == 1:
            winners.append(group[0])
            continue
        best = max(
            group,
            key=lambda e: (
                e.confidence.overall,
                sum(1 for m in e.method_estimates if m.passed),
                abs(e.elasticity.point or 0.0),
            ),
        )
        winners.append(best)
        for loser in group:
            if loser is best:
                continue
            prune_log.append({
                "edge": f"{loser.source_node}->{loser.target_node}",
                "is_first_link": loser.is_first_link,
                "reasons": ["duplicate_edge_lower_confidence"],
                "overall_confidence": loser.confidence.overall,
                "n_methods": len(loser.method_estimates),
                "hypothesis_ids": loser.hypothesis_ids,
            })
    return winners
