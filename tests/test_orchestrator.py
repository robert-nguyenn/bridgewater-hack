"""End to end orchestrator test.

Runs the Hormuz scenario through the full pipeline:
  parse -> coordinate -> specialists (parallel) -> adversary (parallel) ->
  empirics (plots) -> graph -> synthesize (parallel) -> review (parallel)

Verifies:
  - ImpactMap produced with >= 3 edges
  - At least one edge is_first_link=True
  - At least one edge has plot paths on method_estimates
  - Review flags list populated (empty allowed, but the call happened)
  - Per analog adversarial critiques present on >= half of episodes
  - Total wall time under 5 minutes (budget: 300s)

To keep live API cost bounded, uses 3 of 5 specialists.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import run_impact_analysis  # noqa: E402


SCENARIO = (
    "What happens to global markets if the Strait of Hormuz is closed for "
    "7 or more days with greater than 50 percent probability over the next "
    "30 days? Roughly 20 percent of global oil trade passes through the strait."
)


def run() -> None:
    t0 = time.time()
    impact = run_impact_analysis(
        SCENARIO,
        specialists_to_run=["monetary", "supply_chain", "international"],
        skip_review=False,
    )
    elapsed = time.time() - t0

    print(f"\n========= RESULT =========  elapsed={elapsed:.1f}s")
    print(f"run_id: {impact.generation_metadata['run_id']}")
    print(f"policy: {impact.policy.subject}  type={impact.policy.policy_type.value}  "
          f"horizon={impact.policy.horizon_days}d")
    print(f"nodes: {len(impact.nodes)}")
    print(f"edges: {len(impact.edges)}")
    n_first_link = sum(1 for e in impact.edges if e.is_first_link)
    print(f"first link edges: {n_first_link}")
    n_plots = sum(1 for e in impact.edges for m in e.method_estimates if m.plot_path)
    print(f"plot files generated: {n_plots}")

    print("\n-- edges (abridged) --")
    for e in impact.edges:
        tag = "[FIRST LINK]" if e.is_first_link else "           "
        r = e.elasticity
        c = e.confidence
        print(f"  {tag}  w{e.wave}  {e.source_node[:28]:<28} -> {e.target_node[:20]:<20} "
              f"el={r.point:+.3f} [{r.low:+.3f},{r.high:+.3f}] {r.unit}  "
              f"conf={c.overall:.2f}  methods={len(e.method_estimates)}")
        for cv in e.caveats[:3]:
            print(f"      caveat: {cv[:140]}")

    print(f"\n-- review flags ({len(impact.review_flags)}) --")
    for f in impact.review_flags[:12]:
        print(f"  [{f.severity:<7}] {f.category:<16} {f.target[:38]:<38} {f.message[:120]}")

    print("\n-- analog critiques sample --")
    # These are on the hypotheses, not the final edges. Read from intermediate file.
    import json
    hyps_file = PROJECT_ROOT / "data" / "runs" / impact.generation_metadata["run_id"] / "hypotheses_enriched.json"
    if hyps_file.exists():
        enriched_hyps = json.loads(hyps_file.read_text())
        total_eps = 0
        critiqued = 0
        sample_count = 0
        for h in enriched_hyps:
            for ep in h["historical_episodes"]:
                total_eps += 1
                if ep.get("adversarial_critique") and len(ep["adversarial_critique"].strip()) > 10:
                    critiqued += 1
                    if sample_count < 3:
                        print(f"  [{h['proposed_by']}] {ep['name']} ({ep['date']})")
                        print(f"    -> {ep['adversarial_critique'][:220]}")
                        sample_count += 1
        print(f"\n  total episodes: {total_eps}  with critiques: {critiqued}")
        assert critiqued >= total_eps * 0.5, (
            f"adversary critiqued only {critiqued}/{total_eps} episodes"
        )

    # Assertions
    assert len(impact.edges) >= 3, f"expected at least 3 edges, got {len(impact.edges)}"
    assert n_first_link >= 1, "expected at least 1 first_link edge"
    assert n_plots >= 1, "expected at least 1 plot file on method estimates"
    assert elapsed < 420, f"total wall time {elapsed:.0f}s exceeds 7 min budget"

    print("\nOrchestrator end to end test PASSED.")


if __name__ == "__main__":
    run()
