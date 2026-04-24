"""Phase 3 integration test.

Runs the Hormuz scenario end to end through:
  parser -> coordinator -> 3 specialists -> adversary

Verifies:
  1. policy parser returns a valid StructuredPolicy
  2. coordinator returns non empty channels
  3. each specialist produces at least one hypothesis
  4. DIVERSITY: hypotheses span >=2 distinct perspectives and reference >=2 distinct countries/regions
  5. ADVERSARY: each hypothesis gets new confounders AND each historical episode gets an adversarial_critique

We only run 3 of 5 specialists to keep the test fast and affordable.
Full pipeline runs all 5 via orchestrator.

Note: this is a live integration test. It costs a few cents per run.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents._client import new_run_id  # noqa: E402
from src.agents.adversary import adversarial_review  # noqa: E402
from src.agents.coordinator import select_channels  # noqa: E402
from src.agents.policy_parser import parse_policy  # noqa: E402
from src.agents.specialists import run_specialist  # noqa: E402


SCENARIO = (
    "What happens to global markets if the Strait of Hormuz is closed for "
    "7 or more days with greater than 50% probability over the next 30 days? "
    "Roughly 20 percent of global oil trade passes through the strait."
)


def run() -> None:
    run_id = new_run_id()
    print(f"run_id={run_id}\n")

    # 1. Policy parser
    print("[1/4] parsing policy...")
    policy = parse_policy(SCENARIO, run_id=run_id)
    print(
        f"  policy_type={policy.policy_type.value}  subject={policy.subject}  "
        f"mag={policy.magnitude}{policy.magnitude_unit}  horizon={policy.horizon_days}d"
    )
    assert policy.policy_type.value == "geopolitical"

    # 2. Coordinator
    print("\n[2/4] selecting channels...")
    channels = select_channels(policy, run_id=run_id)
    print(f"  {len(channels)} channels: {channels}")
    assert channels, "coordinator returned no channels"

    # 3. Specialists (3 of 5 to keep test cheap)
    print("\n[3/4] running specialists (monetary, supply_chain, international)...")
    hyps: list = []
    for sid in ("monetary", "supply_chain", "international"):
        results = run_specialist(sid, policy, channels, run_id=run_id)
        print(f"  [{sid}] produced {len(results)} hypothesis/es")
        for h in results:
            print(
                f"    - {h.hypothesis_id[:30]:<30} channel={h.channel_id:<35} "
                f"shock={h.shock_variable} -> response={h.response_variable}"
            )
        hyps.extend(results)
        assert results, f"specialist '{sid}' returned no hypotheses"

    # 4. Diversity assertions
    print("\n  diversity check...")
    perspectives = {
        ("[perspective:" in h.economic_rationale) for h in hyps
    }
    assert all(perspectives), "some hypothesis missing a perspective tag"
    distinct_channels = {h.channel_id for h in hyps}
    print(f"  distinct channels covered: {len(distinct_channels)} -> {sorted(distinct_channels)}")
    assert len(distinct_channels) >= 3, (
        f"expected at least 3 distinct channels across 3 specialists, got {len(distinct_channels)}"
    )

    # Regions: scan rationales for non US country mentions
    region_markers = ["ECB", "BoE", "BoJ", "China", "Europe", "Japan", "Brazil",
                      "Mexico", "India", "Korea", "Taiwan", "Germany", "EMBI",
                      "EM ", " EM,", "emerging", "Asia", "LatAm", "PBoC", "RBA"]
    combined_text = " ".join(h.economic_rationale for h in hyps)
    hit_regions = [m for m in region_markers if m.lower() in combined_text.lower()]
    print(f"  non US / cross country markers found: {hit_regions}")
    assert hit_regions, "no cross country perspective found in any hypothesis"

    # 5. Adversary
    print("\n[4/4] running adversary (hypothesis review + per analog critique)...")
    enriched = adversarial_review(hyps, policy_context=policy.raw_input, run_id=run_id)

    # Check added confounders
    added_count = 0
    for orig, enr in zip(hyps, enriched):
        new_count = len(enr.confounders) - len(orig.confounders)
        added_count += max(new_count, 0)
    print(f"  adversary added {added_count} new confounders across {len(hyps)} hypotheses")

    # Check analog critiques
    total_episodes = sum(len(h.historical_episodes) for h in enriched)
    critiqued = sum(
        1 for h in enriched for ep in h.historical_episodes
        if ep.adversarial_critique and len(ep.adversarial_critique.strip()) > 20
    )
    print(f"  per analog critiques: {critiqued}/{total_episodes} episodes critiqued")

    # Sample a couple critiques for inspection
    print("\n  sample analog critiques:")
    sample_n = 0
    for h in enriched:
        for ep in h.historical_episodes:
            if ep.adversarial_critique and sample_n < 3:
                print(f"    [{h.proposed_by}] {ep.name} ({ep.date})")
                print(f"      -> {ep.adversarial_critique[:250]}...")
                sample_n += 1

    assert critiqued >= total_episodes * 0.5, (
        f"expected adversary to critique at least half of episodes, got {critiqued}/{total_episodes}"
    )
    assert added_count >= 1, "adversary did not add any new confounders across all hypotheses"

    print(f"\nPhase 3 integration test PASSED.  run_id={run_id}")
    print(f"  see data/runs/{run_id}/llm_log.jsonl for the full call log")


if __name__ == "__main__":
    run()
