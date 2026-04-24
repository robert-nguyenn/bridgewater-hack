"""Top level orchestrator.

One entry point: run_impact_analysis(raw_policy) returns a full
ImpactMap. Every step is timed and logged. All LLM calls emit JSONL
records to data/runs/<run_id>/llm_log.jsonl. All plots save to
data/runs/<run_id>/plots/.

Flow:
  1. parse_policy (Opus)
  2. select_channels (rule based, falls back to LLM only on empty)
  3. run specialists in parallel (Opus via asyncio.gather)
  4. adversary in parallel: hypothesis review + per analog critique
  5. empirics router for every hypothesis (runs deterministic, writes plots)
  6. build graph, dedupe into edges
  7. synthesize each edge (Opus caveats)
  8. review every edge with vision (Opus)
  9. analog retrieval for the overall scenario
  10. assemble ImpactMap with review flags, save to disk

Errors in any one step are logged and the pipeline continues. A failed
specialist returns []; a failed edge review returns a warning flag.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.agents._client import DEFAULT_MODEL, new_run_id
from src.agents.adversary import adversarial_review_async
from src.agents.coordinator import select_channels
from src.agents.event_researcher import research_events_async
from src.agents.policy_parser import parse_policy
from src.agents.reviewer import review_all_edges_async
from src.agents.specialists import SPECIALIST_IDS, run_all_specialists_parallel
from src.agents.synthesizer import synthesize_edge
from src.empirics.analog_retrieval import retrieve_analogs
from src.empirics.router import estimate_hypothesis
from src.graph.builder import build_graph
from src.loaders import get_data
from src.pipeline.pruner import prune_edges
from src.schemas import Hypothesis, ImpactMap, StructuredPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _save_intermediate(run_id: str, name: str, obj) -> None:
    """Persist any intermediate artifact to data/runs/<run_id>/<name>.json."""
    out = PROJECT_ROOT / "data" / "runs" / run_id / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    elif isinstance(obj, list) and obj and hasattr(obj[0], "model_dump"):
        data = [o.model_dump() for o in obj]
    else:
        data = obj
    out.write_text(json.dumps(data, indent=2, default=str))


def _log_event(run_id: str, event: dict) -> None:
    """Write a free form event line to data/runs/<run_id>/pipeline_log.jsonl."""
    out = PROJECT_ROOT / "data" / "runs" / run_id / "pipeline_log.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with out.open("a") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


def _step(run_id: str, name: str, **extras) -> None:
    print(f"  [{name}] " + "  ".join(f"{k}={v}" for k, v in extras.items()))
    _log_event(run_id, {"step": name, **extras})


async def run_impact_analysis_async(
    raw_policy: str,
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    specialists_to_run: Optional[list[str]] = None,
    skip_review: bool = False,
) -> ImpactMap:
    run_id = run_id or new_run_id()
    print(f"run_id={run_id}")
    t_start = time.time()

    # -------- Step 1: parse --------
    t0 = time.time()
    policy = parse_policy(raw_policy, run_id=run_id)
    _step(run_id, "policy_parser",
          elapsed_s=round(time.time() - t0, 1),
          policy_type=policy.policy_type.value,
          subject=policy.subject)
    _save_intermediate(run_id, "policy", policy)

    # -------- Step 2: select channels --------
    t0 = time.time()
    channels = select_channels(policy, run_id=run_id)
    _step(run_id, "coordinator",
          elapsed_s=round(time.time() - t0, 1),
          n_channels=len(channels))
    _save_intermediate(run_id, "channels", {"channel_ids": channels})

    # -------- Step 2.5: event researcher (web search) --------
    # Runs concurrently with specialists. Both take similar time but the
    # researcher output is only needed downstream (analog retrieval), so we
    # kick it off in parallel and await later.
    t_research = time.time()
    events_task = asyncio.create_task(
        research_events_async(policy, run_id=run_id, model=model)
    )

    # -------- Step 3: specialists in parallel --------
    t0 = time.time()
    specialists_to_run = specialists_to_run or SPECIALIST_IDS
    by_specialist = await run_all_specialists_parallel(
        policy, channels, run_id=run_id, model=model,
        specialist_ids=specialists_to_run,
    )
    all_hyps: list[Hypothesis] = [h for hs in by_specialist.values() for h in hs]
    _step(run_id, "specialists_parallel",
          elapsed_s=round(time.time() - t0, 1),
          n_specialists=len(specialists_to_run),
          n_hypotheses=len(all_hyps),
          per_specialist={k: len(v) for k, v in by_specialist.items()})
    _save_intermediate(run_id, "hypotheses_raw", all_hyps)

    if not all_hyps:
        raise RuntimeError("All specialists returned no hypotheses. Aborting.")

    # -------- Step 4: adversary in parallel --------
    t0 = time.time()
    enriched, rejected_map = await adversarial_review_async(
        all_hyps, policy_context=policy.raw_input, run_id=run_id
    )
    n_critiques = sum(
        1 for h in enriched for ep in h.historical_episodes
        if ep.adversarial_critique and len(ep.adversarial_critique.strip()) > 10
    )
    n_total_eps = sum(len(h.historical_episodes) for h in enriched)
    _step(run_id, "adversary",
          elapsed_s=round(time.time() - t0, 1),
          per_analog_critiques=f"{n_critiques}/{n_total_eps}",
          n_rejected=len(rejected_map),
          new_confounders_total=sum(len(h.confounders) for h in enriched) -
                                 sum(len(h.confounders) for h in all_hyps))
    _save_intermediate(run_id, "hypotheses_enriched", enriched)
    if rejected_map:
        _save_intermediate(run_id, "hypotheses_rejected", rejected_map)

    # Filter rejected hypotheses BEFORE empirics so we don't waste estimator
    # and plot cycles on them. Rejected still live in hypotheses_enriched.json
    # and hypotheses_rejected.json for audit.
    surviving = [h for h in enriched if h.hypothesis_id not in rejected_map]
    if len(surviving) != len(enriched):
        _step(run_id, "adversary_filter",
              survived=len(surviving), rejected=len(enriched) - len(surviving))
    enriched = surviving
    hypotheses_by_id = {h.hypothesis_id: h for h in enriched}

    # -------- Step 5: empirics router (deterministic, off the event loop) --------
    t0 = time.time()
    # run_in_executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()

    def _estimate_all_sync() -> dict[str, list]:
        out: dict[str, list] = {}
        for h in enriched:
            try:
                out[h.hypothesis_id] = estimate_hypothesis(h, run_id=run_id)
            except Exception as exc:
                print(f"    [empirics] FAILED for {h.hypothesis_id}: {type(exc).__name__}: {exc}")
                out[h.hypothesis_id] = []
        return out

    estimates_by_hid = await loop.run_in_executor(None, _estimate_all_sync)
    n_plots = sum(1 for ms in estimates_by_hid.values() for m in ms if m.plot_path)
    _step(run_id, "empirics",
          elapsed_s=round(time.time() - t0, 1),
          n_hypotheses_estimated=len(estimates_by_hid),
          n_plots=n_plots,
          n_failed=sum(1 for k, v in estimates_by_hid.items() if not v))

    # -------- Step 6: graph builder --------
    t0 = time.time()
    nodes, edges = build_graph(policy, enriched, estimates_by_hid)
    _step(run_id, "graph_builder",
          elapsed_s=round(time.time() - t0, 1),
          n_nodes=len(nodes), n_edges=len(edges),
          n_first_links=sum(1 for e in edges if e.is_first_link))

    # -------- Step 7: synthesize edges --------
    t0 = time.time()
    # Synthesizer makes one LLM call per edge for caveats. Run in parallel.
    def _synth_one_sync(edge):
        return synthesize_edge(edge, hypotheses_by_id, run_id=run_id, model=model)

    synthesized = await asyncio.gather(*(
        loop.run_in_executor(None, _synth_one_sync, e) for e in edges
    ))
    edges = list(synthesized)
    _step(run_id, "synthesizer",
          elapsed_s=round(time.time() - t0, 1),
          n_edges=len(edges))

    # -------- Step 8: reviewer --------
    review_flags = []
    if not skip_review:
        t0 = time.time()
        review_flags = await review_all_edges_async(
            edges, hypotheses_by_id, run_id=run_id, model=model,
        )
        errors = sum(1 for f in review_flags if f.severity == "error")
        warnings = sum(1 for f in review_flags if f.severity == "warning")
        _step(run_id, "reviewer",
              elapsed_s=round(time.time() - t0, 1),
              total_flags=len(review_flags), errors=errors, warnings=warnings)

    # -------- Step 8.5: prune dead and duplicate edges --------
    t0 = time.time()
    pre_prune_count = len(edges)
    edges, prune_log = prune_edges(
        edges,
        review_flags=review_flags,
        rejected_hypothesis_ids=set(rejected_map.keys()),
    )
    _save_intermediate(run_id, "pruned_edges", prune_log)
    _step(run_id, "pruner",
          elapsed_s=round(time.time() - t0, 1),
          pruned=pre_prune_count - len(edges),
          kept=len(edges))

    # -------- Step 9: analog retrieval, conditioned on researcher events --------
    # Await the researcher task (may still be running).
    try:
        researched_events = await events_task
    except Exception as exc:
        print(f"    [event_researcher] raised: {type(exc).__name__}: {exc}")
        researched_events = []
    _step(run_id, "event_researcher",
          elapsed_s=round(time.time() - t_research, 1),
          n_events=len(researched_events),
          source="web_search" if researched_events else "static_catalog_fallback")
    if researched_events:
        _save_intermediate(run_id, "researched_events", researched_events)

    t0 = time.time()
    try:
        brent = get_data("DCOILBRENTEU")
        scenario_analogs = retrieve_analogs(
            policy,
            response_series=brent,
            k=5,
            events=researched_events or None,   # None -> fall back to static catalog
        )
    except Exception as exc:
        print(f"    [analog_retrieval] degraded: {type(exc).__name__}: {exc}")
        scenario_analogs = []
    _step(run_id, "analog_retrieval",
          elapsed_s=round(time.time() - t0, 1),
          n_analogs=len(scenario_analogs),
          event_pool=len(researched_events) if researched_events else 0)

    # -------- Step 10: assemble --------
    data_availability = _build_data_availability_report(enriched, estimates_by_hid)

    impact = ImpactMap(
        policy=policy,
        nodes=nodes,
        edges=edges,
        historical_analogs=scenario_analogs,
        kalshi_signals=[],
        data_availability_report=data_availability,
        generation_metadata={
            "run_id": run_id,
            "model": model,
            "schema_version": "0.2",
            "elapsed_total_s": round(time.time() - t_start, 1),
            "n_specialists": len(specialists_to_run),
            "ts_start": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
            "ts_end": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        review_flags=review_flags,
    )
    _save_intermediate(run_id, "impact_map", impact)
    _step(run_id, "done",
          total_elapsed_s=round(time.time() - t_start, 1),
          n_edges=len(edges),
          n_review_flags=len(review_flags))
    return impact


def run_impact_analysis(
    raw_policy: str,
    run_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    specialists_to_run: Optional[list[str]] = None,
    skip_review: bool = False,
) -> ImpactMap:
    """Sync wrapper around the async orchestrator."""
    return asyncio.run(run_impact_analysis_async(
        raw_policy, run_id=run_id, model=model,
        specialists_to_run=specialists_to_run, skip_review=skip_review,
    ))


def _build_data_availability_report(
    hypotheses: list[Hypothesis],
    estimates_by_hid: dict[str, list],
) -> dict:
    n_hyps = len(hypotheses)
    n_with_estimate = sum(1 for hid, ms in estimates_by_hid.items() if any(m.passed for m in ms))
    n_with_plot = sum(1 for hid, ms in estimates_by_hid.items() if any(m.plot_path for m in ms))
    return {
        "n_hypotheses": n_hyps,
        "n_with_passing_estimate": n_with_estimate,
        "n_with_plot": n_with_plot,
    }
