"""Phase 1.6 smoke test.

Constructs hardcoded instances of every schema type with numbers that
come from nowhere in particular. The goal is to prove the pydantic
contract compiles end to end and that hypothesis channel_ids resolve
against configs/channel_catalog.yaml. No empirics, no LLM calls.

Runs as pytest or as a plain script.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import yaml

# Allow running as a script from anywhere under the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import schemas as s  # noqa: E402


def _load_channel_ids() -> set[str]:
    path = PROJECT_ROOT / "configs" / "channel_catalog.yaml"
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return {entry["id"] for entry in data["channels"]}


def build_policy() -> s.StructuredPolicy:
    return s.StructuredPolicy(
        raw_input="Probability that the Strait of Hormuz is closed for >= 7 days exceeds 50% over the next 30 days",
        policy_type=s.PolicyType.GEOPOLITICAL,
        subject="strait_of_hormuz_closure",
        magnitude=0.50,
        magnitude_unit="probability",
        direction="positive",
        horizon_days=30,
        effective_date=date(2026, 4, 23),
        additional_context="Closure would disrupt ~20% of global oil trade",
    )


def build_hypotheses(valid_channels: set[str]) -> list[s.Hypothesis]:
    shared_episodes = [
        s.HistoricalEpisode(
            name="Gulf of Oman tanker attacks",
            date=date(2019, 6, 13),
            magnitude=0.05,
            notes="Limited disruption, oil spiked then faded",
        ),
        s.HistoricalEpisode(
            name="Soleimani airstrike",
            date=date(2020, 1, 3),
            magnitude=0.03,
            notes="Oil spike of roughly 4 percent intraday",
        ),
    ]

    hypotheses = [
        s.Hypothesis(
            hypothesis_id="hyp_hormuz_oil_1",
            proposed_by="supply_chain",
            channel_id="oil_supply_disruption_to_oil",
            shock_variable="supply_disruption_pct_global",
            shock_type=s.VariableType.FUNDAMENTAL,
            shock_source_hints=["event_catalog", "kpler_flows"],
            response_variable="brent_oil",
            response_type=s.VariableType.PRICE,
            response_source_hints=["DCOILBRENTEU", "yfinance:BZ=F"],
            estimator=s.EstimatorType.ANALOG_RETRIEVAL,
            specification_params={"k_analogs": 5, "horizon_days": 30},
            historical_episodes=shared_episodes,
            covariates=["opec_spare_capacity", "global_inventories"],
            confounders=[
                s.Confounder(
                    name="SPR release",
                    mechanism="Government inventory release caps the price response",
                    proxy_variable="spr_drawdown",
                    handling="include_covariate",
                )
            ],
            expected_sign="positive",
            economic_rationale="Physical supply shock to a major chokepoint transmits to benchmark prices.",
            citations=["Kilian 2009 AER", "Caldara Cavallo Iacoviello 2019"],
        ),
        s.Hypothesis(
            hypothesis_id="hyp_hormuz_cpi_2",
            proposed_by="monetary",
            channel_id="oil_to_cpi_energy",
            shock_variable="wti_oil",
            shock_type=s.VariableType.PRICE,
            shock_source_hints=["WTISPLC", "FRED"],
            response_variable="cpi_energy_yoy",
            response_type=s.VariableType.FUNDAMENTAL,
            response_source_hints=["CPIENGSL", "FRED"],
            estimator=s.EstimatorType.LEVEL_REGRESSION,
            specification_params={"differences": True, "lags": 3},
            historical_episodes=shared_episodes,
            covariates=["usd_index", "refining_margin"],
            confounders=[
                s.Confounder(
                    name="Demand driven oil move",
                    mechanism="Oil up on demand reverses typical CPI sign",
                    proxy_variable="global_manufacturing_pmi",
                    handling="sample_restriction",
                )
            ],
            expected_sign="positive",
            economic_rationale="Energy CPI tracks crude prices with a one to three month lag.",
            citations=["Kilian 2009 AER"],
        ),
        s.Hypothesis(
            hypothesis_id="hyp_hormuz_emfx_3",
            proposed_by="international",
            channel_id="commodity_to_em_fx",
            shock_variable="commodity_basket",
            shock_type=s.VariableType.PRICE,
            shock_source_hints=["GSCI", "BCOM"],
            response_variable="em_commodity_fx_index",
            response_type=s.VariableType.PRICE,
            response_source_hints=["yfinance:USDBRL=X", "yfinance:USDCOP=X", "yfinance:USDMXN=X"],
            estimator=s.EstimatorType.LEVEL_REGRESSION,
            specification_params={"differences": True, "lags": 1},
            historical_episodes=shared_episodes,
            covariates=["dxy", "vix"],
            confounders=[
                s.Confounder(
                    name="Global risk appetite",
                    mechanism="Commodities and EM FX both loaded on global risk factor",
                    proxy_variable="vix_change",
                    handling="include_covariate",
                )
            ],
            expected_sign="positive",
            economic_rationale="Commodity exporter FX appreciates when commodity basket rallies.",
            citations=["Chen Rogoff 2003"],
        ),
    ]

    missing = [h.channel_id for h in hypotheses if h.channel_id not in valid_channels]
    assert not missing, f"Unknown channel_ids referenced by hypotheses: {missing}"
    return hypotheses


def build_edges() -> list[s.EdgeObject]:
    return [
        s.EdgeObject(
            source_node="strait_of_hormuz_closure_probability",
            target_node="brent_oil",
            wave=1,
            elasticity=s.EstimateRange(point=12.0, low=6.0, high=25.0, unit="percent_per_unit_prob"),
            confidence=s.ConfidenceBreakdown(
                statistical=0.55, sample=0.40, cross_method=0.60, regime=0.35, overall=0.47
            ),
            lag_days=1,
            causal_share=0.70,
            method_estimates=[
                s.MethodEstimate(
                    method=s.EstimatorType.ANALOG_RETRIEVAL,
                    coefficient=12.0,
                    standard_error=4.0,
                    sample_size=4,
                    r_squared=None,
                    passed=True,
                    notes="Small n of prior episodes, wide uncertainty",
                )
            ],
            confounders_tested=[
                s.Confounder(
                    name="SPR release",
                    mechanism="Government inventory release caps price response",
                    proxy_variable="spr_drawdown",
                    handling="include_covariate",
                )
            ],
            caveats=["Only 2 directly comparable episodes since 1990, regime fit is weak"],
            hypothesis_ids=["hyp_hormuz_oil_1"],
        ),
        s.EdgeObject(
            source_node="brent_oil",
            target_node="cpi_energy_yoy",
            wave=2,
            elasticity=s.EstimateRange(point=0.10, low=0.05, high=0.18, unit="yoy_pp_per_pct_oil"),
            confidence=s.ConfidenceBreakdown(
                statistical=0.75, sample=0.80, cross_method=0.65, regime=0.70, overall=0.73
            ),
            lag_days=30,
            causal_share=0.60,
            method_estimates=[
                s.MethodEstimate(
                    method=s.EstimatorType.LEVEL_REGRESSION,
                    coefficient=0.10,
                    standard_error=0.025,
                    sample_size=300,
                    r_squared=0.42,
                    passed=True,
                    notes=None,
                )
            ],
            confounders_tested=[],
            caveats=["Demand vs supply decomposition not imposed"],
            hypothesis_ids=["hyp_hormuz_cpi_2"],
        ),
        s.EdgeObject(
            source_node="brent_oil",
            target_node="em_commodity_fx_index",
            wave=2,
            elasticity=s.EstimateRange(point=0.50, low=0.30, high=0.75, unit="beta"),
            confidence=s.ConfidenceBreakdown(
                statistical=0.70, sample=0.75, cross_method=0.60, regime=0.65, overall=0.67
            ),
            lag_days=5,
            causal_share=0.55,
            method_estimates=[
                s.MethodEstimate(
                    method=s.EstimatorType.LEVEL_REGRESSION,
                    coefficient=0.55,
                    standard_error=0.10,
                    sample_size=500,
                    r_squared=0.25,
                    passed=True,
                    notes=None,
                )
            ],
            confounders_tested=[],
            caveats=["Pegged EM currencies excluded from sample"],
            hypothesis_ids=["hyp_hormuz_emfx_3"],
        ),
    ]


def build_nodes(edges: list[s.EdgeObject]) -> list[s.NodeObject]:
    node_ids = sorted({e.source_node for e in edges} | {e.target_node for e in edges})
    wave_by_id: dict[str, int] = {}
    for e in edges:
        wave_by_id.setdefault(e.source_node, max(e.wave - 1, 1))
        wave_by_id[e.target_node] = max(wave_by_id.get(e.target_node, 0), e.wave)

    type_by_id = {
        "strait_of_hormuz_closure_probability": s.VariableType.PROBABILITY,
        "brent_oil": s.VariableType.PRICE,
        "cpi_energy_yoy": s.VariableType.FUNDAMENTAL,
        "em_commodity_fx_index": s.VariableType.PRICE,
    }
    return [
        s.NodeObject(
            node_id=nid,
            label=nid.replace("_", " ").title(),
            variable_type=type_by_id.get(nid, s.VariableType.PRICE),
            wave=wave_by_id[nid],
            current_level=None,
            projected_level=None,
            projected_range=None,
            data_source=None,
        )
        for nid in node_ids
    ]


def assemble_map() -> s.ImpactMap:
    valid_channels = _load_channel_ids()
    policy = build_policy()
    hypotheses = build_hypotheses(valid_channels)
    edges = build_edges()
    nodes = build_nodes(edges)
    return s.ImpactMap(
        policy=policy,
        nodes=nodes,
        edges=edges,
        historical_analogs=[
            {"event": "Gulf of Oman tankers 2019", "oil_30d_pct": 4.5, "vix_peak": 18.3},
            {"event": "Soleimani strike 2020", "oil_30d_pct": -3.0, "vix_peak": 18.6},
        ],
        kalshi_signals=[],
        data_availability_report={
            "fred_series_checked": 0,
            "fred_series_available": 0,
            "hf_datasets_checked": 0,
        },
        generation_metadata={
            "run_id": "phase1_smoke_test",
            "schema_version": "0.1",
            "mode": "hardcoded",
        },
    )


def test_phase1_smoke() -> None:
    impact = assemble_map()

    assert len(impact.edges) == 3
    assert all(isinstance(e, s.EdgeObject) for e in impact.edges)
    assert impact.policy.policy_type == s.PolicyType.GEOPOLITICAL

    ids_on_edges = {hid for e in impact.edges for hid in e.hypothesis_ids}
    assert ids_on_edges == {"hyp_hormuz_oil_1", "hyp_hormuz_cpi_2", "hyp_hormuz_emfx_3"}


def print_summary(impact: s.ImpactMap) -> None:
    print(f"\nPolicy:      {impact.policy.subject}  (horizon {impact.policy.horizon_days}d)")
    print(f"             magnitude {impact.policy.magnitude} {impact.policy.magnitude_unit}")
    print(f"Nodes:       {len(impact.nodes)}")
    for n in impact.nodes:
        print(f"  wave {n.wave}  {n.node_id:<45} ({n.variable_type.value})")
    print(f"Edges:       {len(impact.edges)}")
    for e in impact.edges:
        r = e.elasticity
        print(
            f"  {e.source_node:<38} -> {e.target_node:<22} "
            f"el={r.point:>6.3f} [{r.low:>6.3f}, {r.high:>6.3f}] "
            f"conf={e.confidence.overall:.2f} lag={e.lag_days}d"
        )
    print(f"Analogs:     {len(impact.historical_analogs)}")
    print(f"Kalshi sig:  {len(impact.kalshi_signals)}")
    print(f"Run id:      {impact.generation_metadata['run_id']}")


if __name__ == "__main__":
    impact = assemble_map()
    test_phase1_smoke()
    print_summary(impact)
    print("\nPhase 1 smoke test PASSED. Schema contract is stable.")
