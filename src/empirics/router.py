"""Estimator router.

Takes a Hypothesis and dispatches to the right empirical estimator.
Returns the full list of MethodEstimate outputs, including robustness
specifications.

For EVENT_STUDY and LEVEL_REGRESSION the router runs the base spec and
then repeats the estimation with each listed confounder added as a
covariate, so the caller can see how the point estimate moves when the
confounder is absorbed.
"""
from __future__ import annotations

from datetime import date as date_type
from pathlib import Path

import polars as pl

from src.schemas import EstimatorType, Hypothesis, MethodEstimate

from .event_study import run_event_study
from .level_regression import run_level_regression
from .analog_retrieval import retrieve_analogs
from .plotting import plot_analog_retrieval
from .svar_lookup import lookup_svar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_CATALOG = PROJECT_ROOT / "configs" / "event_catalog.csv"


def _load_series_via_unified(name: str) -> pl.DataFrame | None:
    """Try to resolve a variable via the unified loader. Returns None on failure."""
    from src.loaders import get_data

    try:
        return get_data(name)
    except Exception:
        return None


def _event_dates_for(subject: str | None = None, policy_type: str | None = None) -> list[date_type]:
    df = pl.read_csv(EVENT_CATALOG).with_columns(pl.col("date").str.to_date())
    if subject:
        df = df.filter(pl.col("subject").eq(subject))
    if policy_type:
        df = df.filter(pl.col("policy_type").eq(policy_type))
    return df["date"].to_list()


def estimate_hypothesis(hyp: Hypothesis, run_id: str | None = None) -> list[MethodEstimate]:
    """Route a Hypothesis to its estimator and run base + robustness specs.

    Every estimator also emits a plot, saved under data/runs/<run_id>/plots/
    and referenced on MethodEstimate.plot_path.
    """
    estimates: list[MethodEstimate] = []

    if hyp.estimator == EstimatorType.SVAR_LOOKUP:
        horizon_months = int(hyp.specification_params.get("horizon_months", 12))
        estimates.append(
            lookup_svar(
                shock=hyp.shock_variable,
                response=hyp.response_variable,
                horizon_months=horizon_months,
                plot_stem=f"{hyp.hypothesis_id}_svar",
                run_id=run_id,
            )
        )
        return estimates

    # All others need actual data
    resp = _load_series_via_unified(hyp.response_variable)
    shock = _load_series_via_unified(hyp.shock_variable)

    if resp is None:
        estimates.append(MethodEstimate(
            method=hyp.estimator,
            coefficient=None, standard_error=None, sample_size=0,
            r_squared=None, passed=False,
            notes=f"Could not load response '{hyp.response_variable}' from any tier.",
        ))
        return estimates

    covariate_frames: dict[str, pl.DataFrame] = {}
    for c in hyp.covariates:
        df = _load_series_via_unified(c)
        if df is not None:
            covariate_frames[c] = df

    if hyp.estimator == EstimatorType.EVENT_STUDY:
        if hyp.historical_episodes:
            event_dates = [ep.date for ep in hyp.historical_episodes]
        else:
            event_dates = _event_dates_for()

        title = f"{hyp.shock_variable} -> {hyp.response_variable}"
        base = run_event_study(
            response=resp,
            event_dates=event_dates,
            shock=shock,
            window_pre=int(hyp.specification_params.get("window_pre", 1)),
            window_post=int(hyp.specification_params.get("window_post", 1)),
            covariates=covariate_frames or None,
            plot_stem=f"{hyp.hypothesis_id}_event_study",
            plot_title=title,
            shock_label=f"d({hyp.shock_variable})",
            response_label=f"d({hyp.response_variable})",
            run_id=run_id,
        )
        estimates.append(base)

        for j, conf in enumerate(hyp.confounders):
            proxy_df = _load_series_via_unified(conf.proxy_variable)
            if proxy_df is None:
                continue
            robust_cov = {**covariate_frames, f"confounder_{conf.name}": proxy_df}
            robust = run_event_study(
                response=resp,
                event_dates=event_dates,
                shock=shock,
                window_pre=int(hyp.specification_params.get("window_pre", 1)),
                window_post=int(hyp.specification_params.get("window_post", 1)),
                covariates=robust_cov,
                plot_stem=f"{hyp.hypothesis_id}_event_study_robust_{j}",
                plot_title=f"{title}  (robust w/ {conf.name})",
                shock_label=f"d({hyp.shock_variable})",
                response_label=f"d({hyp.response_variable})",
                run_id=run_id,
            )
            robust.notes = (
                f"robustness with '{conf.name}': " + (robust.notes or "")
            ).strip()
            estimates.append(robust)

    elif hyp.estimator == EstimatorType.LEVEL_REGRESSION:
        if shock is None:
            estimates.append(MethodEstimate(
                method=EstimatorType.LEVEL_REGRESSION,
                coefficient=None, standard_error=None, sample_size=0,
                r_squared=None, passed=False,
                notes=f"Could not load shock '{hyp.shock_variable}'.",
            ))
            return estimates

        title = f"{hyp.shock_variable} -> {hyp.response_variable}"
        base = run_level_regression(
            response=resp,
            shock=shock,
            covariates=covariate_frames or None,
            differences=bool(hyp.specification_params.get("differences", True)),
            lags=int(hyp.specification_params.get("lags", 0)),
            plot_stem=f"{hyp.hypothesis_id}_level_reg",
            plot_title=title,
            shock_label=f"d({hyp.shock_variable})",
            response_label=f"d({hyp.response_variable})",
            run_id=run_id,
        )
        estimates.append(base)

        for j, conf in enumerate(hyp.confounders):
            proxy_df = _load_series_via_unified(conf.proxy_variable)
            if proxy_df is None:
                continue
            robust_cov = {**covariate_frames, f"confounder_{conf.name}": proxy_df}
            robust = run_level_regression(
                response=resp,
                shock=shock,
                covariates=robust_cov,
                differences=bool(hyp.specification_params.get("differences", True)),
                lags=int(hyp.specification_params.get("lags", 0)),
                plot_stem=f"{hyp.hypothesis_id}_level_reg_robust_{j}",
                plot_title=f"{title}  (robust w/ {conf.name})",
                shock_label=f"d({hyp.shock_variable})",
                response_label=f"d({hyp.response_variable})",
                run_id=run_id,
            )
            robust.notes = (
                f"robustness with '{conf.name}': " + (robust.notes or "")
            ).strip()
            estimates.append(robust)

    elif hyp.estimator == EstimatorType.ANALOG_RETRIEVAL:
        # Analog retrieval does not produce a scalar coefficient. We flatten
        # into one MethodEstimate with n analogs = sample_size and mean
        # similarity as coefficient so the EdgeObject builder can still
        # consume it.
        # For full results, the pipeline also calls retrieve_analogs directly.
        from src.schemas import PolicyType, StructuredPolicy

        fake_policy = StructuredPolicy(
            raw_input=hyp.economic_rationale,
            policy_type=PolicyType.GEOPOLITICAL,
            subject=hyp.shock_variable,
            magnitude=1.0,
            magnitude_unit="unit",
            direction="positive",
            horizon_days=30,
        )
        analogs = retrieve_analogs(fake_policy, response_series=resp, k=5)
        if not analogs:
            estimates.append(MethodEstimate(
                method=EstimatorType.ANALOG_RETRIEVAL,
                coefficient=None, standard_error=None, sample_size=0,
                r_squared=None, passed=False,
                notes="No analogs found in catalog.",
            ))
        else:
            mean_sim = sum(a["similarity"] for a in analogs) / len(analogs)
            mean_30d = _mean_opt([a["response_30d_pct"] for a in analogs])
            plot_path = plot_analog_retrieval(
                analogs=analogs,
                title=f"analogs for {hyp.shock_variable} -> {hyp.response_variable}",
                stem=f"{hyp.hypothesis_id}_analogs",
                response_label=f"realized {hyp.response_variable} 30d (%)",
                run_id=run_id,
            )
            estimates.append(MethodEstimate(
                method=EstimatorType.ANALOG_RETRIEVAL,
                coefficient=mean_30d,
                standard_error=None,
                sample_size=len(analogs),
                r_squared=None,
                passed=True,
                notes=(
                    f"k={len(analogs)} analogs, mean_sim={mean_sim:.2f}. "
                    f"Coefficient is mean 30d pct change across analogs."
                ),
                plot_path=plot_path,
            ))

    elif hyp.estimator == EstimatorType.CROSS_SECTION:
        estimates.append(MethodEstimate(
            method=EstimatorType.CROSS_SECTION,
            coefficient=None, standard_error=None, sample_size=0,
            r_squared=None, passed=False,
            notes="Cross section estimator not implemented in MVP Phase 2.",
        ))

    elif hyp.estimator == EstimatorType.KALSHI_CONDITIONAL:
        estimates.append(MethodEstimate(
            method=EstimatorType.KALSHI_CONDITIONAL,
            coefficient=None, standard_error=None, sample_size=0,
            r_squared=None, passed=False,
            notes="Kalshi conditional is routed through the Kalshi specialist in Phase 3.",
        ))

    return estimates


def _mean_opt(xs: list) -> float | None:
    vs = [x for x in xs if x is not None]
    return sum(vs) / len(vs) if vs else None
