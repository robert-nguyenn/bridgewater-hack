"""Event study estimator.

Given a response time series and a list of event dates, build windows
around each event and estimate the average response. If a shock series
is provided, run a Kuttner style regression of response change on shock
change across events. Standard errors are HC3.

Windows are specified in calendar days. The implementation uses the
closest available observation on or before target_date, so that non
trading days map cleanly to the last trading close.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from src.schemas import EstimatorType, MethodEstimate

from .plotting import plot_event_study


def _pl_to_indexed(series: pl.DataFrame) -> pd.Series:
    """Convert a polars [date, value] frame to a pandas Series indexed by date."""
    if "date" not in series.columns or "value" not in series.columns:
        raise ValueError(f"expected [date, value] columns, got {series.columns}")
    pdf = series.select(["date", "value"]).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.dropna().sort_values("date").drop_duplicates(subset="date")
    return pdf.set_index("date")["value"]


def _nearest_on_or_before(s: pd.Series, target: pd.Timestamp) -> Optional[float]:
    """Return the value at the last index on or before `target`, or None."""
    sub = s[s.index <= target]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def _window_change(s: pd.Series, event: pd.Timestamp, pre: int, post: int) -> Optional[float]:
    v_pre = _nearest_on_or_before(s, event - timedelta(days=pre))
    v_post = _nearest_on_or_before(s, event + timedelta(days=post))
    if v_pre is None or v_post is None:
        return None
    return v_post - v_pre


def run_event_study(
    response: pl.DataFrame,
    event_dates: list[date],
    shock: Optional[pl.DataFrame] = None,
    window_pre: int = 5,
    window_post: int = 20,
    covariates: Optional[dict[str, pl.DataFrame]] = None,
    plot_stem: Optional[str] = None,
    plot_title: str = "event study",
    run_id: Optional[str] = None,
    shock_label: str = "shock change",
    response_label: str = "response change",
) -> MethodEstimate:
    """Event study with optional shock regressor.

    Parameters
    ----------
    response       [date, value] series, the outcome.
    event_dates    list of event calendar dates.
    shock          optional [date, value] series. If given, the estimator
                   becomes a Kuttner style regression of response change on
                   shock change across events. If None, it estimates the
                   average treatment effect (mean response change).
    window_pre     calendar days before the event to read the pre window.
    window_post    calendar days after the event to read the post window.
    covariates     optional dict of name to [date, value] frames, each used
                   as an additional regressor on the change around the event.
    """
    resp = _pl_to_indexed(response)
    shock_s = _pl_to_indexed(shock) if shock is not None else None
    cov_series: dict[str, pd.Series] = (
        {k: _pl_to_indexed(v) for k, v in covariates.items()} if covariates else {}
    )

    rows = []
    used_dates: list[date] = []
    for ev in event_dates:
        ts = pd.Timestamp(ev)
        dy = _window_change(resp, ts, window_pre, window_post)
        if dy is None:
            continue
        row: dict = {"dy": dy}
        if shock_s is not None:
            dx = _window_change(shock_s, ts, window_pre, window_post)
            if dx is None:
                continue
            row["dx"] = dx
        for name, s in cov_series.items():
            dc = _window_change(s, ts, window_pre, window_post)
            if dc is None:
                row = None
                break
            row[f"cov_{name}"] = dc
        if row is not None:
            rows.append(row)
            used_dates.append(ev)

    if not rows:
        return MethodEstimate(
            method=EstimatorType.EVENT_STUDY,
            coefficient=None,
            standard_error=None,
            sample_size=0,
            r_squared=None,
            passed=False,
            notes="No usable events after window lookup.",
        )

    df = pd.DataFrame(rows)
    y = df["dy"].to_numpy()

    if "dx" in df.columns:
        # Regress dy on dx (+ covariates), return coefficient on dx
        X_cols = ["dx"] + [c for c in df.columns if c.startswith("cov_")]
        X = df[X_cols].to_numpy()
        X = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
        coef_index = 1
        coef = float(model.params[coef_index])
        se = float(model.bse[coef_index])

        plot_path = None
        if plot_stem:
            plot_path = plot_event_study(
                dx=df["dx"].tolist(),
                dy=df["dy"].tolist(),
                event_dates=used_dates,
                coef=coef,
                se=se,
                title=plot_title,
                stem=plot_stem,
                shock_label=shock_label,
                response_label=response_label,
                run_id=run_id,
            )

        return MethodEstimate(
            method=EstimatorType.EVENT_STUDY,
            coefficient=coef,
            standard_error=se,
            sample_size=int(model.nobs),
            r_squared=float(model.rsquared),
            passed=True,
            notes=(
                f"Kuttner style: dy on dx with {len(X_cols)-1} covariate(s). "
                f"window=[-{window_pre}, +{window_post}] days. HC3 SE."
            ),
            plot_path=plot_path,
        )
    else:
        X = sm.add_constant(np.zeros(len(y)), has_constant="add")
        model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
        coef = float(model.params[0])
        se = float(model.bse[0])

        plot_path = None
        if plot_stem:
            plot_path = plot_event_study(
                dx=[0.0] * len(df),  # no shock regressor, plot vs event index
                dy=df["dy"].tolist(),
                event_dates=used_dates,
                coef=None,
                se=None,
                title=plot_title + " (ATE)",
                stem=plot_stem,
                shock_label="event dummy",
                response_label=response_label,
                run_id=run_id,
            )

        return MethodEstimate(
            method=EstimatorType.EVENT_STUDY,
            coefficient=coef,
            standard_error=se,
            sample_size=int(model.nobs),
            r_squared=None,
            passed=True,
            notes=(
                f"ATE: mean response change around events. "
                f"window=[-{window_pre}, +{window_post}] days. HC3 SE."
            ),
            plot_path=plot_path,
        )
