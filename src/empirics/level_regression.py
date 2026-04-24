"""Level regression estimator.

Standard OLS on levels or first differences with optional lags and
covariates. Newey West standard errors when lags > 0, HC3 otherwise.

All inputs are polars [date, value] frames. They are inner joined on
date and then aligned to the response calendar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from src.schemas import EstimatorType, MethodEstimate

from .plotting import plot_level_regression


def _to_indexed(series: pl.DataFrame, name: str) -> pd.Series:
    pdf = series.select(["date", "value"]).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.dropna().sort_values("date").drop_duplicates(subset="date")
    s = pdf.set_index("date")["value"]
    s.name = name
    return s


def run_level_regression(
    response: pl.DataFrame,
    shock: pl.DataFrame,
    covariates: Optional[dict[str, pl.DataFrame]] = None,
    differences: bool = True,
    lags: int = 0,
    plot_stem: Optional[str] = None,
    plot_title: str = "level regression",
    shock_label: str = "d(shock)",
    response_label: str = "d(response)",
    run_id: Optional[str] = None,
) -> MethodEstimate:
    """OLS regression of response on shock (+ covariates).

    Parameters
    ----------
    response      [date, value] outcome series.
    shock         [date, value] regressor of interest.
    covariates    optional name to frame mapping of additional regressors.
    differences   if True, first difference all series before regressing.
    lags          if > 0, include lags of shock up to this many periods and
                  report Newey West SE at lags horizon.
    """
    y = _to_indexed(response, "y")
    x = _to_indexed(shock, "x")
    cov = {k: _to_indexed(v, f"cov_{k}") for k, v in (covariates or {}).items()}

    frames = [y, x] + list(cov.values())
    df = pd.concat(frames, axis=1, join="inner").dropna()

    if df.empty:
        return MethodEstimate(
            method=EstimatorType.LEVEL_REGRESSION,
            coefficient=None,
            standard_error=None,
            sample_size=0,
            r_squared=None,
            passed=False,
            notes="No overlapping dates across response, shock, and covariates.",
        )

    if differences:
        df = df.diff().dropna()

    # Optional lags of shock
    shock_cols = ["x"]
    for lag in range(1, lags + 1):
        col = f"x_lag{lag}"
        df[col] = df["x"].shift(lag)
        shock_cols.append(col)
    df = df.dropna()
    if df.empty:
        return MethodEstimate(
            method=EstimatorType.LEVEL_REGRESSION,
            coefficient=None,
            standard_error=None,
            sample_size=0,
            r_squared=None,
            passed=False,
            notes="Empty sample after differencing and lagging.",
        )

    cov_cols = [c for c in df.columns if c.startswith("cov_")]
    X = df[shock_cols + cov_cols].to_numpy()
    X = sm.add_constant(X, has_constant="add")
    y_arr = df["y"].to_numpy()

    if lags > 0:
        model = sm.OLS(y_arr, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
        se_type = f"Newey-West(maxlags={max(lags,1)})"
    else:
        model = sm.OLS(y_arr, X).fit(cov_type="HC3")
        se_type = "HC3"

    # Sum the shock coefficient + lag coefficients to get the cumulative effect
    # but also report the contemporaneous point estimate as the primary number.
    coef_contemp = float(model.params[1])  # x is first after intercept
    se_contemp = float(model.bse[1])

    cumulative = float(sum(model.params[i] for i in range(1, 1 + len(shock_cols))))

    plot_path = None
    if plot_stem:
        plot_path = plot_level_regression(
            x=df["x"].tolist(),
            y=df["y"].tolist(),
            coef=coef_contemp,
            se=se_contemp,
            r2=float(model.rsquared),
            title=plot_title,
            stem=plot_stem,
            shock_label=shock_label,
            response_label=response_label,
            run_id=run_id,
        )

    return MethodEstimate(
        method=EstimatorType.LEVEL_REGRESSION,
        coefficient=coef_contemp,
        standard_error=se_contemp,
        sample_size=int(model.nobs),
        r_squared=float(model.rsquared),
        passed=True,
        notes=(
            f"OLS {se_type}. differences={differences} lags={lags}. "
            f"n_covariates={len(cov_cols)}. cumulative_effect={cumulative:.4f}"
        ),
        plot_path=plot_path,
    )
