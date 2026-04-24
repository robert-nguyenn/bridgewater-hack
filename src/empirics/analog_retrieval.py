"""Analog retrieval.

Given a StructuredPolicy and the current macro state, find k nearest
historical analogs from the event catalog. For each analog, report what
the realized response was over 30, 90, 180 days after the event.

Feature vector:
    - one hot policy type (5 dims)
    - normalized magnitude (1 dim, z scored within magnitude_unit)
    - macro state at event date (unrate, cpi yoy, fed funds, 10y yield,
      equity 6m return, dxy change 6m)

Similarity: cosine over standardized features.

Response readout: for each analog, pull the response series from the
unified loader and compute pct or pp change at 30 60 180 days.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from src.schemas import PolicyType, StructuredPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_CATALOG = PROJECT_ROOT / "configs" / "event_catalog.csv"

POLICY_TYPE_ORDER = [pt.value for pt in PolicyType]

# Macro state features pulled from FRED Tier 1
MACRO_FEATURES = {
    "unrate":   "UNRATE",
    "cpi_yoy":  "CPIAUCSL",     # compute yoy inside
    "fed_funds": "DFF",
    "dgs10":    "DGS10",
    "dxy":      "DTWEXBGS",
}


def _load_catalog() -> pd.DataFrame:
    df = pl.read_csv(EVENT_CATALOG).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _macro_series() -> dict[str, pd.Series]:
    """Load FRED series needed for macro state features."""
    from src.loaders.core import load_series

    out: dict[str, pd.Series] = {}
    for feat_name, series_id in MACRO_FEATURES.items():
        try:
            pdf = load_series(series_id).to_pandas()
            pdf["date"] = pd.to_datetime(pdf["date"])
            s = pdf.set_index("date")["value"]
            out[feat_name] = s
        except FileNotFoundError:
            continue
    return out


def _macro_state(series: dict[str, pd.Series], at: pd.Timestamp) -> dict[str, float]:
    """Read each macro series at `at`. CPI is transformed to YoY."""
    state: dict[str, float] = {}
    for feat_name, s in series.items():
        if feat_name == "cpi_yoy":
            yoy = s.pct_change(12).dropna()
            v = yoy[yoy.index <= at]
            state[feat_name] = float(v.iloc[-1]) if not v.empty else np.nan
        else:
            v = s[s.index <= at]
            state[feat_name] = float(v.iloc[-1]) if not v.empty else np.nan
    return state


def _feature_vector(
    policy_type: str,
    magnitude: float,
    macro_state: dict[str, float],
    feat_names: list[str],
) -> np.ndarray:
    pt_vec = [1.0 if policy_type == pt else 0.0 for pt in POLICY_TYPE_ORDER]
    mac_vec = [macro_state.get(f, 0.0) for f in feat_names]
    return np.array(pt_vec + [float(magnitude)] + mac_vec, dtype=float)


def _zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd[sd == 0] = 1.0
    out = (mat - mu) / sd
    return np.nan_to_num(out)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve_analogs(
    policy: StructuredPolicy,
    response_series: Optional[pl.DataFrame] = None,
    k: int = 5,
) -> list[dict]:
    """Find k nearest historical analogs from the event catalog.

    Parameters
    ----------
    policy           the current scenario to match
    response_series  optional [date, value] frame to compute realized responses
    k                number of analogs to return

    Returns
    -------
    list of dicts, each containing
        event_name, date, policy_type, magnitude, similarity,
        macro_state_at_event, response_30d, response_90d, response_180d
    """
    catalog = _load_catalog()
    macro = _macro_series()
    feat_names = list(MACRO_FEATURES.keys())

    # Feature matrix for every event in catalog
    rows = []
    for _, row in catalog.iterrows():
        state = _macro_state(macro, row["date"])
        rows.append(
            {
                "name": row["description"],
                "date": row["date"],
                "policy_type": row["policy_type"],
                "subject": row["subject"],
                "magnitude": float(row["magnitude"]),
                "state": state,
            }
        )
    if not rows:
        return []

    mat = np.vstack([
        _feature_vector(r["policy_type"], r["magnitude"], r["state"], feat_names)
        for r in rows
    ])
    mat_z = _zscore(mat)

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    query_state = _macro_state(macro, today)
    query_vec = _feature_vector(policy.policy_type.value, policy.magnitude, query_state, feat_names)
    # Standardize query using the same mean and sd as the matrix
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd[sd == 0] = 1.0
    query_z = np.nan_to_num((query_vec - mu) / sd)

    sims = np.array([_cosine(query_z, mat_z[i]) for i in range(mat_z.shape[0])])
    order = np.argsort(-sims)[:k]

    # Response readouts
    resp_s = None
    if response_series is not None:
        resp_pdf = response_series.select(["date", "value"]).to_pandas()
        resp_pdf["date"] = pd.to_datetime(resp_pdf["date"])
        resp_s = resp_pdf.dropna().sort_values("date").set_index("date")["value"]

    def _pct_change_at(base_date: pd.Timestamp, horizon_days: int) -> Optional[float]:
        if resp_s is None:
            return None
        base_slice = resp_s[resp_s.index <= base_date]
        post_slice = resp_s[resp_s.index <= base_date + pd.Timedelta(days=horizon_days)]
        if base_slice.empty or post_slice.empty:
            return None
        v0 = base_slice.iloc[-1]
        v1 = post_slice.iloc[-1]
        if v0 == 0:
            return float(v1 - v0)
        return float((v1 - v0) / abs(v0) * 100.0)

    out = []
    for idx in order:
        r = rows[idx]
        out.append({
            "event_name": r["name"],
            "event_date": r["date"].date(),
            "policy_type": r["policy_type"],
            "subject": r["subject"],
            "magnitude": r["magnitude"],
            "similarity": float(sims[idx]),
            "macro_state_at_event": r["state"],
            "macro_state_today": query_state,
            "response_30d_pct": _pct_change_at(r["date"], 30),
            "response_90d_pct": _pct_change_at(r["date"], 90),
            "response_180d_pct": _pct_change_at(r["date"], 180),
        })
    return out
