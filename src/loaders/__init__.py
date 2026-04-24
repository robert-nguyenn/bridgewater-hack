"""Unified loader interface.

Resolves a variable name to data by trying Tier 1, Tier 2, Tier 3 in
order. Empirics code calls get_data and does not care where the bytes
came from.

See core.py for Tier 1 FRED and HF pre pulls, fallback.py for Tier 3
live calls, extended.py for Tier 2 lazy lookups.
"""
from __future__ import annotations

from typing import Optional, Sequence

import polars as pl

from .core import load_series as _load_fred_cached, load_hf_file as _load_hf_cached, HF_FILES


# Named alias map, for friendly variable names that downstream code uses.
# Left side is what an empirics hypothesis might ask for, right side is
# a FRED series id or an HF cache path.
ALIASES: dict[str, str] = {
    # Rates and curve
    "fed_funds_rate":      "FEDFUNDS",
    "fed_funds_target":    "DFF",
    "dff":                 "DFF",
    "sofr":                "SOFR",
    "dgs2":                "DGS2",
    "dgs5":                "DGS5",
    "dgs10":               "DGS10",
    "dgs30":               "DGS30",
    "treasury_2y_yield":   "DGS2",
    "treasury_10y_yield":  "DGS10",
    # Inflation
    "cpi_headline":        "CPIAUCSL",
    "cpi_core":            "CPILFESL",
    "pce":                 "PCEPI",
    "pce_core":            "PCEPILFE",
    # Activity
    "real_gdp":            "GDP",
    "unemployment_rate":   "UNRATE",
    "nonfarm_payrolls":    "PAYEMS",
    "industrial_production": "INDPRO",
    # Commodities and FX
    "wti_oil":             "WTISPLC",
    "brent_oil":           "DCOILBRENTEU",
    "gold":                "GOLDAMGBD228NLBM",
    "dxy":                 "DTWEXBGS",
    "usd_index":           "DTWEXBGS",
    "usdcny":              "DEXCHUS",
    "eurusd":              "DEXUSEU",
    "usdjpy":              "DEXJPUS",
    # Credit and vol
    "baa_spread":          "BAA10Y",
    "aaa_spread":          "AAA10Y",
    "tips_10y":            "DFII10",
    "breakeven_10y":       "T10YIE",
    "vix":                 "VIXCLS",
    "vixcls":              "VIXCLS",
}


def get_data(
    variable_name: str,
    source_hints: Optional[Sequence[str]] = None,
    date_range: Optional[tuple] = None,
) -> pl.DataFrame:
    """Unified data accessor.

    Resolution order:
        1. If variable resolves to a FRED id via ALIASES or looks like one, try Tier 1 FRED cache.
        2. If variable matches a known HF cache path, read it.
        3. Fall back to Tier 3 live fetch (fredapi, yfinance) via fallback module.

    Returns a polars DataFrame, typically with columns [date, value] for
    time series, or richer columns for panel data from HF.
    """
    key = variable_name.strip()

    # Alias resolution
    fred_id = ALIASES.get(key.lower(), key if key.isupper() else None)

    # 1. Tier 1 FRED cache
    if fred_id:
        try:
            df = _load_fred_cached(fred_id)
            return _apply_date_range(df, date_range)
        except FileNotFoundError:
            pass

    # 2. Tier 1 HF cache (exact path match)
    if key in HF_FILES:
        return _load_hf_cached(key)

    # 3. Tier 3 fallback
    from . import fallback

    if fred_id:
        try:
            df = fallback.fred_fallback(fred_id)
            return _apply_date_range(df, date_range)
        except Exception:
            pass

    # Last resort: treat as yfinance ticker if it has an uppercase letter
    if key and any(c.isalpha() and c.isupper() for c in key) and len(key) <= 12:
        try:
            return _apply_date_range(fallback.yfinance_fallback(key), date_range)
        except Exception:
            pass

    raise KeyError(f"Could not resolve variable '{variable_name}' through any tier.")


def _apply_date_range(df: pl.DataFrame, dr: Optional[tuple]) -> pl.DataFrame:
    if dr is None or "date" not in df.columns:
        return df
    start, end = dr
    q = df
    if start is not None:
        q = q.filter(pl.col("date") >= start)
    if end is not None:
        q = q.filter(pl.col("date") <= end)
    return q


__all__ = ["get_data", "ALIASES"]
