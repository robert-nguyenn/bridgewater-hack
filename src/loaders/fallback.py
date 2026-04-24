"""Tier 3 live fetch. Used when a variable is not in any Tier 1 or Tier 2 cache.

Every live fetch caches the response so the demo does not re hit rate
limits. Retries use tenacity with jittered exponential backoff.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "tier3"
FRED_T3 = CACHE_ROOT / "fred"
YFIN_T3 = CACHE_ROOT / "yfinance"
KALSHI_T3 = CACHE_ROOT / "kalshi"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def fred_fallback(series_id: str, start: str = "2000-01-01") -> pl.DataFrame:
    """Live FRED fetch. Caches result under data/cache/tier3/fred/."""
    FRED_T3.mkdir(parents=True, exist_ok=True)
    target = FRED_T3 / f"{series_id}.parquet"
    if target.exists():
        return pl.read_parquet(target)

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set, cannot live fetch.")

    from fredapi import Fred

    raw = Fred(api_key=api_key).get_series(series_id, observation_start=start)
    if raw is None or len(raw) == 0:
        raise ValueError(f"FRED returned empty for '{series_id}'")

    # Normalize: always return [date, value] regardless of what pandas named
    # the columns on reset_index (varies by pandas/fredapi version).
    pdf = raw.reset_index()
    pdf.columns = ["date", "value"]
    df = pl.from_pandas(pdf).with_columns(pl.col("date").cast(pl.Date))
    df = df.select(["date", "value"]).drop_nulls()
    df.write_parquet(target)
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def yfinance_fallback(ticker: str, start: str = "2000-01-01") -> pl.DataFrame:
    """Live yfinance fetch. Returns daily adjusted close as [date, value].

    Caches under data/cache/tier3/yfinance/.
    """
    YFIN_T3.mkdir(parents=True, exist_ok=True)
    target = YFIN_T3 / f"{ticker.replace('/', '_')}.parquet"
    if target.exists():
        return pl.read_parquet(target)

    import yfinance as yf

    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned empty for '{ticker}'")

    # yfinance returns MultiIndex columns if auto_adjust. Flatten.
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = [c[0] for c in raw.columns]

    close_col = "Close" if "Close" in raw.columns else raw.columns[0]
    sub = raw[[close_col]].reset_index()
    sub.columns = ["date", "value"]

    df = pl.from_pandas(sub).with_columns(pl.col("date").cast(pl.Date))
    df.write_parquet(target)
    return df


def kalshi_live(contract_id: str) -> pl.DataFrame:
    """Placeholder for Kalshi live fetch. Teammate owns the implementation.

    See src/loaders/kalshi.py (pre existing work) for the real integration.
    This stub exists so the unified loader has a known entry point.
    """
    raise NotImplementedError(
        "Kalshi live fetch is owned by teammate code in src/loaders/kalshi.py"
    )


def cache_stats() -> dict:
    """Report what is in Tier 3 cache, for debugging."""
    fred_files = list(FRED_T3.glob("*.parquet")) if FRED_T3.exists() else []
    yfin_files = list(YFIN_T3.glob("*.parquet")) if YFIN_T3.exists() else []
    return {
        "fred_fallback_cached": [f.stem for f in fred_files],
        "yfinance_cached": [f.stem for f in yfin_files],
        "as_of": _utc_now(),
    }
