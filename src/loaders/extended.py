"""Tier 2 lazy loader.

Reads company and sector data from the HF yahoo finance parquets that
were pulled by Tier 1, or constructs derived series on demand.

Scope is deliberately narrow for the MVP. Heavier functionality (sector
aggregates, transcript retrieval) is stubbed for Phase 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from .core import HF_CACHE


def _hf_path(rel: str) -> Path:
    return HF_CACHE / rel


def load_stock_prices(tickers: Optional[list[str]] = None) -> pl.DataFrame:
    """Read daily stock prices from the preloaded HF parquet.

    If tickers is given, filter to those. Else return the full frame.
    """
    path = _hf_path("yfin/stock_prices.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.loaders.core preload-hf` first."
        )
    df = pl.read_parquet(path)
    if tickers is None:
        return df
    # Stock parquet schema varies; try common column names defensively.
    for col in ("symbol", "ticker", "Symbol", "Ticker"):
        if col in df.columns:
            return df.filter(pl.col(col).is_in(tickers))
    raise KeyError(f"No ticker column found in {path.name}. Columns: {df.columns}")


def load_stock_fundamentals() -> pl.DataFrame:
    path = _hf_path("yfin/stock_statement.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pl.read_parquet(path)


def load_speeches() -> pl.DataFrame:
    """Read the ECB and FED speech corpus parquet."""
    path = _hf_path("text/ecb_fed_speeches.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pl.read_parquet(path)


def load_central_bank_sentences() -> pl.DataFrame:
    """Read annotated central bank speech sentences."""
    path = _hf_path("text/cb_sentences_annotated.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pl.read_parquet(path)


def load_ag_news() -> pl.DataFrame:
    """Read AG News business corpus."""
    path = _hf_path("text/ag_news.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pl.read_parquet(path)


def load_constructed_series(name: str) -> pl.DataFrame:
    """Compute derived series not available directly.

    Currently supported:
        effective_tariff_rate_china_semis  (built from event_catalog.csv)

    Raises NotImplementedError for unsupported names.
    """
    if name == "effective_tariff_rate_china_semis":
        return _build_effective_china_tariff()
    raise NotImplementedError(f"Constructed series '{name}' not implemented yet.")


def _build_effective_china_tariff() -> pl.DataFrame:
    """Cumulative effective tariff rate on Chinese semis, from event_catalog.

    This is a stair step series: each China list tariff announcement raises
    the effective rate on its date. Tier 2 constructs it on demand from the
    event catalog so there is no separate data file to maintain.
    """
    project_root = HF_CACHE.parents[2]
    catalog = pl.read_csv(project_root / "configs" / "event_catalog.csv")

    china_events = catalog.filter(
        pl.col("subject").str.contains("china_")
        & pl.col("policy_type").eq("trade")
    ).sort("date")

    # Stair step: carry forward cumulative tariff
    running = 0.0
    rows = []
    for row in china_events.iter_rows(named=True):
        running += float(row["magnitude"])
        rows.append({"date": row["date"], "value": running})
    if not rows:
        return pl.DataFrame({"date": [], "value": []})
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())
