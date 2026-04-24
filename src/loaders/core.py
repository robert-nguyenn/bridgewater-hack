"""Tier 1 preloader.

Pulls a curated set of macro time series from FRED and the project
HuggingFace dataset, converts everything to polars DataFrames with a
standard shape of [date, value], and writes them to
data/cache/tier1/. A manifest.json records what was pulled and when,
so reruns only refresh stale series.

Design notes:

* Every network call has graceful degradation. A missing FRED key or
  a single failing series does not abort the whole preload.
* The cache layout is deliberately flat and uses series_id as the
  filename, so load_series(series_id) is a one liner.
* polars is the in memory format. fredapi and yfinance return pandas,
  so we convert at the boundary.

Intended entry points:

    python -m src.loaders.core preload     # pulls everything
    python -m src.loaders.core status      # prints what is cached

From code:

    from src.loaders.core import load_series
    df = load_series("DGS2")
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "tier1"
FRED_CACHE = CACHE_ROOT / "fred"
HF_CACHE = CACHE_ROOT / "hf"
MANIFEST_PATH = CACHE_ROOT / "manifest.json"

HF_REPO = "BridgewaterAIHackathon/BW-AI-Hackathon"

# ---------------------------------------------------------------------------
# FRED series to pull. Each entry is (series_id, frequency_hint) grouped by
# economic role. Frequency hint is informational, not used for fetching.
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, list[str]] = {
    "rates":      ["DFF", "DGS2", "DGS5", "DGS10", "DGS30", "FEDFUNDS", "SOFR"],
    "inflation":  ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"],
    "activity":   ["GDP", "UNRATE", "PAYEMS", "INDPRO", "RSXFS"],
    "prices":     ["WTISPLC", "DCOILBRENTEU", "GOLDAMGBD228NLBM"],
    "fx":         ["DTWEXBGS", "DEXCHUS", "DEXUSEU", "DEXJPUS"],
    "credit":     ["BAA10Y", "AAA10Y", "DFII10", "T10YIE"],
    "volatility": ["VIXCLS"],
}

FRED_START = "2000-01-01"

# ---------------------------------------------------------------------------
# HF files to pull. Each entry maps a (repo path) to a local cache subpath.
# ---------------------------------------------------------------------------
HF_FILES: dict[str, str] = {
    # Macro time series (structured)
    "Structured_Data/Macro/USA/USA_10_Year_Yield.csv":    "macro/usa_10y_yield.csv",
    "Structured_Data/Macro/USA/Core_CPI_Raw.csv":         "macro/usa_cpi_core.csv",
    "Structured_Data/Macro/USA/Headline_CPI_Raw.csv":     "macro/usa_cpi_headline.csv",
    "Structured_Data/Macro/USA/RGDP.csv":                 "macro/usa_rgdp.csv",
    "Structured_Data/Macro/USA/CorporateDebtIndex.csv":   "macro/usa_corp_debt.csv",
    "Structured_Data/Macro/Euro/EUR_10_Year_Yield.csv":   "macro/eur_10y_yield.csv",
    "Structured_Data/Macro/Euro/Core_CPI_Raw.csv":        "macro/eur_cpi_core.csv",
    "Structured_Data/Macro/Euro/RGDP.csv":                "macro/eur_rgdp.csv",
    "Structured_Data/Macro/UK/GBR_10_Year_Yield.csv":     "macro/uk_10y_yield.csv",
    "Structured_Data/Macro/UK/Core_CPI_YoY.csv":          "macro/uk_cpi_core.csv",
    "Structured_Data/Macro/UK/RGDP.csv":                  "macro/uk_rgdp.csv",
    "Structured_Data/Macro/Japan/JPN_10_Year_Yield.csv":  "macro/jpn_10y_yield.csv",
    "Structured_Data/Macro/Japan/Core_CPI_YoY.csv":       "macro/jpn_cpi_core.csv",
    "Structured_Data/Macro/Japan/RGDP.csv":               "macro/jpn_rgdp.csv",
    "Structured_Data/Macro/Canada/CAN_10_Year_Yield.csv": "macro/can_10y_yield.csv",
    "Structured_Data/Macro/Canada/Core_CPI_YoY.csv":      "macro/can_cpi_core.csv",
    "Structured_Data/Macro/Canada/RGDP.csv":              "macro/can_rgdp.csv",
    "Structured_Data/Macro/Australia/AUS_10_Year_Yield.csv": "macro/aus_10y_yield.csv",
    "Structured_Data/Macro/Australia/Core_CPI_YoY.csv":   "macro/aus_cpi_core.csv",
    "Structured_Data/Macro/Australia/RGDP.csv":           "macro/aus_rgdp.csv",
    "Structured_Data/Macro/Oil/Brent.csv":                "macro/oil_brent.csv",
    "Structured_Data/Macro/Oil/WTI.csv":                  "macro/oil_wti.csv",

    # USA factor returns. Paths contain brackets so copy filenames literally.
    "Structured_Data/SNE/USA_Factor_Returns/[usa]_[age]_[monthly]_[vw_cap].csv":         "factors/usa_age_monthly.csv",
    "Structured_Data/SNE/USA_Factor_Returns/[usa]_[all_themes]_[monthly]_[vw_cap].csv":  "factors/usa_themes_monthly.csv",
    "Structured_Data/SNE/USA_Factor_Returns/[usa]_[gics]_[monthly]_[vw_cap].csv":        "factors/usa_gics_monthly.csv",

    # Yahoo Finance preprocessed parquets (equities, FX, treasury)
    "Structured_Data/SNE/yahoo-finance-data/company_tickers.json":          "yfin/company_tickers.json",
    "Structured_Data/SNE/yahoo-finance-data/daily_treasury_yield.parquet":  "yfin/daily_treasury_yield.parquet",
    "Structured_Data/SNE/yahoo-finance-data/exchange_rate.parquet":         "yfin/exchange_rate.parquet",
    "Structured_Data/SNE/yahoo-finance-data/stock_prices.parquet":          "yfin/stock_prices.parquet",
    "Structured_Data/SNE/yahoo-finance-data/stock_statement.parquet":       "yfin/stock_statement.parquet",
    "Structured_Data/SNE/yahoo-finance-data/stock_news.parquet":            "yfin/stock_news.parquet",

    # Text corpora (speeches, news). Keep to the lighter ones for Tier 1.
    "Unstructured_Data/Macro/ECB-FED-speeches/train-00000-of-00001.parquet": "text/ecb_fed_speeches.parquet",
    "Unstructured_Data/Macro/ag_news/train-00000-of-00001.parquet":          "text/ag_news.parquet",
    "Unstructured_Data/Macro/central_bank_communications/sentences_annotated.parquet": "text/cb_sentences_annotated.parquet",
}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _read_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"created": _utc_now(), "fred": {}, "hf": {}}


def _write_manifest(m: dict) -> None:
    m["updated"] = _utc_now()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, default=str))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------
def load_series(series_id: str) -> pl.DataFrame:
    """Read a FRED series from Tier 1 cache.

    Returns columns [date, value]. Raises FileNotFoundError if the series
    is not cached. Does not hit the network.
    """
    path = FRED_CACHE / f"{series_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Tier 1 cache miss for FRED series '{series_id}'. "
            f"Run `python -m src.loaders.core preload` or call Tier 3 fallback."
        )
    return pl.read_parquet(path)


def load_hf_file(hf_path: str) -> pl.DataFrame:
    """Read a cached HF file by its original repo path.

    Auto detects csv vs parquet. Returns a polars DataFrame.
    """
    if hf_path not in HF_FILES:
        raise KeyError(f"HF path not registered in HF_FILES: {hf_path}")
    local = HF_CACHE / HF_FILES[hf_path]
    if not local.exists():
        raise FileNotFoundError(f"Tier 1 HF cache miss: {local}")
    if local.suffix == ".parquet":
        return pl.read_parquet(local)
    if local.suffix == ".csv":
        return pl.read_csv(local, try_parse_dates=True)
    if local.suffix == ".json":
        return pl.read_json(local)
    raise ValueError(f"Unsupported file type: {local.suffix}")


# ---------------------------------------------------------------------------
# FRED preloader
# ---------------------------------------------------------------------------
def _fetch_one_fred(series_id: str, api_key: str) -> Optional[pl.DataFrame]:
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    raw = fred.get_series(series_id, observation_start=FRED_START)
    if raw is None or len(raw) == 0:
        return None

    # Normalize column names regardless of what pandas/fredapi chose.
    pdf = raw.reset_index()
    pdf.columns = ["date", "value"]
    return (
        pl.from_pandas(pdf)
        .with_columns(pl.col("date").cast(pl.Date))
        .select(["date", "value"])
        .drop_nulls()
    )


def preload_fred(force: bool = False) -> dict:
    """Pull every series in FRED_SERIES to data/cache/tier1/fred/.

    Returns a summary dict with counts of pulled, skipped, failed.
    """
    manifest = _read_manifest()
    api_key = os.environ.get("FRED_API_KEY")
    FRED_CACHE.mkdir(parents=True, exist_ok=True)

    if not api_key:
        print("  [fred] FRED_API_KEY not set, skipping FRED preload.")
        return {"pulled": 0, "skipped": 0, "failed": 0, "reason": "no_api_key"}

    all_ids: list[str] = [sid for group in FRED_SERIES.values() for sid in group]
    pulled = skipped = failed = 0
    errors: list[str] = []

    for sid in all_ids:
        target = FRED_CACHE / f"{sid}.parquet"
        if target.exists() and not force:
            skipped += 1
            continue
        try:
            df = _fetch_one_fred(sid, api_key)
            if df is None or df.is_empty():
                failed += 1
                errors.append(f"{sid}: empty response")
                continue
            df.write_parquet(target)
            manifest["fred"][sid] = {
                "path": str(target.relative_to(PROJECT_ROOT)),
                "n_rows": df.height,
                "date_min": str(df["date"].min()),
                "date_max": str(df["date"].max()),
                "fetched_at": _utc_now(),
            }
            pulled += 1
            print(f"  [fred] {sid:<12} {df.height:>6} rows  {df['date'].min()} to {df['date'].max()}")
        except Exception as exc:
            failed += 1
            errors.append(f"{sid}: {type(exc).__name__}: {exc}")
            print(f"  [fred] {sid:<12} FAILED  {type(exc).__name__}: {exc}")

    _write_manifest(manifest)
    summary = {"pulled": pulled, "skipped": skipped, "failed": failed, "errors": errors}
    print(f"  [fred] summary  pulled={pulled} skipped={skipped} failed={failed}")
    return summary


# ---------------------------------------------------------------------------
# HF preloader
# ---------------------------------------------------------------------------
def preload_hf(force: bool = False) -> dict:
    """Download every file in HF_FILES to data/cache/tier1/hf/.

    Uses hf_hub_download which handles streaming and local hashing.
    """
    from huggingface_hub import hf_hub_download

    manifest = _read_manifest()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("  [hf] HF_TOKEN not set, skipping HF preload.")
        return {"pulled": 0, "skipped": 0, "failed": 0, "reason": "no_token"}

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    pulled = skipped = failed = 0
    errors: list[str] = []

    for repo_path, local_rel in HF_FILES.items():
        target = HF_CACHE / local_rel
        if target.exists() and not force:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=repo_path,
                token=token,
            )
            # hf_hub_download caches to its own path. Copy into our cache.
            import shutil
            shutil.copy(downloaded, target)

            size = target.stat().st_size
            manifest["hf"][repo_path] = {
                "local": str(target.relative_to(PROJECT_ROOT)),
                "size_bytes": size,
                "fetched_at": _utc_now(),
            }
            pulled += 1
            print(f"  [hf]  {local_rel:<40} {size/1024:>8.1f} KB")
        except Exception as exc:
            failed += 1
            errors.append(f"{repo_path}: {type(exc).__name__}: {exc}")
            print(f"  [hf]  {local_rel:<40} FAILED  {type(exc).__name__}: {exc}")

    _write_manifest(manifest)
    summary = {"pulled": pulled, "skipped": skipped, "failed": failed, "errors": errors}
    print(f"  [hf]  summary  pulled={pulled} skipped={skipped} failed={failed}")
    return summary


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def preload_all(force: bool = False) -> dict:
    print(f"Preload start. cache_root={CACHE_ROOT}")
    fred = preload_fred(force=force)
    hf = preload_hf(force=force)
    print("Preload done.")
    return {"fred": fred, "hf": hf}


def status() -> None:
    manifest = _read_manifest()
    print(f"Manifest at {MANIFEST_PATH}")
    print(f"  FRED cached: {len(manifest.get('fred', {}))} series")
    print(f"  HF cached:   {len(manifest.get('hf', {}))} files")

    # Flag gaps against the canonical lists
    cached_fred = set(manifest.get("fred", {}).keys())
    expected_fred = {s for g in FRED_SERIES.values() for s in g}
    missing_fred = expected_fred - cached_fred
    if missing_fred:
        print(f"  FRED missing: {sorted(missing_fred)}")

    cached_hf = set(manifest.get("hf", {}).keys())
    missing_hf = set(HF_FILES) - cached_hf
    if missing_hf:
        print(f"  HF missing: {len(missing_hf)} files")


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    cmd = args[0] if args else "status"
    force = "--force" in args

    if cmd == "preload":
        preload_all(force=force)
    elif cmd == "preload-fred":
        preload_fred(force=force)
    elif cmd == "preload-hf":
        preload_hf(force=force)
    elif cmd == "status":
        status()
    else:
        print(f"usage: python -m src.loaders.core [preload|preload-fred|preload-hf|status] [--force]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
