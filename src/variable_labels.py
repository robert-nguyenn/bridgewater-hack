"""Human readable labels for variables.

Maps FRED series IDs, friendly aliases, and common LLM invented names
to descriptive English labels. Used by the graph builder and UI.

Unknown identifiers fall back to a humanize() helper that handles
snake_case, camelCase, common abbreviations, and yfinance tickers.
"""
from __future__ import annotations

import re


# Primary lookup. Keys are normalized to lower case for case insensitive
# matching. Values are descriptive English strings.
LABELS: dict[str, str] = {
    # Policy rates and Treasury curve
    "dff":         "Fed funds effective rate",
    "fedfunds":    "Fed funds target rate",
    "sofr":        "SOFR overnight rate",
    "dgs2":        "2-year Treasury yield",
    "dgs5":        "5-year Treasury yield",
    "dgs10":       "10-year Treasury yield",
    "dgs30":       "30-year Treasury yield",
    "dfii10":      "10-year TIPS real yield",
    "t10yie":      "10-year breakeven inflation",
    "baa10y":      "BAA corporate spread over 10Y",
    "aaa10y":      "AAA corporate spread over 10Y",

    # Inflation
    "cpiaucsl":    "Headline CPI (all items)",
    "cpilfesl":    "Core CPI (ex food and energy)",
    "pcepi":       "Headline PCE price index",
    "pcepilfe":    "Core PCE price index",
    "cpiengsl":    "CPI energy",
    "cpi_energy_yoy":  "CPI energy (YoY)",
    "cpi_core":    "Core CPI",
    "cpi_headline": "Headline CPI",

    # Activity
    "gdp":         "US real GDP",
    "real_gdp":    "Real GDP",
    "unrate":      "US unemployment rate",
    "payems":      "Nonfarm payrolls",
    "indpro":      "Industrial production",
    "rsxfs":       "Retail sales",
    "industrial_production": "Industrial production",
    "nonfarm_payrolls": "Nonfarm payrolls",
    "unemployment_rate": "Unemployment rate",

    # Commodities and FX
    "wtisplc":     "WTI crude oil price",
    "wti_oil":     "WTI crude oil price",
    "dcoilbrenteu": "Brent crude oil price",
    "brent_oil":   "Brent crude oil price",
    "goldamgbd228nlbm": "Gold price (London AM)",
    "gold":        "Gold price",
    "dxy":         "US dollar index (broad)",
    "dtwexbgs":    "US dollar index (broad trade-weighted)",
    "usd_index":   "US dollar index",
    "dexchus":     "USD / CNY exchange rate",
    "usdcny":      "USD / CNY exchange rate",
    "dexuseu":     "USD / EUR exchange rate",
    "eurusd":      "EUR / USD exchange rate",
    "dexjpus":     "USD / JPY exchange rate",
    "usdjpy":      "USD / JPY exchange rate",
    "usdbrl=x":    "USD / BRL exchange rate",
    "usdmxn=x":    "USD / MXN exchange rate",
    "usdnok=x":    "USD / NOK exchange rate",
    "usdcad=x":    "USD / CAD exchange rate",
    "usdinr=x":    "USD / INR exchange rate",

    # Volatility and credit
    "vixcls":      "VIX (equity volatility)",
    "vix":         "VIX (equity volatility)",
    "tips_10y":    "10-year TIPS real yield",
    "breakeven_10y": "10-year breakeven inflation",
    "baa_spread":  "BAA corporate spread",
    "aaa_spread":  "AAA corporate spread",

    # Equities (common ticker shortcuts the LLM uses)
    "sp500":       "S&P 500 index",
    "spx":         "S&P 500 index",
    "eem":         "EM equity ETF (EEM)",
    "em_equity_index": "EM equity index",
    "msci_em":     "MSCI Emerging Markets",
    "nasdaq":      "NASDAQ Composite",

    # Scenario trigger patterns (LLM invented names we expect to see)
    "hormuz_closure_probability": "Probability of Strait of Hormuz closure",
    "strait_of_hormuz_closure_probability": "Probability of Strait of Hormuz closure",
    "strait_of_hormuz_closure": "Strait of Hormuz closure",
    "supply_disruption_pct_global": "Global oil supply disruption (% of global)",
    "effective_tariff_rate":       "Effective tariff rate",
    "effective_tariff_rate_china_semis": "Effective tariff on Chinese semiconductors",
    "commodity_basket":            "Commodity basket",
    "bcom_energy_index":           "Bloomberg commodity energy index",
}


# Regex patterns for heuristic humanization when nothing else matches.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^usd(?P<ccy>[a-z]{3})(?:=x)?$", re.I), r"USD / \g<ccy> exchange rate"),
    (re.compile(r"^(?P<ccy>[a-z]{3})usd=x$", re.I), r"\g<ccy> / USD exchange rate"),
    (re.compile(r"^dgs(\d+)$", re.I), r"\1-year Treasury yield"),
    (re.compile(r"^cpi(.*)_yoy$", re.I), lambda m: f"CPI {m.group(1).strip('_')} (YoY)"),
]


_ABBREV_EXPANSIONS: dict[str, str] = {
    "cpi": "CPI",
    "pce": "PCE",
    "gdp": "GDP",
    "usd": "USD",
    "eur": "EUR",
    "jpy": "JPY",
    "gbp": "GBP",
    "chf": "CHF",
    "cny": "CNY",
    "yoy": "(YoY)",
    "mom": "(MoM)",
    "ema": "EMA",
    "sma": "SMA",
    "pmi": "PMI",
    "ism": "ISM",
    "fx":  "FX",
    "em":  "EM",
    "dm":  "DM",
    "vix": "VIX",
    "ecb": "ECB",
    "fed": "Fed",
    "boj": "BoJ",
    "boe": "BoE",
    "opec": "OPEC",
    "spr": "SPR",
    "ttf": "TTF",
    "btp": "BTP",
}


def humanize(identifier: str) -> str:
    """Return a natural English label for a variable identifier.

    Resolution order:
        1. Exact lower case match in LABELS
        2. Heuristic pattern match (DGS<n>, USD/<ccy>, etc.)
        3. Fall back to title cased snake_case with abbreviation expansion
    """
    if identifier is None:
        return ""
    s = str(identifier).strip()
    if not s:
        return ""

    key = s.lower()
    if key in LABELS:
        return LABELS[key]

    for pat, repl in _PATTERNS:
        m = pat.match(s)
        if m:
            if callable(repl):
                return repl(m)
            return m.expand(repl)

    # Fall back: split, expand abbreviations, title case
    parts = re.split(r"[_\-\s/]+", s)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if p.lower() in _ABBREV_EXPANSIONS:
            out.append(_ABBREV_EXPANSIONS[p.lower()])
        elif p.isupper() and len(p) <= 5:
            # Keep short all caps tokens as is (FRED codes, currency codes)
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out)


__all__ = ["LABELS", "humanize"]
