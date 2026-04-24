"""Analog retrieval with semantic text + structural similarity.

Similarity is a weighted hybrid:

  final_sim = 0.60 * text_sim + 0.20 * type_match + 0.20 * magnitude_sim

  text_sim       : cosine between sentence-transformer embeddings of the
                   scenario raw_input and each event description. Primary
                   signal, captures mechanism similarity even when one-hot
                   categories match only superficially.
  type_match     : 1.0 if policy_types match, else 0.3
  magnitude_sim  : 1 - |log ratio| / 5 if magnitude units match, else 0.5

The old one-hot + macro-state feature approach had a failure mode where
all events of a given policy_type clustered at similarity >= 0.85 for any
query of that type, regardless of actual relevance. Semantic text
embeddings fix that.

Embeddings are lazy loaded once per process. Event embeddings are cached
to disk at data/cache/analog_embeddings.npz keyed by a sha of the catalog
CSV contents.
"""
from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from src.schemas import PolicyType, StructuredPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_CATALOG = PROJECT_ROOT / "configs" / "event_catalog.csv"
EMBED_CACHE = PROJECT_ROOT / "data" / "cache" / "analog_embeddings.npz"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

POLICY_TYPE_ORDER = [pt.value for pt in PolicyType]

# Macro state features (kept for backward compat + as a tiebreaker feature
# we may surface in the UI, but they no longer drive the primary score).
MACRO_FEATURES = {
    "unrate":   "UNRATE",
    "cpi_yoy":  "CPIAUCSL",
    "fed_funds": "DFF",
    "dgs10":    "DGS10",
    "dxy":      "DTWEXBGS",
}

# Hybrid weights
W_TEXT = 0.60
W_TYPE = 0.20
W_MAG = 0.20


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


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
_EMBED_MODEL = None
_EVENT_EMBED_CACHE: Optional[tuple[str, np.ndarray, list[dict]]] = None


def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL)
    return _EMBED_MODEL


def _catalog_sha() -> str:
    return hashlib.sha1(EVENT_CATALOG.read_bytes()).hexdigest()


def _event_text(row: dict) -> str:
    """What we embed for each event. Pandas NaN and None both coerce to ''."""
    def _str(v):
        if v is None:
            return ""
        try:
            if isinstance(v, float) and math.isnan(v):
                return ""
        except TypeError:
            pass
        return str(v).strip()
    parts = [_str(row.get("description")), _str(row.get("notes")), _str(row.get("subject"))]
    return ". ".join(p for p in parts if p).strip()


def _ensure_event_embeddings(catalog_rows: list[dict]) -> np.ndarray:
    """Embed catalog events. Cached to disk by catalog content hash."""
    global _EVENT_EMBED_CACHE
    sha = _catalog_sha()

    if _EVENT_EMBED_CACHE and _EVENT_EMBED_CACHE[0] == sha:
        return _EVENT_EMBED_CACHE[1]

    if EMBED_CACHE.exists():
        try:
            npz = np.load(EMBED_CACHE, allow_pickle=True)
            if str(npz.get("sha", "")) == sha:
                arr = npz["emb"]
                _EVENT_EMBED_CACHE = (sha, arr, catalog_rows)
                return arr
        except Exception:
            pass

    model = _get_model()
    texts = [_event_text(r) for r in catalog_rows]
    arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBED_CACHE, emb=arr, sha=sha)
    _EVENT_EMBED_CACHE = (sha, arr, catalog_rows)
    return arr


def _embed_query(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode([text], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]


# ---------------------------------------------------------------------------
# Similarity components
# ---------------------------------------------------------------------------
def _type_match(query_type: str, event_type: str) -> float:
    return 1.0 if query_type == event_type else 0.3


def _magnitude_sim(
    query_mag: float,
    query_unit: str,
    event_mag: float,
    event_unit: str,
) -> float:
    """If units match, compare magnitudes on a log scale. Else neutral 0.5."""
    q_unit = (query_unit or "").strip().lower()
    e_unit = (event_unit or "").strip().lower()
    # Treat similar unit names as the same
    if q_unit != e_unit:
        return 0.5
    a, b = abs(float(query_mag)), abs(float(event_mag))
    if a == 0 or b == 0:
        return 0.5
    try:
        ratio = math.log10(max(a, b) / min(a, b))   # 0 when equal, grows with divergence
    except ValueError:
        return 0.5
    # ratio of 0 -> 1.0, ratio of 2 (100x off) -> 0.2
    return max(0.0, 1.0 - ratio / 2.5)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def retrieve_analogs(
    policy: StructuredPolicy,
    response_series: Optional[pl.DataFrame] = None,
    k: int = 5,
) -> list[dict]:
    """Find k nearest historical analogs from the event catalog.

    Returns list of dicts, each with a similarity breakdown so the UI can
    show WHY the analog was picked.
    """
    catalog = _load_catalog()
    catalog_rows = [
        {
            "description": row.get("description", ""),
            "subject": row.get("subject", ""),
            "notes": row.get("notes", ""),
            "policy_type": row["policy_type"],
            "magnitude": float(row["magnitude"]),
            "magnitude_unit": row.get("magnitude_unit", ""),
            "date": row["date"],
        }
        for _, row in catalog.iterrows()
    ]
    if not catalog_rows:
        return []

    # Embeddings
    event_emb = _ensure_event_embeddings(catalog_rows)
    query_text = (policy.raw_input or policy.subject or "").strip()
    query_emb = _embed_query(query_text)
    text_sims = event_emb @ query_emb   # both normalized, so dot == cosine

    # Structural components per event
    type_sims = np.array([
        _type_match(policy.policy_type.value, r["policy_type"]) for r in catalog_rows
    ])
    mag_sims = np.array([
        _magnitude_sim(policy.magnitude, policy.magnitude_unit,
                       r["magnitude"], r["magnitude_unit"])
        for r in catalog_rows
    ])

    final = W_TEXT * text_sims + W_TYPE * type_sims + W_MAG * mag_sims
    order = np.argsort(-final)[:k]

    # Optional response readout
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
        r = catalog_rows[idx]
        ev_date = pd.to_datetime(r["date"])
        out.append({
            "event_name": r["description"],
            "event_date": ev_date.date(),
            "policy_type": r["policy_type"],
            "subject": r["subject"],
            "magnitude": r["magnitude"],
            "magnitude_unit": r["magnitude_unit"],
            "similarity": float(final[idx]),
            "similarity_breakdown": {
                "text": float(text_sims[idx]),
                "type_match": float(type_sims[idx]),
                "magnitude": float(mag_sims[idx]),
            },
            "response_30d_pct": _pct_change_at(ev_date, 30),
            "response_90d_pct": _pct_change_at(ev_date, 90),
            "response_180d_pct": _pct_change_at(ev_date, 180),
        })
    return out
