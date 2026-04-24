"""Plot generation for every estimator output.

Each function here produces a matplotlib PNG and returns the relative
file path for storage on MethodEstimate.plot_path. Plots are the
primary artifact the review agent inspects to verify that the numbers
match the claims.

All plots use the Agg backend so they work headless. Output paths are
relative to PROJECT_ROOT so they can be serialized.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _plots_dir(run_id: Optional[str]) -> Path:
    base = PROJECT_ROOT / "data" / "runs" / (run_id or "standalone") / "plots"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _save(fig, run_id: Optional[str], stem: str) -> str:
    out = _plots_dir(run_id) / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Event study: scatter of shock change vs response change with fitted line
# ---------------------------------------------------------------------------
def plot_event_study(
    *,
    dx: list[float],
    dy: list[float],
    event_dates: list[date],
    coef: Optional[float],
    se: Optional[float],
    title: str,
    stem: str,
    shock_label: str = "shock change",
    response_label: str = "response change",
    run_id: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    if dx and dy:
        ax.scatter(dx, dy, s=40, alpha=0.75, edgecolor="k", linewidth=0.4)
        # Annotate points with year for recognizability
        for x, y, d in zip(dx, dy, event_dates):
            ax.annotate(str(d.year), (x, y), xytext=(4, 2), textcoords="offset points",
                        fontsize=7, alpha=0.6)
        if coef is not None and len(dx) >= 2:
            xs = np.linspace(min(dx), max(dx), 50)
            ys = coef * xs
            ax.plot(xs, ys, "r-", lw=1.5, label=f"fit beta={coef:.3f}")
            if se is not None and se > 0:
                band = 1.96 * se * np.abs(xs)
                ax.fill_between(xs, ys - band, ys + band, color="r", alpha=0.12, label="95% CI")
            ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No valid events in window", ha="center", va="center",
                transform=ax.transAxes)

    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel(shock_label)
    ax.set_ylabel(response_label)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)
    return _save(fig, run_id, stem)


# ---------------------------------------------------------------------------
# Level regression: scatter in first differences + fitted line
# ---------------------------------------------------------------------------
def plot_level_regression(
    *,
    x: list[float],
    y: list[float],
    coef: Optional[float],
    se: Optional[float],
    r2: Optional[float],
    title: str,
    stem: str,
    shock_label: str = "d(shock)",
    response_label: str = "d(response)",
    run_id: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    if x and y:
        # Subsample for readability if huge
        xs_all, ys_all = np.asarray(x), np.asarray(y)
        if len(xs_all) > 1000:
            idx = np.random.RandomState(0).choice(len(xs_all), 1000, replace=False)
            xs_plot, ys_plot = xs_all[idx], ys_all[idx]
        else:
            xs_plot, ys_plot = xs_all, ys_all
        ax.scatter(xs_plot, ys_plot, s=8, alpha=0.25, color="steelblue")

        if coef is not None and len(xs_all) >= 2:
            xs = np.linspace(xs_all.min(), xs_all.max(), 50)
            ys = coef * xs
            label = f"fit beta={coef:.3f}"
            if r2 is not None:
                label += f"  R^2={r2:.2f}"
            ax.plot(xs, ys, "r-", lw=1.5, label=label)
            if se is not None and se > 0:
                band = 1.96 * se * np.abs(xs)
                ax.fill_between(xs, ys - band, ys + band, color="r", alpha=0.12, label="95% CI")
            ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel(shock_label)
    ax.set_ylabel(response_label)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)
    return _save(fig, run_id, stem)


# ---------------------------------------------------------------------------
# Analog retrieval: bar chart of top-k analogs with similarity + realized response
# ---------------------------------------------------------------------------
def plot_analog_retrieval(
    *,
    analogs: list[dict],   # from retrieve_analogs
    title: str,
    stem: str,
    response_label: str = "realized response 30d (%)",
    run_id: Optional[str] = None,
) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    if not analogs:
        ax1.text(0.5, 0.5, "No analogs", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No analogs", ha="center", va="center", transform=ax2.transAxes)
        return _save(fig, run_id, stem)

    labels = [f"{a['event_date']}\n{a['event_name'][:28]}" for a in analogs]
    sims = [a.get("similarity", 0.0) for a in analogs]
    r30 = [a.get("response_30d_pct") for a in analogs]
    r90 = [a.get("response_90d_pct") for a in analogs]

    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, sims, color="steelblue", alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("cosine similarity")
    ax1.set_xlim(0, 1)
    ax1.set_title("analog similarity", fontsize=10)
    ax1.grid(True, alpha=0.2, axis="x")

    r30_plot = [0.0 if v is None else v for v in r30]
    r90_plot = [0.0 if v is None else v for v in r90]
    width = 0.4
    ax2.barh(y_pos - width/2, r30_plot, width, label="30d", color="tomato", alpha=0.85)
    ax2.barh(y_pos + width/2, r90_plot, width, label="90d", color="sandybrown", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel(response_label)
    ax2.set_title("realized response", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.2, axis="x")

    fig.suptitle(title, fontsize=11)
    return _save(fig, run_id, stem)


# ---------------------------------------------------------------------------
# SVAR impulse response curve
# ---------------------------------------------------------------------------
def plot_svar_irf(
    *,
    horizons: list[dict],   # [{months, elasticity, low, high}, ...]
    title: str,
    stem: str,
    unit: str = "",
    run_id: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not horizons:
        ax.text(0.5, 0.5, "No horizons", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, run_id, stem)

    months = [h["months"] for h in horizons]
    pts = [h["elasticity"] for h in horizons]
    lows = [h["low"] for h in horizons]
    highs = [h["high"] for h in horizons]

    ax.plot(months, pts, "o-", color="navy", lw=1.5)
    ax.fill_between(months, lows, highs, color="navy", alpha=0.18, label="band")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("horizon (months)")
    ax.set_ylabel(f"elasticity ({unit})" if unit else "elasticity")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, run_id, stem)
