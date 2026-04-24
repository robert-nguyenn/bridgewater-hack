"""Structural VAR impulse response lookup.

Reads configs/svar_lookup.yaml. Given a shock name and a horizon in
months, returns a MethodEstimate with the published point estimate and
approximate bands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.schemas import EstimatorType, MethodEstimate

from .plotting import plot_svar_irf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVAR_PATH = PROJECT_ROOT / "configs" / "svar_lookup.yaml"


def _load() -> list[dict]:
    with SVAR_PATH.open() as fh:
        return yaml.safe_load(fh)["impulse_responses"]


def lookup_svar(
    shock: str,
    response: str,
    horizon_months: int,
    plot_stem: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MethodEstimate:
    entries = _load()
    for e in entries:
        if e["shock"] != shock or e["response"] != response:
            continue
        best = _closest_horizon(e["horizons"], horizon_months)
        if best is None:
            continue
        se = max(abs(best["high"] - best["elasticity"]), abs(best["elasticity"] - best["low"]))

        plot_path = None
        if plot_stem:
            plot_path = plot_svar_irf(
                horizons=e["horizons"],
                title=f"{shock} -> {response}",
                stem=plot_stem,
                unit=e.get("unit", ""),
                run_id=run_id,
            )

        return MethodEstimate(
            method=EstimatorType.SVAR_LOOKUP,
            coefficient=float(best["elasticity"]),
            standard_error=float(se),
            sample_size=None,
            r_squared=None,
            passed=True,
            notes=(
                f"{e['source']}. horizon={best['months']} months. "
                f"band=[{best['low']}, {best['high']}] {e.get('unit', '')}"
            ),
            plot_path=plot_path,
        )
    return MethodEstimate(
        method=EstimatorType.SVAR_LOOKUP,
        coefficient=None,
        standard_error=None,
        sample_size=None,
        r_squared=None,
        passed=False,
        notes=f"No matching SVAR entry for shock={shock}, response={response}",
    )


def _closest_horizon(horizons: list[dict], target_months: int) -> Optional[dict]:
    if not horizons:
        return None
    return min(horizons, key=lambda h: abs(h["months"] - target_months))
