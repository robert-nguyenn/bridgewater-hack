"""FastAPI backend.

Endpoints:
  POST /api/analyze         start a run, return run_id immediately
  GET  /api/runs/{run_id}   current status and (when done) full ImpactMap
  GET  /api/runs            list recent run ids on disk
  GET  /api/plots/{run_id}/{filename}  serve a plot png
  GET  /                    serve static/index.html (the SPA)

Runs are executed in a background task. Status is tracked in an
in-memory dict plus the pipeline_log.jsonl on disk so even after a
server restart a run's artifacts remain readable.

Launch:
    .venv/bin/uvicorn src.ui.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents._client import new_run_id
from src.pipeline.orchestrator import run_impact_analysis_async
from src.variable_labels import humanize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(title="bridgewater-hack impact mapper")


# In memory status tracker. Keys are run_ids; values are status dicts.
_RUN_STATUS: dict[str, dict] = {}


def _rehumanize_labels(impact_map: dict) -> None:
    """Overwrite node labels with natural English on the way out.

    Runs saved before the label dictionary landed have raw FRED codes or
    snake_case in node.label. Post processing here ensures the UI always
    renders human readable names regardless of when the run was saved.
    """
    for n in impact_map.get("nodes", []) or []:
        nid = n.get("node_id")
        if nid:
            n["label"] = humanize(nid)


class AnalyzeRequest(BaseModel):
    policy: str
    specialists: Optional[list[str]] = None   # default uses 3 for speed
    skip_review: bool = False


class AnalyzeResponse(BaseModel):
    run_id: str
    status: str


async def _run_background(run_id: str, req: AnalyzeRequest) -> None:
    _RUN_STATUS[run_id] = {"state": "running", "step": "starting", "error": None}
    try:
        impact = await run_impact_analysis_async(
            raw_policy=req.policy,
            run_id=run_id,
            specialists_to_run=req.specialists or ["monetary", "supply_chain", "international"],
            skip_review=req.skip_review,
        )
        _RUN_STATUS[run_id] = {
            "state": "done",
            "step": "done",
            "error": None,
            "summary": {
                "n_edges": len(impact.edges),
                "n_nodes": len(impact.nodes),
                "n_first_link": sum(1 for e in impact.edges if e.is_first_link),
                "n_review_flags": len(impact.review_flags),
            },
        }
    except Exception as exc:
        _RUN_STATUS[run_id] = {
            "state": "failed",
            "step": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, background: BackgroundTasks) -> AnalyzeResponse:
    run_id = new_run_id()
    background.add_task(_run_background, run_id, req)
    _RUN_STATUS[run_id] = {"state": "queued", "step": "queued", "error": None}
    return AnalyzeResponse(run_id=run_id, status="queued")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Combined status and (when done) the ImpactMap JSON."""
    status = _RUN_STATUS.get(run_id)
    impact_file = RUNS_DIR / run_id / "impact_map.json"
    out: dict = {"run_id": run_id, "status": status or {"state": "unknown"}}

    # Include pipeline log steps so far so the UI can show progress
    log_file = RUNS_DIR / run_id / "pipeline_log.jsonl"
    if log_file.exists():
        try:
            out["steps"] = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        except Exception:
            out["steps"] = []

    if impact_file.exists():
        try:
            impact_map = json.loads(impact_file.read_text())
            _rehumanize_labels(impact_map)
            out["impact_map"] = impact_map
        except Exception as exc:
            out["impact_map"] = None
            out["impact_map_error"] = str(exc)

    if not status and not impact_file.exists():
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return out


@app.get("/api/runs")
def list_runs() -> dict:
    """List run ids present on disk, newest first."""
    if not RUNS_DIR.exists():
        return {"runs": []}
    entries = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        impact_file = p / "impact_map.json"
        policy_file = p / "policy.json"
        record = {"run_id": p.name, "has_impact_map": impact_file.exists()}
        if policy_file.exists():
            try:
                pol = json.loads(policy_file.read_text())
                record["subject"] = pol.get("subject")
                record["policy_type"] = pol.get("policy_type")
                record["raw_input"] = (pol.get("raw_input") or "")[:120]
            except Exception:
                pass
        entries.append(record)
    return {"runs": entries[:50]}


@app.get("/api/plots/{run_id}/{filename}")
def get_plot(run_id: str, filename: str) -> FileResponse:
    path = RUNS_DIR / run_id / "plots" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"plot not found: {run_id}/{filename}")
    # Simple path traversal guard
    try:
        path.resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")
    return FileResponse(path, media_type="image/png")


# Static frontend. Mounted last so /api routes take precedence.
if STATIC_DIR.exists():
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            return HTMLResponse("<h1>bridgewater-hack UI</h1><p>static/index.html missing</p>")
        return HTMLResponse(idx.read_text())

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
