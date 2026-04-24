# bridgewater-hack

Policy Impact Scenario Mapper. Takes a plain-language policy action
(e.g. "25% tariff on Chinese semiconductors" or "Strait of Hormuz
closed for 7+ days with >50% probability"), decomposes it into a
causal graph of economic channels, estimates magnitudes empirically
from FRED and Hugging Face data, and returns an interactive impact
map with per-edge regression plots and an adversarial review pass.

## Architecture at a glance

```
raw policy text
     │
     ▼
┌──────────────┐     ┌────────────────────────┐
│ policy parser│────▶│ coordinator (rule + LLM)│
└──────────────┘     └────────┬───────────────┘
                              │ channel_ids
                              ▼
                 ┌──────────────────────────────┐
                 │ 5 specialists (parallel Opus)│
                 │  monetary, supply_chain,     │
                 │  financial_conditions,       │
                 │  international, behavioral   │
                 └────────┬─────────────────────┘
                          │ hypotheses
                          ▼
                 ┌───────────────────────────┐
                 │ adversary (per-hypothesis │
                 │  + per-analog parallel)   │
                 └────────┬──────────────────┘
                          ▼
                 ┌────────────────────────────┐
                 │ empirics router            │
                 │  event_study / level_reg / │
                 │  analog_retrieval /        │
                 │  svar_lookup               │
                 │  all emit matplotlib plots │
                 └────────┬───────────────────┘
                          ▼
                 ┌────────────────────────────┐
                 │ graph builder              │
                 │  (first-link detection)    │
                 └────────┬───────────────────┘
                          ▼
                 ┌────────────────────────────┐
                 │ synthesizer (per-edge)     │
                 └────────┬───────────────────┘
                          ▼
                 ┌────────────────────────────┐
                 │ reviewer (vision on plots) │
                 └────────┬───────────────────┘
                          ▼
                       ImpactMap
```

## Prerequisites

- Python 3.12 (the project pins `>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) for dependency management
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS / Linux
  ```
- API keys:
  - `ANTHROPIC_API_KEY` for the agents (https://console.anthropic.com)
  - `FRED_API_KEY` for macro data (https://fred.stlouisfed.org/docs/api/api_key.html, free)
  - `HF_TOKEN` with read scope for the dataset (https://huggingface.co/settings/tokens)

## Setup

### 1. Clone and sync

```bash
git clone https://github.com/robert-nguyenn/bridgewater-hack.git
cd bridgewater-hack
uv sync
```

`uv sync` creates `.venv/` and installs everything from `uv.lock`.
All four teammates get byte-identical environments.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
FRED_API_KEY=...
HF_TOKEN=hf_...
```

`.env` is gitignored.

### 3. Preload the data cache (run once)

```bash
.venv/bin/python -m src.loaders.core preload
```

This pulls:
- 27 FRED series (rates, inflation, activity, FX, credit, VIX) from 2000-present
- 34 HF files (macro CSVs by country, USA factor returns, Yahoo Finance
  preprocessed parquets, central-bank speech corpora)

Total ~1.5 GB, cached to `data/cache/tier1/` (gitignored). Takes ~2 min
on a warm home connection.

Check progress:
```bash
.venv/bin/python -m src.loaders.core status
```

### 4. Verify

```bash
.venv/bin/python tests/test_pipeline.py     # schemas smoke test, no network
.venv/bin/python tests/test_empirics.py     # empirics integration, needs tier 1 cache
```

Both should print `PASSED` at the end.

## Running the pipeline

### Full end-to-end scenario

```python
from src.pipeline.orchestrator import run_impact_analysis

impact = run_impact_analysis(
    "What if the Strait of Hormuz is closed for 7+ days with >50% probability over 30 days?"
)
print(f"run_id: {impact.generation_metadata['run_id']}")
print(f"edges: {len(impact.edges)}  first_link: {sum(1 for e in impact.edges if e.is_first_link)}")
```

Or run the test as a driver:
```bash
.venv/bin/python tests/test_orchestrator.py
```

Expect ~2.5-3 min wall time (uses 3 of 5 specialists to keep API cost
bounded; full 5 specialists takes ~3.5 min).

### Artifacts per run

Every run creates `data/runs/<run_id>/` with:
- `impact_map.json` — full ImpactMap output
- `hypotheses_enriched.json` — all hypotheses with per-analog adversarial critiques
- `policy.json`, `channels.json` — intermediate artifacts
- `plots/` — matplotlib PNGs, one per method estimate
- `llm_log.jsonl` — every Claude call with tokens, cache hits, elapsed time
- `pipeline_log.jsonl` — step-by-step orchestrator timings

## Project layout

```
bridgewater-hack/
  configs/
    channel_catalog.yaml       # 19 transmission channels
    event_catalog.csv          # 51 historical policy events
    svar_lookup.yaml           # 12 published impulse responses
    specialist_prompts/        # 5 prompt files, one per specialist
  src/
    schemas.py                 # THE pydantic contract
    loaders/
      core.py                  # Tier 1 FRED + HF preloader
      extended.py              # Tier 2 lazy HF parquet access
      fallback.py              # Tier 3 live FRED / yfinance
      __init__.py              # get_data() unified entry
    empirics/
      event_study.py
      level_regression.py
      analog_retrieval.py
      svar_lookup.py
      plotting.py              # matplotlib figures for all estimators
      router.py
    agents/
      _client.py               # Anthropic wrapper with caching and logging
      policy_parser.py
      coordinator.py
      specialists.py           # shared runner, configs in configs/specialist_prompts/
      adversary.py             # per-hypothesis + per-analog critiques
      synthesizer.py           # rolls up estimates into EstimateRange + caveats
      reviewer.py              # vision-based plot inspection
    graph/
      builder.py
    pipeline/
      orchestrator.py          # async top-level flow
  tests/
    test_pipeline.py
    test_empirics.py
    test_agents.py
    test_orchestrator.py
```

## Daily workflow

```bash
cd bridgewater-hack
source .venv/bin/activate        # or: uv run <command>
```

Add a dependency:
```bash
uv add <package>
git add pyproject.toml uv.lock
git commit -m "Add <package>"
```

## Troubleshooting

**`ModuleNotFoundError` after git pull** — dependencies moved. Run:
```bash
uv sync
```

**FRED 404 on `GOLDAMGBD228NLBM`** — FRED retired that series code. Not used
by any current hypothesis. Ignore.

**`ColumnNotFoundError: unable to find column "date"`** on a Tier 3 fallback
series — clear the Tier 3 cache and let it re-fetch with the normalized
column names:
```bash
rm -rf data/cache/tier3/
```

**Anthropic `max_tokens` hit on adversary** — shouldn't happen anymore; the
adversary splits per hypothesis. If it does, lower the number of episodes
per hypothesis in specialist prompts.

**PowerShell activation error** (Windows only):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Dependencies (major)

- [anthropic](https://docs.anthropic.com) — Claude API client, Opus 4.7
- [pydantic](https://docs.pydantic.dev) — schemas
- [polars](https://pola.rs) — columnar dataframes throughout
- [fredapi](https://github.com/mortada/fredapi) — FRED client
- [yfinance](https://github.com/ranaroussi/yfinance) — Tier 3 fallback
- [statsmodels](https://www.statsmodels.org) — OLS, HC3, Newey-West
- [matplotlib](https://matplotlib.org) — per-estimator plots
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) — dataset access
- [fastapi](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org) — UI backend
- [sentence-transformers](https://www.sbert.net) + [faiss-cpu](https://github.com/facebookresearch/faiss) — semantic retrieval (Phase 4)
