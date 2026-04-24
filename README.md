# bridgewater-hack

## Setup

Requires Python 3.11+.

### 1. Clone

```bash
git clone https://github.com/robert-nguyenn/bridgewater-hack.git
cd bridgewater-hack
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `OPENAI_API_KEY` — OpenAI API key
- `HF_TOKEN` — Hugging Face access token (from https://huggingface.co/settings/tokens)

## Dependencies

- [huggingface_hub](https://github.com/huggingface/huggingface_hub) — HF model/dataset access
- [datasets](https://github.com/huggingface/datasets) — dataset loading
- [requests](https://github.com/psf/requests) — HTTP client
- [openai](https://github.com/openai/openai-python) — OpenAI API client
