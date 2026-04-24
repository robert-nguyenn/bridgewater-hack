# bridgewater-hack

## Prerequisites

- Python 3.11+ ([download](https://www.python.org/downloads/) — check *Add Python to PATH* on install)
- Git

## Setup

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

> If activation fails with a "scripts disabled" error, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API keys

Copy the template:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and paste your real values:

```
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
```

- OpenAI key → https://platform.openai.com/api-keys
- Hugging Face token → https://huggingface.co/settings/tokens (Read scope is fine)

`.env` is gitignored — never commit it.

### 5. Verify setup

```bash
python test_setup.py
```

Expected output:
```
Python: 3.11.x
OPENAI_API_KEY: set
HF_TOKEN: set
All imports OK
```

## Daily workflow

Every new terminal:
```powershell
cd bridgewater-hack
.venv\Scripts\Activate      # macOS/Linux: source .venv/bin/activate
```

When you add a package:
```bash
pip install <package>
# then add it to requirements.txt with its pinned version
git add requirements.txt
git commit -m "Add <package>"
git push
```

## Using keys in Python

`.env` loads automatically via `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()

import os
openai_key = os.environ["OPENAI_API_KEY"]
hf_token = os.environ["HF_TOKEN"]
```

## Dependencies

- [huggingface_hub](https://github.com/huggingface/huggingface_hub) — HF model/dataset access
- [datasets](https://github.com/huggingface/datasets) — dataset loading
- [requests](https://github.com/psf/requests) — HTTP client
- [openai](https://github.com/openai/openai-python) — OpenAI API client
- [python-dotenv](https://github.com/theskumar/python-dotenv) — loads `.env` into env vars
