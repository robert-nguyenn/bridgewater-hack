"""Smoke test: verifies venv, deps, and .env are set up correctly."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

print(f"Python: {sys.version.split()[0]}")

checks = {
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "HF_TOKEN": os.environ.get("HF_TOKEN"),
}
for name, value in checks.items():
    status = "set" if value else "MISSING"
    print(f"{name}: {status}")

import datasets  # noqa: F401
import huggingface_hub  # noqa: F401
import openai  # noqa: F401
import requests  # noqa: F401

print("All imports OK")

if not all(checks.values()):
    print("\nOne or more env vars missing. Copy .env.example to .env and fill in real values.")
    sys.exit(1)
