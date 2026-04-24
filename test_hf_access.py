"""Verifies auth + load_dataset() against the Bridgewater HF dataset."""
import os

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_ID = "BridgewaterAIHackathon/BW-AI-Hackathon"
api = HfApi(token=os.environ["HF_TOKEN"])

info = api.dataset_info(REPO_ID)
files = api.list_repo_files(REPO_ID, repo_type="dataset")

print(f"Access confirmed: {REPO_ID}")
print(f"Private: {info.private}   Files: {len(files)}")

top_dirs = sorted({f.split("/")[0] for f in files if "/" in f})
print("\nTop-level structure:")
for d in top_dirs:
    count = sum(1 for f in files if f.startswith(d + "/"))
    print(f"  {d}/   ({count} files)")

print("\nLoading a small CSV to verify load_dataset() works...")
ds = load_dataset(
    REPO_ID,
    data_files="Structured_Data/Macro/USA/RGDP.csv",
    token=os.environ["HF_TOKEN"],
)
df = ds["train"].to_pandas()
print(f"  Loaded: {len(df)} rows, columns = {list(df.columns)}")
print(df.head())
