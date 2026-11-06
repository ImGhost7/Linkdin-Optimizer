"""Build data/jobs_sample.csv from the public LinkedIn Job Postings (2023-24) dataset.

Source : https://huggingface.co/datasets/datastax/linkedin_job_listings  (mirror of the Kaggle
         "LinkedIn Job Postings (2023 - 2024)" dataset, 123,849 rows, 517 MB).
Method : we do NOT download the 517 MB CSV. We read ~400 random 100-row windows (~1/3 of the
         dataset) through the Hugging Face datasets-server `rows` HTTP API, then stratify
         client-side (the server-side `filter`/`search` endpoints are unreliable for this dataset):
           * "tech/business" slice - rows whose title matches a keyword in TITLE_KEYWORDS (quota each)
           * "random" slice        - other rows, so the pool also contains unrelated jobs and ranking
                                     is a realistic, non-trivial task
Seeded, so the sample is reproducible:   python data/build_jobs_sample.py
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

DATASET = "datastax/linkedin_job_listings"
BASE = "https://datasets-server.huggingface.co"
TOTAL_ROWS = 123_849
SEED = 42
OUT = Path(__file__).parent / "jobs_sample.csv"

# keyword -> max rows to take (client-side title match). ~1,300 "tech/business" rows.
TITLE_KEYWORDS = {
    "data scientist": 110, "machine learning": 110, "data analyst": 110, "data engineer": 110,
    "software engineer": 140, "frontend": 80, "front end": 40, "backend": 80, "full stack": 90,
    "devops": 90, "cloud": 70, "security": 80, "product manager": 90, "project manager": 60,
    "marketing": 90, "business analyst": 70, "mobile": 50, "qa ": 40, "ui/ux": 40,
}
N_RANDOM = 700
N_WINDOWS = 400
KEEP = ["job_id", "title", "company_name", "location", "description", "skills_desc",
        "formatted_experience_level", "formatted_work_type", "job_posting_url"]


def get(client: httpx.Client, path: str, params: dict, tries: int = 8) -> dict:
    """GET with retry - the datasets-server occasionally returns transient 5xx/auth errors."""
    for attempt in range(tries):
        try:
            r = client.get(f"{BASE}/{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"datasets-server request failed: {path} {params}")


CACHE = Path(__file__).parent / ".cache" / "windows"     # resumable: finished windows are kept on disk


def fetch_window(offset: int) -> list[dict]:
    """One 100-row window; cached on disk. Returns [] (and warns) if the API keeps failing."""
    f = CACHE / f"{offset}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    try:
        with httpx.Client() as client:
            page = get(client, "rows", {"dataset": DATASET, "config": "default", "split": "train",
                                        "offset": offset, "length": 100})
    except RuntimeError as exc:
        print(f"  ! skipping window {offset}: {exc}", flush=True)
        return []
    rows = [r["row"] for r in page["rows"]]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def main() -> None:
    rng = random.Random(SEED)
    n_windows = (TOTAL_ROWS - 100) // 100
    offsets = [w * 100 for w in rng.sample(range(n_windows), N_WINDOWS)]
    pool: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, rows in enumerate(ex.map(fetch_window, offsets), 1):
            pool += rows
            if i % 50 == 0:
                print(f"  fetched {i}/{N_WINDOWS} windows ({len(pool)} rows)", flush=True)
    df = pd.DataFrame(pool).drop_duplicates("job_id")
    df["description"] = df["description"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["description"].str.len() > 200]           # drop empty/stub postings
    df = df.sample(frac=1, random_state=SEED)             # shuffle so quotas pick random rows

    picked, taken = [], set()
    for kw, quota in TITLE_KEYWORDS.items():
        hit = df[df["title"].str.contains(kw, case=False, regex=False, na=False) & ~df["job_id"].isin(taken)]
        hit = hit.head(quota)
        taken |= set(hit["job_id"])
        picked.append(hit)
        print(f"  {kw!r:20} available={len(df[df['title'].str.contains(kw, case=False, regex=False, na=False)]):5d} taken={len(hit)}")
    rest = df[~df["job_id"].isin(taken)].head(N_RANDOM)
    out = pd.concat(picked + [rest])[KEEP]
    out["description"] = out["description"].str.slice(0, 4000)     # keep the repo small
    out["skills_desc"] = out["skills_desc"].fillna("").str.slice(0, 1000)
    out = out.sample(frac=1, random_state=SEED).head(2000).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
