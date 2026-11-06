"""Load the bundled job-postings sample and pre-extract each job's required skills."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from ..optimizer.skills import get_extractor


def load_jobs(path: Path = config.JOBS_CSV_PATH) -> pd.DataFrame:
    """Read the CSV and add two derived columns:

    text    - what both the TF-IDF and embedding models see: title (repeated once, so it carries
              more weight), the explicit skills field, then the description.
    skills  - list of canonical taxonomy skills mentioned anywhere in `text` (ordered by frequency).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:  python data/build_jobs_sample.py")
    df = pd.read_csv(path, dtype={"job_id": str}).fillna("")
    df["text"] = (df["title"] + ". " + df["title"] + ". " + df["skills_desc"].astype(str) + " "
                  + df["description"].astype(str)).str.strip()
    ex = get_extractor()
    df["skills"] = [[h.skill for h in ex.extract(t)] for t in df["text"]]
    return df.reset_index(drop=True)
