"""Run the offline evaluation and write docs/RESULTS.md (+ data/eval/*.json).

    python -m backend.eval.run_eval              # uses data/eval/labelled_set.json (built on first run)
    python -m backend.eval.run_eval --rebuild    # regenerate the synthetic labelled set first

Compared systems (all use the SAME scoring code path as the app: raw scores -> per-query min-max -> weights):
    Random baseline | TF-IDF only | Embeddings only | Skills only | Hybrid (default weights)
Metrics: Precision@5, Recall@10, MRR, NDCG@10 over each profile's labelled candidate pool.
Extras: 95% bootstrap CIs over profiles, explicit-vs-obfuscated breakdown, a weight sweep, and a
2-fold cross-validated weight choice so the "tuned" number is not fitted on the data it is scored on.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np

from .. import config
from ..matcher.hybrid import HybridMatcher, Query, RawScores
from ..matcher.jobs import load_jobs
from ..matcher.models import Weights
from ..parser.models import Profile
from .metrics import all_metrics
from .synthetic import N_G1, N_G2, N_HARD, N_RANDOM, SEED, build_labelled_set

EVAL_DIR = config.DATA_DIR / "eval"
SET_PATH = EVAL_DIR / "labelled_set.json"
METRIC_NAMES = ["P@5", "R@10", "MRR", "NDCG@10"]
N_BOOT, N_RANDOM_PERMS = 2000, 200
# The blend chosen before any result was seen. The shipped default was later changed to the weights that the
# cross-validated sweep preferred, so the preset is evaluated too, as an un-tuned reference point.
PRESET = Weights(embedding=0.5, tfidf=0.2, skills=0.3)


def _same(a: Weights, b: Weights) -> bool:
    return all(abs(getattr(a.normalised(), k) - getattr(b.normalised(), k)) < 1e-9 for k in ("embedding", "tfidf", "skills"))


def load_or_build_set(jobs, rebuild: bool) -> list[dict]:
    if SET_PATH.exists() and not rebuild:
        return json.loads(SET_PATH.read_text(encoding="utf-8"))
    data = build_labelled_set(jobs)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SET_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def rank_grades(scores: np.ndarray, pool_idx: np.ndarray, pool_grades: np.ndarray) -> list[int]:
    """Grades of the candidate pool in the order a system would rank it (stable on ties)."""
    order = np.argsort(-scores[pool_idx], kind="stable")
    return pool_grades[order].tolist()


def evaluate(matcher: HybridMatcher, labelled: list[dict]):
    id_to_row = {str(j): i for i, j in enumerate(matcher.jobs["job_id"])}
    rng = random.Random(SEED)
    default_w = Weights(**config.DEFAULT_WEIGHTS)
    methods = {"TF-IDF only": Weights(embedding=0, tfidf=1, skills=0),
               "Embeddings only": Weights(embedding=1, tfidf=0, skills=0),
               "Skills only": Weights(embedding=0, tfidf=0, skills=1),
               f"Hybrid ({default_w.embedding:.1f} emb / {default_w.tfidf:.1f} tfidf / {default_w.skills:.1f} skills)": default_w}
    if not _same(default_w, PRESET):       # keep the weights that were fixed before any result was seen as a reference row
        methods["Preset blend (0.5 emb / 0.2 tfidf / 0.3 skills)"] = PRESET
    grid = [Weights(embedding=a / 4, tfidf=b / 4, skills=(4 - a - b) / 4) for a, b in itertools.product(range(5), repeat=2) if a + b <= 4]

    per = {m: [] for m in ["Random baseline", *methods]}          # method -> [metric dict per profile]
    sweep = [[] for _ in grid]                                      # grid combo -> [metric dict per profile]
    for item in labelled:
        q = Query.from_profile(Profile(**item["profile"]))
        norm = matcher.raw_scores(q).normalised()
        ids = list(item["judgments"])
        pool_idx = np.array([id_to_row[i] for i in ids])
        pool_grades = np.array([item["judgments"][i] for i in ids])
        all_g = pool_grades.tolist()
        for name, w in methods.items():
            per[name].append(all_metrics(rank_grades(matcher.combine(norm, w), pool_idx, pool_grades), all_g))
        for gi, w in enumerate(grid):
            sweep[gi].append(all_metrics(rank_grades(matcher.combine(norm, w), pool_idx, pool_grades), all_g))
        # random baseline = expectation over many shuffles of the pool
        shuffles = []
        for _ in range(N_RANDOM_PERMS):
            g = pool_grades.copy()
            np.random.default_rng(rng.randrange(2 ** 31)).shuffle(g)
            shuffles.append(all_metrics(g.tolist(), all_g))
        per["Random baseline"].append({k: float(np.mean([s[k] for s in shuffles])) for k in METRIC_NAMES})
    return per, grid, sweep


def arr(rows: list[dict], metric: str | None = None) -> np.ndarray:
    return np.array([[r[m] for m in METRIC_NAMES] for r in rows]) if metric is None else np.array([r[metric] for r in rows])


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(x: float) -> str:
    return f"{x:.3f}"


def results_table(per, rng) -> str:
    lines = ["| System | " + " | ".join(METRIC_NAMES) + " |", "|---|" + "---|" * len(METRIC_NAMES)]
    best = {m: max(arr(per[s], m).mean() for s in per if s != "Random baseline") for m in METRIC_NAMES}
    for system, rows in per.items():
        cells = []
        for m in METRIC_NAMES:
            x = arr(rows, m)
            lo, hi = boot_ci(x, rng)
            cell = f"{fmt(x.mean())} <sub>[{fmt(lo)}, {fmt(hi)}]</sub>"
            cells.append(f"**{cell}**" if system != "Random baseline" and abs(x.mean() - best[m]) < 1e-12 else cell)
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def subset_table(per, labelled) -> str:
    groups = {"Explicit-title profiles": [i for i, x in enumerate(labelled) if not x["obfuscated"]],
              "Obfuscated profiles": [i for i, x in enumerate(labelled) if x["obfuscated"]]}
    lines = ["| Subset | System | " + " | ".join(METRIC_NAMES) + " |", "|---|---|" + "---|" * len(METRIC_NAMES)]
    for gname, idx in groups.items():
        for system, rows in per.items():
            sub = [rows[i] for i in idx]
            lines.append(f"| {gname} (n={len(idx)}) | {system} | " + " | ".join(fmt(arr(sub, m).mean()) for m in METRIC_NAMES) + " |")
    return "\n".join(lines)


def paired_table(per, rng) -> str:
    hyb = next(k for k in per if k.startswith("Hybrid"))
    lines = ["| Comparison (NDCG@10) | Mean difference | 95% CI | Reading |", "|---|---|---|---|"]
    for other in ("TF-IDF only", "Embeddings only", "Skills only"):
        d = arr(per[hyb], "NDCG@10") - arr(per[other], "NDCG@10")
        lo, hi = boot_ci(d, rng)
        verdict = "hybrid better" if lo > 0 else "other better" if hi < 0 else "no clear difference (CI includes 0)"
        lines.append(f"| Hybrid − {other} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    return "\n".join(lines)


def sweep_section(grid, sweep, per) -> tuple[str, dict]:
    ndcg = np.array([arr(rows, "NDCG@10") for rows in sweep])        # (combos, profiles)
    n = ndcg.shape[1]
    folds = [np.arange(n)[::2], np.arange(n)[1::2]]
    held_out, chosen = [], []
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        best = int(np.argmax(ndcg[:, train].mean(axis=1)))
        chosen.append(grid[best])
        held_out.append(ndcg[best, test])
    cv = float(np.concatenate(held_out).mean())
    hyb = next(k for k in per if k.startswith("Hybrid"))
    default = float(arr(per[hyb], "NDCG@10").mean())
    order = np.argsort(-ndcg.mean(axis=1))
    rows = ["| Embedding | TF-IDF | Skills | P@5 | NDCG@10 |", "|---|---|---|---|---|"]
    for gi in order[:8]:
        w = grid[gi]
        rows.append(f"| {w.embedding:.2f} | {w.tfidf:.2f} | {w.skills:.2f} | {fmt(arr(sweep[gi], 'P@5').mean())} | {fmt(ndcg[gi].mean())} |")
    ch = " and ".join(f"({w.embedding:.2f}/{w.tfidf:.2f}/{w.skills:.2f})" for w in chosen)
    txt = (f"{chr(10).join(rows)}\n\n_Top 8 of {len(grid)} weight combinations on a 0.25 grid, scored on the full set - "
           f"this is optimistic because the best row is picked on the same data it is scored on._\n\n"
           f"**Cross-validated check.** Choosing weights on one half of the profiles and scoring on the other half "
           f"(2-fold; chosen combos {ch}) gives held-out NDCG@10 = **{cv:.3f}**. The shipped default "
           f"({config.DEFAULT_WEIGHTS['embedding']:.2f}/{config.DEFAULT_WEIGHTS['tfidf']:.2f}/{config.DEFAULT_WEIGHTS['skills']:.2f}) scores {default:.3f} "
           f"on the full set" + ("" if _same(Weights(**config.DEFAULT_WEIGHTS), PRESET) else
           ", but it was picked after seeing this sweep, so that number is optimistic; the held-out figure is the fairer estimate "
           "of how such weights would generalise") + ".")
    return txt, {"cv_ndcg10": cv, "default_ndcg10": default}


def write_report(per, grid, sweep, labelled, matcher, runtime: float, path) -> dict:
    rng = np.random.default_rng(SEED)
    sweep_txt, sweep_info = sweep_section(grid, sweep, per)
    n_explicit = sum(not x["obfuscated"] for x in labelled)
    md = f"""# Evaluation results

> Generated by `python -m backend.eval.run_eval` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. **Do not edit by hand** - re-run the script.

## Setup

| | |
|---|---|
| Job corpus | {len(matcher.jobs):,} real LinkedIn job postings (see `data/README.md`) |
| Labelled profiles | **{len(labelled)} SYNTHETIC profiles** (10 role families x 3; {n_explicit} state their role, {len(labelled) - n_explicit} are "obfuscated") |
| Candidate pool per profile | 100 jobs = {N_G2} grade-2 (own role family) + {N_G1} grade-1 (adjacent family) + {N_HARD} hard negatives (other tech roles) + {N_RANDOM} random |
| Relevance label | rule on the job **title** (regex per role family) - a proxy, not human judgement |
| Relevant = | grade >= 1 for P@5 / R@10 / MRR; graded gain 2^g - 1 for NDCG@10 |
| Embedding model | `{config.EMBEDDING_MODEL}` |
| Hybrid weights | embedding {config.DEFAULT_WEIGHTS['embedding']:.2f} / TF-IDF {config.DEFAULT_WEIGHTS['tfidf']:.2f} / skills {config.DEFAULT_WEIGHTS['skills']:.2f} ({'fixed in advance' if _same(Weights(**config.DEFAULT_WEIGHTS), PRESET) else 'chosen after the weight sweep below, so optimistic on this set; the original 0.5/0.2/0.3 blend is listed as "Preset blend"'}) |
| Seeds | profile/pool sampling {SEED}; bootstrap {N_BOOT} resamples; random baseline = mean of {N_RANDOM_PERMS} shuffles |
| Environment | Python {platform.python_version()}, {platform.system()}; evaluation took {runtime:.0f}s |

## Main results (all {len(labelled)} profiles)

Mean per profile, with 95% bootstrap confidence interval over profiles in <sub>[low, high]</sub>. Bold marks the highest mean per metric - **not** a statistically significant win; check the paired comparison below.

{results_table(per, rng)}

### Is the hybrid actually better? (paired bootstrap on NDCG@10)

{paired_table(per, rng)}

## Breakdown: explicit vs obfuscated profiles

Obfuscated profiles have generic job titles and a headline that names no role, so the title keywords a TF-IDF model relies on are missing.

{subset_table(per, labelled)}

## Weight sensitivity

{sweep_txt}

## How to read these numbers (limitations)

* **Synthetic profiles.** They are generated from templates, so they are cleaner and more "on-topic" than real profiles. Absolute numbers will be optimistic; the *comparison between systems* is the meaningful part.
* **Title-based relevance labels.** A job counts as relevant if its *title* matches the profile's role family. The matchers also read the title, which favours keyword methods. There is no human annotation.
* **Small pool.** Each profile is ranked against 100 candidates, not the whole corpus, and Recall@10 is measured against the {N_G2 + N_G1} relevant jobs in that pool. Metrics from a different pool size are not comparable.
* **n = {len(labelled)}.** Confidence intervals are wide; differences smaller than the CI are not evidence of a real gap.
* **Skills-only scores** depend on the bundled taxonomy; a skill outside it is invisible to that component.
"""
    path.write_text(md, encoding="utf-8")
    summary = {sys: {m: float(arr(rows, m).mean()) for m in METRIC_NAMES} for sys, rows in per.items()}
    return {"summary": summary, "sweep": sweep_info}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="regenerate the synthetic labelled set")
    ap.add_argument("--out", default=str(config.DOCS_DIR / "RESULTS.md"))
    args = ap.parse_args()

    t0 = time.time()
    print("loading jobs + embeddings (first run builds the embedding cache) ...", flush=True)
    jobs = load_jobs()
    labelled = load_or_build_set(jobs, args.rebuild)
    matcher = HybridMatcher(jobs)
    print(f"evaluating {len(labelled)} profiles ...", flush=True)
    per, grid, sweep = evaluate(matcher, labelled)
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = write_report(per, grid, sweep, labelled, matcher, time.time() - t0, out)
    (EVAL_DIR / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for system, m in result["summary"].items():
        print(f"  {system:55s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
