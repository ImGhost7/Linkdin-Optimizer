"""Hybrid job matcher: semantic embeddings + TF-IDF keywords + skill overlap.

For one profile (the "query") and N jobs we compute three similarity vectors of length N:

  embedding : cosine(MiniLM(profile), MiniLM(job))        -> understands meaning ("ML" ~ "machine learning")
  tfidf     : cosine(TFIDF(profile), TFIDF(job))          -> exact-keyword overlap, rare words weigh more
  skills    : |profile skills ∩ job skills| / (|job skills| + K)
              = share of the job's requirements the candidate covers. The +K smoothing stops a posting
                that lists a single skill from scoring 100% just because you have that one skill.

The three scales differ (cosines of MiniLM are ~0.2-0.6, TF-IDF ones ~0.02-0.3), so before mixing
we min-max normalise each vector over the job pool for that query, then take the weighted sum:

  score = w_e * norm(embedding) + w_t * norm(tfidf) + w_s * norm(skills)        (weights sum to 1)

Because the normalisation is per query, scores say "how good is this job RELATIVE to the others
for this profile", not an absolute probability. The un-normalised cosine is returned as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import config
from ..optimizer.analyzer import build_skill_report
from ..optimizer.skills import get_extractor
from ..parser.models import Profile
from . import embeddings as emb
from .gap import distinctive_absent_terms, make_tips
from .models import Components, JobMatch, Weights

SKILL_SMOOTHING_K = 2.0


@dataclass
class Query:
    """A profile reduced to what the matcher needs."""
    text: str
    skills: set[str]
    headline: str = ""

    @classmethod
    def from_profile(cls, p: Profile) -> "Query":
        report = build_skill_report(p)
        return cls(text=p.full_text(), skills=set(report.listed_recognised) | {h.skill for h in report.mentioned},
                   headline=p.headline)


@dataclass
class RawScores:
    embedding: np.ndarray
    tfidf: np.ndarray
    skills: np.ndarray

    def normalised(self) -> "RawScores":
        return RawScores(*(_minmax(a) for a in (self.embedding, self.tfidf, self.skills)))


def _minmax(a: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    return np.clip((a - lo) / (hi - lo), 0, 1) if hi - lo > 1e-9 else np.zeros_like(a)


class HybridMatcher:
    def __init__(self, jobs: pd.DataFrame, job_embeddings: np.ndarray | None = None,
                 encode_fn: Callable[[list[str]], np.ndarray] | None = None):
        """`job_embeddings`/`encode_fn` can be injected (unit tests use a tiny fake encoder)."""
        self.jobs = jobs.reset_index(drop=True)
        self.encode = encode_fn or emb.encode
        self.job_emb = job_embeddings if job_embeddings is not None else emb.corpus_embeddings(list(self.jobs["text"]))
        self.job_skills: list[list[str]] = list(self.jobs["skills"])
        self.job_skill_sets = [set(s) for s in self.job_skills]
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8,
                                     sublinear_tf=True, max_features=60000)
        self.job_tfidf = self.tfidf.fit_transform(self.jobs["text"])
        self.vocab = list(self.tfidf.get_feature_names_out())

    # ----------------------------------------------------------- scoring
    def raw_scores(self, q: Query) -> RawScores:
        q_emb = self.encode([q.text])[0]
        e = np.clip(self.job_emb @ q_emb, 0, None)
        t = (self.job_tfidf @ self.tfidf.transform([q.text]).T).toarray().ravel()
        s = np.array([len(q.skills & js) / (len(js) + SKILL_SMOOTHING_K) for js in self.job_skill_sets])
        return RawScores(e, t, s)

    @staticmethod
    def combine(norm: RawScores, w: Weights) -> np.ndarray:
        w = w.normalised()
        return w.embedding * norm.embedding + w.tfidf * norm.tfidf + w.skills * norm.skills

    # ----------------------------------------------------------- public API
    def rank(self, q: Query, top_k: int = 10, weights: Weights | None = None) -> list[JobMatch]:
        weights = weights or Weights(**config.DEFAULT_WEIGHTS)
        raw = self.raw_scores(q)
        norm = raw.normalised()
        final = self.combine(norm, weights)
        top = np.argsort(-final)[:top_k]
        return [self._build(i, rank + 1, q, final[i], norm, raw) for rank, i in enumerate(top)]

    def match_job_description(self, q: Query, jd_text: str, title: str = "Pasted job description",
                              weights: Weights | None = None) -> JobMatch:
        """Score ONE pasted job description. It is normalised against the bundled pool, so its score
        is comparable with the dataset matches ('would this JD rank near the top for me?')."""
        weights = weights or Weights(**config.DEFAULT_WEIGHTS)
        pool = self.raw_scores(q)
        jd_skills = [h.skill for h in get_extractor().extract(jd_text)]
        jd_emb = self.encode([jd_text])[0]
        e = max(float(jd_emb @ self.encode([q.text])[0]), 0.0)
        tf = float((self.tfidf.transform([jd_text]) @ self.tfidf.transform([q.text]).T).toarray()[0, 0])
        sk = len(q.skills & set(jd_skills)) / (len(jd_skills) + SKILL_SMOOTHING_K)
        comp = RawScores(*(np.array([v]) for v in (e, tf, sk)))
        norm = RawScores(*(_minmax(c, p.min(), p.max()) for c, p in zip((comp.embedding, comp.tfidf, comp.skills),
                                                                        (pool.embedding, pool.tfidf, pool.skills))))
        score = float(self.combine(norm, weights)[0])
        absent = distinctive_absent_terms(self.tfidf.transform([jd_text]), self.vocab, q.text, exclude=title)
        return self._assemble(0, "pasted-jd", title, "", "", "", score, norm, e, 0, jd_skills, q, absent, jd_text[:300])

    # ----------------------------------------------------------- helpers
    def _build(self, i: int, rank: int, q: Query, score: float, norm: RawScores, raw: RawScores) -> JobMatch:
        row = self.jobs.iloc[i]
        absent = distinctive_absent_terms(self.job_tfidf[i], self.vocab, q.text, exclude=str(row["title"]))
        return self._assemble(rank, str(row["job_id"]), str(row["title"]), str(row.get("company_name", "")),
                              str(row.get("location", "")), str(row.get("job_posting_url", "")), float(score), norm,
                              float(raw.embedding[i]), i, self.job_skills[i], q, absent, str(row["description"])[:300])

    @staticmethod
    def _assemble(rank, job_id, title, company, location, url, score, norm, raw_cos, i, job_skills, q, absent, snippet):
        matched = [s for s in job_skills if s in q.skills]
        missing = [s for s in job_skills if s not in q.skills]
        return JobMatch(
            rank=rank, job_id=job_id, title=title, company=company, location=location, url=url,
            score=round(score * 100, 1),
            components=Components(embedding=round(float(norm.embedding[i]), 3), tfidf=round(float(norm.tfidf[i]), 3),
                                  skills=round(float(norm.skills[i]), 3)),
            raw_cosine=round(raw_cos, 3), matched_skills=matched, missing_skills=missing,
            tips=make_tips(title, matched, missing, q.headline, absent), snippet=snippet)
