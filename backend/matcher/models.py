"""Schemas for job matching (also the API response shape)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Weights(BaseModel):
    """Hybrid weights. They are re-normalised to sum to 1, so only their ratios matter."""
    embedding: float = Field(0.5, ge=0)
    tfidf: float = Field(0.2, ge=0)
    skills: float = Field(0.3, ge=0)

    def normalised(self) -> "Weights":
        total = self.embedding + self.tfidf + self.skills
        if total <= 0:
            raise ValueError("at least one weight must be > 0")
        return Weights(embedding=self.embedding / total, tfidf=self.tfidf / total, skills=self.skills / total)


class Components(BaseModel):
    embedding: float   # semantic similarity, 0-1 (min-max normalised over the job pool)
    tfidf: float       # keyword overlap, 0-1 (normalised the same way)
    skills: float      # share of the job's required skills the profile covers, 0-1 (normalised)


class JobMatch(BaseModel):
    rank: int
    job_id: str
    title: str
    company: str = ""
    location: str = ""
    url: str = ""
    score: float                      # 0-100, weighted sum of the normalised components
    components: Components
    raw_cosine: float                 # un-normalised embedding cosine, for transparency
    matched_skills: list[str]
    missing_skills: list[str]
    tips: list[str]                   # exactly 3
    snippet: str = ""
