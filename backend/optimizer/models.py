"""Result schemas for the optimizer (also the API response shape)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    name: str                 # "length", "action_verbs", ...
    score: float              # 0-100
    weight: float             # share of the section score
    detail: str               # human-readable evidence, e.g. "3 of 5 bullets start with an action verb"
    tip: str | None = None    # advice, only present when the metric is weak


class SectionScore(BaseModel):
    section: str
    score: float              # 0-100 weighted sum of its metrics
    weight: float             # share of the overall score
    present: bool
    metrics: list[MetricScore] = Field(default_factory=list)


class SkillHit(BaseModel):
    skill: str
    category: str
    count: int = 1
    source: str = ""          # where it was found: "experience", "about", ...


class SkillReport(BaseModel):
    listed: list[str]                         # skills as written in the Skills section
    listed_recognised: list[str]              # ... that map to the taxonomy (canonical names)
    mentioned: list[SkillHit]                 # skills found in about/experience text
    missing_from_skills_section: list[SkillHit]
    categories: dict[str, int]                # category -> number of distinct skills mentioned/listed


class RewriteSuggestions(BaseModel):
    source: str                               # "gemini" or "rule-based"
    headline_options: list[str]
    about: str
    note: str = ""


class Analysis(BaseModel):
    overall_score: float
    grade: str
    sections: list[SectionScore]
    skills: SkillReport
    top_tips: list[str]
    rewrites: RewriteSuggestions
