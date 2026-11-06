"""Facade: parsed Profile in -> full Analysis out (skills report + scores + rewrite suggestions)."""
from __future__ import annotations

from collections import Counter

from ..parser.models import Profile
from .llm import gemini_rewrite
from .models import Analysis, SkillHit, SkillReport
from .rewrite import rule_based_rewrite
from .scoring import grade, score_profile, top_tips
from .skills import SOFT_CATEGORY, get_extractor


def build_skill_report(p: Profile) -> SkillReport:
    """Compare the skills the user LISTED with the skills their text actually MENTIONS."""
    ex = get_extractor()
    listed_canon: dict[str, str] = {}                    # canonical name -> text as the user wrote it
    for raw in p.skills:
        c = ex.canonicalise(raw)
        if c:
            listed_canon.setdefault(c, raw)

    merged: dict[str, SkillHit] = {}
    # experience first so that it is reported as the "source" when a skill appears in several places
    for source, text in (("experience", p.experience_text()), ("about", p.about), ("headline", p.headline)):
        for h in ex.extract(text, source=source):
            if h.skill in merged:
                merged[h.skill].count += h.count
            else:
                merged[h.skill] = h
    mentioned = sorted(merged.values(), key=lambda h: -h.count)

    # Missing = used in experience/about (a headline-only mention is not evidence) but not in Skills.
    missing = [h for h in mentioned if h.skill not in listed_canon and h.source in ("experience", "about")]
    missing.sort(key=lambda h: (h.category == SOFT_CATEGORY, -h.count))   # technical gaps first

    cats = Counter(ex.category[s] for s in set(listed_canon) | set(merged))
    return SkillReport(listed=p.skills, listed_recognised=list(listed_canon), mentioned=mentioned,
                       missing_from_skills_section=missing, categories=dict(cats))


def analyze_profile(p: Profile, use_llm: bool = True) -> Analysis:
    report = build_skill_report(p)
    overall, sections = score_profile(p, report)
    rewrites = (gemini_rewrite(p, report) if use_llm else None) or rule_based_rewrite(p, report)
    return Analysis(overall_score=overall, grade=grade(overall), sections=sections, skills=report,
                    top_tips=top_tips(sections), rewrites=rewrites)
