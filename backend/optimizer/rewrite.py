"""Rule-based headline / About rewrites (the offline fallback when no LLM key is configured).

These are *templates filled with the user's own facts* (role, skills, years, measured achievements).
Nothing is invented: if the profile has no metric, the draft simply has no metric - and says so.
"""
from __future__ import annotations

import re
from datetime import date

from ..parser.models import Profile
from .models import RewriteSuggestions, SkillReport
from .scoring import _statements, count_quantified, starts_with_action_verb
from .skills import SOFT_CATEGORY


def infer_role(p: Profile) -> str:
    """Target role label: first segment of the headline, else most recent job title."""
    if p.headline.strip():
        first = re.split(r"\s*[|•·]\s*|\s+-\s+|\s+at\s+|\s+@\s+", p.headline.strip())[0].strip()
        if 2 < len(first) < 60:
            return first
    if p.experience and p.experience[0].title:
        return p.experience[0].title
    return "Professional"


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _to_month_index(text: str, default_month: int) -> int | None:
    """'March 2022' -> 2022*12+3 ; '2020' -> uses default_month ; 'Present' -> today."""
    if re.search(r"present|current|now", text, re.I):
        return date.today().year * 12 + date.today().month
    m = re.search(r"(?:([A-Za-z]{3})[a-z]*\.?,?\s+)?((?:19|20)\d{2})", text)
    if not m:
        return None
    month = _MONTHS.get((m.group(1) or "").lower(), default_month)
    return int(m.group(2)) * 12 + month


def years_of_experience(p: Profile) -> int:
    """Whole years from the earliest role start to the latest role end (month precision)."""
    starts, ends = [], []
    for e in p.experience:
        parts = re.split(r"\s+-\s+|\s+to\s+", e.date_range)
        if len(parts) == 2:
            s, en = _to_month_index(parts[0], 1), _to_month_index(parts[1], 12)
            if s and en:
                starts.append(s)
                ends.append(en)
    return max(0, (max(ends) - min(starts)) // 12) if starts else 0


def top_skills(p: Profile, report: SkillReport, k: int = 8, technical_only: bool = True) -> list[str]:
    """User-listed (recognised) skills first, then those mentioned in the text, by frequency."""
    cat = {h.skill: h.category for h in report.mentioned}
    out: list[str] = []
    for s in report.listed_recognised + [h.skill for h in report.mentioned]:
        if s not in out and not (technical_only and cat.get(s) == SOFT_CATEGORY):
            out.append(s)
    return out[:k]


def best_achievements(p: Profile, k: int = 2) -> list[str]:
    """Quantified, verb-led statements from the experience section (the strongest evidence you have)."""
    cands = []
    for e in p.experience:
        for s in _statements(e.description):
            if starts_with_action_verb(s) and count_quantified(s) and len(s) <= 200:
                cands.append((count_quantified(s), s.rstrip(".")))
    cands.sort(key=lambda t: -t[0])
    return [s for _, s in cands[:k]]


def rule_based_rewrite(p: Profile, report: SkillReport) -> RewriteSuggestions:
    role = infer_role(p)
    skills = top_skills(p, report)
    yrs = years_of_experience(p)
    s3, s2 = skills[:3], skills[:2]
    wins = best_achievements(p)

    options: list[str] = []
    if s3:
        options.append(f"{role} | {' | '.join(s3)}")
    if s2:
        exp = f"{yrs}+ years" if yrs >= 1 else "hands-on experience"
        options.append(f"{role} with {exp} in {' & '.join(s2)}")
    if s2 and wins and len(wins[0]) <= 70:
        options.append(f"{role} | {s2[0]} | {wins[0]}")
    if not options:
        options.append(f"{role} | add 2-3 key skills here | add one measurable result")
    options = [o[:220] for o in options]

    lead = f"{role} with {yrs}+ years of experience" if yrs >= 1 else role
    para1 = f"{lead} in {', '.join(s3[:-1])} and {s3[-1]}." if len(s3) >= 2 else f"{lead}."
    if wins:
        para2 = "Selected results:\n" + "\n".join(f"• {w}." for w in wins)
    else:
        para2 = "Selected results:\n• [Add 2-3 measurable achievements: what you built, for whom, and the % / $ / time impact.]"
    para3 = f"Core skills: {', '.join(skills[:8])}." if skills else "Core skills: [list your 6-8 strongest tools and skills]."
    para4 = "I'm open to new opportunities and happy to connect - feel free to message me."
    return RewriteSuggestions(
        source="rule-based", headline_options=options, about="\n\n".join([para1, para2, para3, para4]),
        note="Template draft built only from facts in your profile. Edit it into your own voice. "
             "Set GEMINI_API_KEY in .env for AI-written suggestions.")
