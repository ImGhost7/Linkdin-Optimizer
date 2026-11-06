"""Profile scoring rubric.

Every section gets 0-100 = weighted sum of a few METRICS, each itself scored 0-100:

  completeness   are the expected fields present?
  length         is the text inside a recommended word/char range?   (trapezoid, see `band`)
  action_verbs   share of statements that start with a strong verb ("Built", "Led" ...)
  quantified     share of statements that contain a number / % / $ (measurable impact)
  keywords       how many distinct, recognised skills (from the taxonomy) appear
  readability    Flesch Reading Ease of the text (about) / statement length (experience)

The overall score is a weighted mean of the section scores (see SECTION_WEIGHTS). All numbers
below are heuristics grounded in common LinkedIn-profile guidance, NOT learned parameters -
they are kept in plain dictionaries on purpose so they are easy to explain and to change.
"""
from __future__ import annotations

import re

from ..nlp import get_nlp
from ..parser.models import Profile
from .models import MetricScore, SectionScore, SkillReport
from .skills import SOFT_CATEGORY, get_extractor

# ------------------------------------------------------------------ rubric
SECTION_WEIGHTS = {"headline": 0.15, "about": 0.25, "experience": 0.30,
                   "education": 0.10, "skills": 0.15, "certifications": 0.05}

# metric -> weight, per section (each row sums to 1.0; enforced by a unit test)
METRIC_WEIGHTS = {
    "headline":       {"completeness": 0.30, "length": 0.30, "keywords": 0.30, "clarity": 0.10},
    "about":          {"completeness": 0.15, "length": 0.20, "action_verbs": 0.15, "quantified": 0.15,
                       "keywords": 0.20, "readability": 0.15},
    "experience":     {"completeness": 0.25, "length": 0.15, "action_verbs": 0.20, "quantified": 0.20,
                       "keywords": 0.10, "readability": 0.10},
    "education":      {"completeness": 0.60, "detail": 0.40},
    "skills":         {"count": 0.35, "recognised": 0.20, "coverage": 0.30, "diversity": 0.15},
    "certifications": {"count": 0.70, "completeness": 0.30},
}

ACTION_VERBS = set("""
accelerated achieved adapted administered advised analysed analyzed architected assembled assessed
audited automated automate boosted brought built calculated captained championed collaborated
completed conceived conducted configured consolidated constructed consulted contributed converted
coordinated created cut debugged decreased defined delivered deployed designed detected developed
devised directed discovered documented doubled drove earned educated eliminated enabled engineered
enhanced established estimated evaluated exceeded executed expanded expedited experimented extended
extracted facilitated finalized fixed forecasted formulated founded generated grew guided handled
headed identified implemented improved increased initiated innovated inspected installed integrated
introduced investigated launched led leveraged maintained managed mapped mentored migrated minimized
modeled modelled modernized monitored negotiated normalized onboarded operated optimized orchestrated
organized outperformed overhauled owned partnered performed pioneered planned presented prioritized
produced programmed proposed prototyped published raised ran rebuilt recommended reduced refactored
refined released remodeled replaced reported researched resolved restructured revamped reviewed
rewrote saved scaled scripted secured selected served shipped simplified solved spearheaded standardized
streamlined strengthened structured supervised supported surpassed synthesized tailored taught tested
trained transformed translated troubleshot tuned unified upgraded validated visualized wrote won
build lead develop design create manage deliver drive implement improve launch analyze analyse
automate optimize optimise architect engineer mentor train research ship scale deploy
""".split())
# phrases that make a statement sound passive / vague
WEAK_STARTS = ("responsible for", "worked on", "helped with", "helped", "duties included", "tasked with",
               "involved in", "assisted with", "participated in")
_SKIP_LEADING = {"i", "we", "also", "successfully", "then", "and", "independently", "recently", "actively"}
_CTA = re.compile(r"\b(reach out|contact|connect with|email|e-mail|dm me|open to|looking for|get in touch|let's talk|@)\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])[$₹€£]?\d[\d,]*(?:\.\d+)?\s?(?:%|[kKmMbB]\b|x\b|\+|\s?(?:million|billion|thousand|lakh|crore))?")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


# ------------------------------------------------------------------ text utilities
def band(x: float, lo_bad: float, lo_ok: float, hi_ok: float, hi_bad: float) -> float:
    """Trapezoid membership in [0,1]: 0 outside (lo_bad, hi_bad), 1 inside [lo_ok, hi_ok], linear ramps between."""
    if x <= lo_bad or x >= hi_bad:
        return 0.0
    if x < lo_ok:
        return (x - lo_bad) / (lo_ok - lo_bad)
    if x <= hi_ok:
        return 1.0
    return (hi_bad - x) / (hi_bad - hi_ok)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][\w'+#./-]*", text))


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1                                          # silent final e
    return max(n, 1)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words). Higher = easier."""
    sents = _statements(text)
    words = re.findall(r"[A-Za-z']+", text)
    if not sents or not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syl / len(words))


def _statements(text: str) -> list[str]:
    """Split into statements: each line is one (bullets/PDF lines), long lines are sentence-split by spaCy."""
    nlp = get_nlp()
    out: list[str] = []
    for line in (ln.strip(" -•*\t") for ln in re.split(r"\n+", text)):
        if not line:
            continue
        if word_count(line) > 35:
            out += [s.text.strip() for s in nlp(line).sents if s.text.strip()]
        else:
            out.append(line)
    return out


def starts_with_action_verb(statement: str) -> bool:
    """True if the statement opens with a strong verb (after skipping 'I', 'successfully', ...)."""
    tokens = re.findall(r"[A-Za-z']+", statement)
    i = 0
    while i < len(tokens) - 1 and tokens[i].lower() in _SKIP_LEADING:
        i += 1
    if not tokens:
        return False
    return _is_action_verb(tokens[i])


def _is_action_verb(word: str) -> bool:
    w = word.lower()
    if w in ACTION_VERBS:
        return True
    # regular inflections: -s/-es, -d/-ed, -ing, -ied->-y, doubled consonant (-pped)
    stems = {w[:-1], w[:-2], w[:-3] + "y", w[:-3], w[:-3] + "e", w[:-4]} if len(w) > 4 else set()
    return any(s in ACTION_VERBS for s in stems if len(s) > 2) and w.endswith(("s", "ed", "ing", "d"))


def count_quantified(statement: str) -> int:
    """Number of measurable figures (50%, $2M, 3x, 10k users, 2M documents) - bare years don't count."""
    n = 0
    for m in _NUMBER.finditer(statement):
        tok = m.group().strip()
        if _YEAR.match(tok):
            continue
        n += 1
    return n


def _pct(k: int, n: int) -> float:
    return 100.0 * k / n if n else 0.0


def _metric(name: str, section: str, score01: float, detail: str, tip: str | None) -> MetricScore:
    score = round(max(0.0, min(1.0, score01)) * 100, 1)
    return MetricScore(name=name, score=score, weight=METRIC_WEIGHTS[section][name], detail=detail,
                       tip=tip if score < 70 else None)


def _combine(section: str, metrics: list[MetricScore], present: bool = True) -> SectionScore:
    score = sum(m.score * m.weight for m in metrics) if present else 0.0
    return SectionScore(section=section, score=round(score, 1), weight=SECTION_WEIGHTS[section],
                        present=present, metrics=metrics)


def _empty(section: str, tip: str) -> SectionScore:
    """A missing section scores 0 - with one actionable tip instead of per-metric noise."""
    ms = [MetricScore(name=n, score=0, weight=w, detail="section is empty", tip=tip if i == 0 else None)
          for i, (n, w) in enumerate(METRIC_WEIGHTS[section].items())]
    return _combine(section, ms, present=False)


# ------------------------------------------------------------------ section scorers
def score_headline(p: Profile) -> SectionScore:
    h = p.headline.strip()
    if not h:
        return _empty("headline", "Add a headline: 'Role | top skills | what you deliver'. It is the most-seen line of your profile.")
    ex = get_extractor()
    skills = ex.extract(h)
    segments = [s for s in re.split(r"[|•·/–—]| - ", h) if s.strip()]
    comp = 0.4 + (0.3 if len(segments) >= 2 else 0.0) + (0.3 if skills or re.search(r"engineer|developer|analyst|scientist|manager|designer|student|intern", h, re.I) else 0.0)
    kw = {0: 0.0, 1: 0.5, 2: 0.8}.get(len(skills), 1.0)
    letters = [c for c in h if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0
    clarity = 1.0 - (0.5 if caps > 0.5 else 0) - (0.3 if len(_EMOJI.findall(h)) > 2 else 0) - (0.2 if h.count("#") > 0 else 0)
    return _combine("headline", [
        _metric("completeness", "headline", comp, f"{len(segments)} segment(s); role/skill keyword {'found' if comp >= 0.7 else 'not found'}",
                "Structure it as 'Role | Skill | Skill | Impact' - separators make it scannable."),
        _metric("length", "headline", band(len(h), 10, 60, 150, 220), f"{len(h)} characters (ideal 60-150, max 220)",
                "Aim for 60-150 characters: long enough for keywords, short enough to read in a search result."),
        _metric("keywords", "headline", kw, f"{len(skills)} recognised skill(s) in headline: {', '.join(s.skill for s in skills[:4]) or 'none'}",
                "Put 2-3 of your strongest skills in the headline - recruiters search on them."),
        _metric("clarity", "headline", clarity, "no shouting/emoji clutter" if clarity >= 0.9 else "headline looks cluttered (ALL CAPS, emojis or hashtags)",
                "Avoid ALL CAPS, emoji strings and hashtags in the headline."),
    ])


def score_about(p: Profile) -> SectionScore:
    text = p.about.strip()
    if not text:
        return _empty("about", "Write an About section (150-300 words): who you are, what you've achieved, what you want next.")
    ex = get_extractor()
    stm = _statements(text)
    n_words = word_count(text)
    paras = [x for x in re.split(r"\n\s*\n", text) if x.strip()]
    skills = ex.extract(text)
    verbs = sum(starts_with_action_verb(s) or any(_is_action_verb(w) for w in re.findall(r"[A-Za-z']+", s)[:4]) for s in stm)
    quant = sum(count_quantified(s) > 0 for s in stm)
    fre = flesch_reading_ease(text)
    comp = (0.4 if n_words >= 40 else 0.15) + (0.3 if len(paras) >= 2 or len(stm) >= 4 else 0.0) + (0.3 if _CTA.search(text) else 0.0)
    return _combine("about", [
        _metric("completeness", "about", comp, f"{len(paras)} paragraph(s), {len(stm)} statement(s); call-to-action {'yes' if _CTA.search(text) else 'no'}",
                "Structure the About as: who you are -> proof of impact -> skills -> how to reach you / what you're open to."),
        _metric("length", "about", band(n_words, 20, 120, 350, 600), f"{n_words} words (ideal 120-350)",
                "Aim for 120-350 words - short Abouts miss keywords, very long ones don't get read."),
        _metric("action_verbs", "about", verbs / len(stm) if stm else 0, f"{verbs} of {len(stm)} statements use a strong action verb",
                "Open sentences with action verbs (Built, Led, Reduced) instead of 'I am' / 'responsible for'."),
        _metric("quantified", "about", {0: 0.0, 1: 0.6}.get(sum(count_quantified(s) for s in stm), 1.0),
                f"{quant} statement(s) contain a measurable number",
                "Add 2+ measurable results (%, $, users, time saved) to the About."),
        _metric("keywords", "about", min(len(skills) / 8, 1.0), f"{len(skills)} distinct recognised skills (target 8+)",
                "Mention 8+ relevant tools/skills by name so you show up in recruiter keyword searches."),
        _metric("readability", "about", band(fre, 10, 30, 70, 100), f"Flesch Reading Ease {fre:.0f} (target 30-70)",
                "Use shorter sentences and simpler words (Flesch 30-70 reads as professional but clear)."),
    ])


def score_experience(p: Profile) -> SectionScore:
    entries = p.experience
    if not entries:
        return _empty("experience", "Add your work experience (internships, freelance and projects count) with 3-5 impact bullets each.")
    ex = get_extractor()
    fields = [(bool(e.title), bool(e.company), bool(e.date_range), bool(e.description.strip())) for e in entries]
    fill = sum(sum(f) for f in fields) / (4 * len(entries))
    count_factor = min(1.0, 0.6 + 0.2 * len(entries))
    stm = [s for e in entries for s in _statements(e.description)]
    desc_words = [word_count(e.description) for e in entries]
    avg_words = sum(desc_words) / len(entries)
    verbs = sum(starts_with_action_verb(s) for s in stm)
    weak = sum(s.lower().startswith(WEAK_STARTS) for s in stm)
    quant = sum(count_quantified(s) > 0 for s in stm)
    skills = ex.extract(p.experience_text())
    good_len = sum(8 <= word_count(s) <= 35 for s in stm)
    return _combine("experience", [
        _metric("completeness", "experience", fill * count_factor,
                f"{len(entries)} role(s); {fill:.0%} of title/company/dates/description filled",
                "Fill title, company, dates and a description for every role; list at least 2-3 roles/projects."),
        _metric("length", "experience", band(avg_words, 5, 40, 120, 250), f"avg {avg_words:.0f} words per role (ideal 40-120)",
                "Give each role 40-120 words: 3-5 concise bullets on what you did and what changed."),
        _metric("action_verbs", "experience", verbs / len(stm) if stm else 0,
                f"{verbs} of {len(stm)} statements start with an action verb" + (f" ({weak} weak opener(s) like 'responsible for')" if weak else ""),
                "Start every bullet with a strong verb (Built, Led, Reduced); replace 'responsible for' / 'worked on' / 'helped with'."),
        _metric("quantified", "experience", quant / len(stm) if stm else 0, f"{quant} of {len(stm)} statements contain a number or metric",
                "Quantify impact: add %, $, users, volume or time saved to at least half of your bullets."),
        _metric("keywords", "experience", min(len(skills) / 8, 1.0), f"{len(skills)} distinct recognised skills across roles (target 8+)",
                "Name the specific tools/technologies you used in each role."),
        _metric("readability", "experience", good_len / len(stm) if stm else 0, f"{good_len} of {len(stm)} statements are 8-35 words",
                "Keep bullets to one idea of 8-35 words - too short says nothing, too long won't be read."),
    ])


def score_education(p: Profile) -> SectionScore:
    if not p.education:
        return _empty("education", "Add your education (degree, field of study, school, years).")
    comp = sum((bool(e.school) + bool(e.degree) + bool(e.dates)) / 3 for e in p.education) / len(p.education)
    detail = sum(("," in e.degree or " in " in e.degree.lower() or "-" in e.degree) for e in p.education) / len(p.education)
    return _combine("education", [
        _metric("completeness", "education", comp, f"{comp:.0%} of school/degree/dates filled across {len(p.education)} entr(ies)",
                "Add school, degree and start/end year for every education entry."),
        _metric("detail", "education", detail, "field of study " + ("present" if detail >= 0.5 else "missing"),
                "Include your field of study (e.g. 'B.Tech - Computer Science')."),
    ])


def score_skills(p: Profile, report: SkillReport) -> SectionScore:
    if not p.skills:
        return _empty("skills", "Add a Skills section - LinkedIn allows 50; list at least 15 that match the roles you want.")
    n = len(p.skills)
    recognised = len(report.listed_recognised)
    mentioned = {h.skill for h in report.mentioned}
    missing = len(report.missing_from_skills_section)
    coverage = (len(mentioned) - missing) / len(mentioned) if mentioned else 0.0
    cats = {c for c, k in report.categories.items() if k}
    has_soft = SOFT_CATEGORY in cats
    technical_cats = len(cats - {SOFT_CATEGORY})
    diversity = min(technical_cats / 2, 1.0) * 0.7 + (0.3 if has_soft else 0.0)
    pdf_hint = " (LinkedIn's PDF export only includes your top 3 skills - paste your full list for accurate scoring)" if n <= 3 else ""
    return _combine("skills", [
        _metric("count", "skills", band(n, 0, 15, 50, 51), f"{n} skill(s) listed (target 15-50){pdf_hint}",
                "List 15+ skills (up to 50), most relevant first."),
        _metric("recognised", "skills", recognised / n, f"{recognised} of {n} listed skills are standard, searchable skills",
                "Use standard skill names (e.g. 'Machine Learning' not 'ML stuff') so LinkedIn and recruiters can match them."),
        _metric("coverage", "skills", coverage, f"{len(mentioned) - missing} of {len(mentioned)} skills you mention in your text are listed in Skills",
                "Add the skills you mention in your experience to the Skills section: " + ", ".join(h.skill for h in report.missing_from_skills_section[:6])),
        _metric("diversity", "skills", diversity, f"{technical_cats} technical categor(ies); soft skills {'present' if has_soft else 'absent'}",
                "Balance technical skills across areas and add 2-3 soft skills (e.g. Leadership, Communication)."),
    ])


def score_certifications(p: Profile) -> SectionScore:
    if not p.certifications:
        return _empty("certifications", "Add 1-3 relevant certifications (cloud, data or role-specific) to boost credibility.")
    n = len(p.certifications)
    named = sum(len(c) >= 12 for c in p.certifications) / n
    return _combine("certifications", [
        _metric("count", "certifications", {1: 0.6, 2: 0.85}.get(n, 1.0), f"{n} certification(s)",
                "Add 2-3 recognised certifications relevant to your target role."),
        _metric("completeness", "certifications", named, f"{named:.0%} have a descriptive name",
                "Use the full official certificate name."),
    ])


# ------------------------------------------------------------------ overall
def grade(score: float) -> str:
    for cutoff, g in ((85, "A"), (70, "B"), (55, "C"), (40, "D")):
        if score >= cutoff:
            return g
    return "E"


def score_profile(p: Profile, report: SkillReport) -> tuple[float, list[SectionScore]]:
    sections = [score_headline(p), score_about(p), score_experience(p), score_education(p),
                score_skills(p, report), score_certifications(p)]
    overall = sum(s.score * s.weight for s in sections)
    return round(overall, 1), sections


def top_tips(sections: list[SectionScore], k: int = 5) -> list[str]:
    """The tips that would add the most points: (weight lost) = section weight * metric weight * (100 - score)."""
    cands = []
    for s in sections:
        for m in s.metrics:
            if m.tip:
                cands.append((s.weight * m.weight * (100 - m.score), f"[{s.section.title()}] {m.tip}"))
    cands.sort(key=lambda t: -t[0])
    return [t for _, t in cands[:k]]
