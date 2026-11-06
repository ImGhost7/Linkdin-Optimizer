"""Skill-gap analysis and the 3 concrete tips shown with every job match.

The tips are templates, but each one is filled with job-specific evidence (the missing skills,
the job title, the job's most distinctive keywords absent from the profile), so no two matches
read the same.
"""
from __future__ import annotations

import re

from ..optimizer.skills import get_extractor

# What a good "first step" looks like for each kind of skill gap.
CATEGORY_ACTIONS = {
    "cloud_devops": "a small project deployed on a free cloud tier (or an entry-level cloud certification)",
    "machine_learning": "an end-to-end project or Kaggle notebook that uses it",
    "ml_libraries": "a notebook or small repo that uses it, published on GitHub",
    "data_engineering_analytics": "a dashboard or pipeline built on a public dataset",
    "databases": "a project with a real schema and a few non-trivial queries",
    "web_frontend": "a small deployed web app on GitHub",
    "backend_frameworks": "a small API with tests, published on GitHub",
    "programming_languages": "a few practice projects or an open-source contribution in it",
    "software_engineering": "a documented example in a side project or code-review practice",
    "security": "a hands-on lab (e.g. an intro CTF) and a short write-up",
    "product_business": "a short case study or one-page write-up of a decision you drove",
    "marketing_design": "a mini portfolio piece or campaign case study",
    "soft_skills": "a concrete example (situation, action, result) in your experience bullets",
}
_TITLE_NOISE = {"senior", "sr", "jr", "junior", "lead", "principal", "staff", "i", "ii", "iii", "iv", "the", "and",
                "of", "for", "a", "an", "to", "in", "at", "with", "intern", "associate", "remote", "hybrid"}
_STOP = set("""about above after again against all also and any are because been before being below between both
but can could did does doing down during each few for from further had has have having here how into its itself
just more most must need not now off once only other our out over own same should some such than that the their
them then there these they this those through too under until very was were what when where which while who whom
why will with would you your work working team experience ability job role company skills required including
years year strong good great preferred requirements responsibilities qualifications position candidate etc""".split())


def distinctive_absent_terms(job_tfidf_row, vocab: list[str], profile_text: str, k: int = 3,
                             exclude: str = "") -> list[str]:
    """Highest-TF-IDF single words of the job that never occur in the profile (and are not boilerplate).
    `exclude` = extra text whose words are skipped too (the job title: tip 2 already covers it)."""
    profile_words = set(re.findall(r"[a-z][a-z+#.-]*", f"{profile_text} {exclude}".lower()))
    row = job_tfidf_row.tocoo()
    order = sorted(zip(row.col, row.data), key=lambda t: -t[1])
    out: list[str] = []
    for col, _ in order:
        term = vocab[col]
        if " " in term or len(term) < 4 or term in _STOP or term in profile_words or not term.isalpha():
            continue
        out.append(term)
        if len(out) == k:
            break
    return out


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def make_tips(job_title: str, matched: list[str], missing: list[str], profile_headline: str,
              absent_terms: list[str]) -> list[str]:
    """Exactly three tips: (1) close the skill gap, (2) position the profile, (3) mirror the keywords."""
    ex = get_extractor()
    # 1 ---- skill gap
    if missing:
        top = missing[:3]
        action = CATEGORY_ACTIONS.get(ex.category.get(top[0], ""), "a small project that uses it")
        t1 = (f"Close the skill gap: this role asks for {_join(top)}, which your profile doesn't show. "
              f"Build {action} and add it to your Experience/Projects and Skills sections.")
    else:
        t1 = ("You cover every skill we detected in this posting - make that obvious by listing them in your "
              "Skills section (top 3 are shown first) and naming them in your About.")
    # 2 ---- positioning
    title_words = [w for w in re.findall(r"[a-z+#]+", job_title.lower()) if w not in _TITLE_NOISE]
    headline_l = profile_headline.lower()
    if title_words and not any(w in headline_l for w in title_words):
        lead = " | ".join([job_title.strip()] + matched[:2])
        t2 = f"Align your headline with this role, e.g. \"{lead}\" - only if it honestly describes you."
    elif matched:
        t2 = (f"Lead your About with your strongest matches ({_join(matched[:3])}); recruiters often read only "
              "the first 2-3 lines.")
    else:
        t2 = "Rewrite the first two lines of your About to state the role you want and your 2-3 most relevant skills."
    # 3 ---- keywords / evidence
    if absent_terms:
        t3 = (f"Mirror the posting's language where it is truthful: \"{_join(absent_terms)}\" appear prominently in "
              "the job description but nowhere in your profile.")
    else:
        t3 = ("Back your claims with numbers: quantify your two most relevant achievements for this role "
              "(%, $, users, time saved).")
    return [t1, t2, t3]
