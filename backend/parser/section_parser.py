"""Turn raw profile text (from a LinkedIn PDF or pasted by the user) into a structured `Profile`.

Strategy (all rule-based, so it is deterministic and explainable):
  1. Split the text into sections by recognising heading lines ("Summary", "Experience", ...).
  2. Everything BEFORE the first heading is the "preamble" = name, headline, location.
  3. Each section has its own small parser (experience is the interesting one: it is anchored on
     date-range lines such as "March 2022 - Present (2 years 8 months)").
"""
from __future__ import annotations

import re

from .models import Education, Experience, Profile
from .pdf_reader import extract_pdf_text

# ---------------------------------------------------------------- headings
# normalised heading text -> canonical section key
HEADINGS: dict[str, str] = {
    "name": "name", "headline": "headline", "location": "location",
    "contact": "contact",
    "summary": "about", "about": "about", "professional summary": "about", "bio": "about",
    "experience": "experience", "work experience": "experience",
    "professional experience": "experience", "employment history": "experience",
    "education": "education",
    "top skills": "skills", "skills": "skills", "technical skills": "skills", "key skills": "skills",
    "core skills": "skills", "skills & endorsements": "skills",
    "certifications": "certifications", "certification": "certifications", "certificates": "certifications",
    "licenses & certifications": "certifications", "licenses and certifications": "certifications",
    "languages": "languages", "honors-awards": "other", "honors & awards": "other", "awards": "other",
    "publications": "other", "patents": "other", "projects": "other", "volunteer experience": "other",
    "courses": "other", "recommendations": "other", "interests": "other",
}
# sections whose content is a single value that may be written inline ("Headline: Data Scientist")
_INLINE_OK = {"name", "headline", "location", "about", "skills", "certifications"}

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE = rf"(?:{_MONTH}\.?,?\s+)?(?:19|20)\d{{2}}"
DATE_RANGE = re.compile(
    rf"(?P<start>{_DATE})\s*(?:[-–—]|to)\s*(?P<end>{_DATE}|Present|Current|Now)"
    rf"\)?(?:\s*[·(]\s*(?P<dur>[^)]*?)\)?)?\s*$", re.I)
_BULLET = re.compile(r"^\s*[•\-\*·▪●◦–]\s+")
_SENT_END = (".", "!", "?", ";", ":", ",")
# one-word lines that are the wrapped tail of a certification name
_CERT_TAILS = {"certificate", "certification", "professional", "specialization", "associate", "exam", "foundations"}

_DEGREE_WORDS = re.compile(
    r"\b(bachelor|master|b\.?\s?tech|m\.?\s?tech|b\.?\s?e\b|m\.?\s?e\b|b\.?\s?sc|m\.?\s?sc|bca|mca|mba|ph\.?d|"
    r"diploma|associate|degree|high school|a-levels|undergraduate|postgraduate|b\.?a\b|m\.?a\b)", re.I)
_SCHOOL_WORDS = re.compile(r"\b(university|college|institute|school|academy|iit|nit|iiit|polytechnic)\b", re.I)
_YEARS = re.compile(r"\(?\s*((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|Present)\s*\)?", re.I)


# ---------------------------------------------------------------- helpers
def _norm_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower().rstrip(":")).strip()


def split_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (preamble_lines, {section_key: lines}). Blank lines are kept (paragraph breaks)."""
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        norm = _norm_heading(line)
        key = HEADINGS.get(norm)
        # "Headline: Data Scientist | NLP"  -> heading + inline content
        if key is None and ":" in line:
            head, _, rest = line.partition(":")
            k2 = HEADINGS.get(_norm_heading(head))
            if k2 in _INLINE_OK and rest.strip():
                current = k2
                sections.setdefault(current, []).append(rest.strip())
                continue
        if key is not None and len(line) < 45:
            current = key
            sections.setdefault(current, [])
        elif current is None:
            preamble.append(line)
        else:
            sections[current].append(line)
    return preamble, sections


def _join_wrapped(lines: list[str]) -> str:
    """Re-join PDF-wrapped lines into prose, one statement per line.

    A new line starts at a bullet, or when the previous line ended a sentence (. ! ?) and this one
    starts with a capital/digit. Otherwise the PDF merely wrapped mid-sentence, so we join with a space.
    """
    out: list[str] = []
    for ln in (x.strip() for x in lines):
        if not ln:
            continue
        new_statement = _BULLET.match(ln) or (out and out[-1].endswith((".", "!", "?")) and (ln[0].isupper() or ln[0].isdigit()))
        if not out or new_statement:
            out.append(_BULLET.sub("", ln))
        else:
            out[-1] = f"{out[-1]} {ln}"
    return "\n".join(out)


def _paragraphs(lines: list[str]) -> str:
    """Join lines into text, using blank lines as paragraph separators."""
    paras, buf = [], []
    for ln in lines + [""]:
        if ln.strip():
            buf.append(ln)
        elif buf:
            # inside a paragraph, wrapped lines are simply joined with spaces unless bulleted
            paras.append(_join_wrapped(buf) if any(_BULLET.match(b) for b in buf) else " ".join(b.strip() for b in buf))
            buf = []
    return "\n\n".join(paras)


# ---------------------------------------------------------------- preamble
def _looks_like_location(line: str) -> bool:
    return len(line) < 60 and "|" not in line and (
        "," in line or bool(re.search(r"\b(Area|Region|Remote|India|United States|Kingdom|Canada)\b", line)))


def parse_preamble(lines: list[str]) -> tuple[str, str, str]:
    """name, headline, location from the lines above the first section heading."""
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return "", "", ""
    name, rest = lines[0], lines[1:]
    location = ""
    if len(rest) >= 2 and _looks_like_location(rest[-1]):
        location, rest = rest[-1], rest[:-1]
    return name, " ".join(rest).strip(), location


# ---------------------------------------------------------------- experience
def _looks_like_company(line: str) -> bool:
    """A company line is short, not a sentence and not a bullet."""
    return (0 < len(line) <= 70 and not line.endswith(_SENT_END) and not _BULLET.match(line)
            and not DATE_RANGE.search(line))


def _split_title_company(header: str, inline: bool = False) -> tuple[str, str]:
    """'Software Engineer at Google' -> ('Software Engineer', 'Google').

    With `inline=True` (title and dates share one line, so there is no separate company line) we also
    accept 'Intern, Beta Labs', 'Intern | Beta Labs' and 'Intern - Beta Labs'.
    """
    m = re.split(r"\s+(?:at|@)\s+", header, maxsplit=1)
    if len(m) == 1 and inline:
        m = re.split(r"\s*[,|]\s+|\s+[-–—]\s+", header, maxsplit=1)
    return (m[0].strip(), m[1].strip()) if len(m) == 2 else (header.strip(), "")


def parse_experience(lines: list[str]) -> list[Experience]:
    """Anchor on date-range lines; look backwards for title/company, forwards for the description."""
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return []
    date_idx = [i for i, ln in enumerate(lines) if DATE_RANGE.search(ln)]
    if not date_idx:                         # free text with no dates -> one undated entry
        return [Experience(description=_join_wrapped(lines))]

    # pass 1: work out where each entry's header starts and what its title/company are
    heads: list[dict] = []
    prev_company = ""
    for k, i in enumerate(date_idx):
        m = DATE_RANGE.search(lines[i])
        inline = lines[i][:m.start()].strip(" ()|·,-–—")
        floor = date_idx[k - 1] if k else -1          # never reach into the previous date line
        if inline:                                    # "Data Analyst, Acme Corp  (Jan 2020 - Present)"
            title, company = _split_title_company(inline, inline=True)
            start = i
        else:
            t_idx = i - 1
            if t_idx <= floor:                        # date line with nothing above it
                title, company, start = "", prev_company, i
            else:
                title, company = _split_title_company(lines[t_idx])
                start = t_idx
                c_idx = t_idx - 1
                if not company and c_idx > floor and _looks_like_company(lines[c_idx]) \
                        and (k == 0 or c_idx > floor + 1 or not lines[c_idx - 1].endswith(_SENT_END)):
                    company, start = lines[c_idx], c_idx
        company = company or prev_company            # several roles at one company share a header
        prev_company = company
        heads.append({"title": title, "company": company, "start": start, "date_i": i, "m": m})

    # pass 2: description = everything between this date line and the next entry's header
    out: list[Experience] = []
    for n, h in enumerate(heads):
        end = heads[n + 1]["start"] if n + 1 < len(heads) else len(lines)
        body = lines[h["date_i"] + 1:end]
        location = ""
        if body and _looks_like_location(body[0]) and not _BULLET.match(body[0]) and not body[0].endswith("."):
            location, body = body[0], body[1:]
        m = h["m"]
        out.append(Experience(
            title=h["title"], company=h["company"],
            date_range=f"{m.group('start')} - {m.group('end')}",
            duration=(m.group("dur") or "").strip(),
            location=location, description=_join_wrapped(body)))
    return out


# ---------------------------------------------------------------- education
def parse_education(lines: list[str]) -> list[Education]:
    entries: list[Education] = []
    for ln in (x.strip() for x in lines):
        if not ln:
            continue
        ym = _YEARS.search(ln)
        is_degree = bool(_DEGREE_WORDS.search(ln)) or "·" in ln or bool(ym)
        if is_degree and entries and not entries[-1].degree and not entries[-1].dates:
            cur = entries[-1]                         # degree line belongs to the school above
        elif is_degree:
            # single-line style: "B.Tech in CSE, XYZ University, 2020-2024"
            parts = [p.strip() for p in re.split(r",|·", ln) if p.strip()]
            schools = [p for p in parts if _SCHOOL_WORDS.search(p)]
            cur = Education(school=schools[0] if schools else "")
            entries.append(cur)
            if schools:
                ln = ", ".join(p for p in parts if p != schools[0])
        else:
            entries.append(Education(school=ln))
            continue
        if ym:
            cur.dates = f"{ym.group(1)} - {ym.group(2)}"
            ln = _YEARS.sub("", ln)
        cur.degree = ln.strip(" ·,()-–")
    return entries


# ---------------------------------------------------------------- skills & certifications
def parse_skills(lines: list[str]) -> list[str]:
    """One skill per line (PDF) or separated by , ; | • · (pasted). Order kept, duplicates dropped."""
    seen: dict[str, str] = {}
    for ln in lines:
        for part in re.split(r"[,;|•·●▪]|\s{2,}", ln):
            s = part.strip(" -*\t")
            if 1 < len(s) <= 60 and s.lower() not in seen:
                seen[s.lower()] = s
    return list(seen.values())


def parse_certifications(lines: list[str]) -> list[str]:
    """One certification per line, re-joining names that the PDF wrapped over two lines."""
    certs: list[str] = []
    for ln in (x.strip(" -*•·●▪\t") for x in lines):
        if not ln:
            continue
        wrapped = certs and (ln[0].islower() or ln.lower() in _CERT_TAILS
                             or certs[-1].lower().endswith((" and", " of", " &", " for", " in", "-", " to", " with")))
        if wrapped:
            certs[-1] = f"{certs[-1]} {ln}"
        else:
            certs.append(ln)
    return certs


# ---------------------------------------------------------------- public API
def parse_profile_text(text: str) -> Profile:
    """Parse pasted / extracted profile text into a Profile."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    preamble, sec = split_sections(text)
    name, headline, location = parse_preamble(preamble)
    # explicit labels ("Headline: ...") win over the positional guess
    name = " ".join(sec.get("name", [])).strip() or name
    headline = " ".join(x for x in sec.get("headline", []) if x.strip()).strip() or headline
    location = " ".join(sec.get("location", [])).strip() or location

    about = _paragraphs(sec.get("about", []))
    if not sec and not about:                        # no headings at all -> treat the text as the About
        about = _paragraphs(preamble)
        name = headline = location = ""

    return Profile(
        name=name, headline=headline, location=location, about=about,
        experience=parse_experience(sec.get("experience", [])),
        education=parse_education(sec.get("education", [])),
        skills=parse_skills(sec.get("skills", [])),
        certifications=parse_certifications(sec.get("certifications", [])),
        raw_text=text.strip(),
    )


def parse_profile_pdf(data: bytes) -> Profile:
    """Parse the bytes of a LinkedIn 'Save to PDF' export."""
    return parse_profile_text(extract_pdf_text(data))
