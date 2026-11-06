"""Extract text from a LinkedIn "Save to PDF" export.

LinkedIn's PDF has a two-column layout: a narrow left sidebar (Contact, Top Skills, Languages,
Certifications) and a wide main column (name, headline, Summary, Experience, Education). Plain
`extract_text()` interleaves the two columns line-by-line, which scrambles sections. So we detect
the column split from the x-position of the main-column headings and read each column separately
(main column first, then sidebar), for every page.
"""
from __future__ import annotations

import io
import re

import pdfplumber

# Headings that live in the MAIN column of a LinkedIn PDF; their x-position tells us where it starts.
_MAIN_HEADINGS = {"summary", "about", "experience", "education"}
_FOOTER_RE = re.compile(r"^\s*Page \d+ of \d+\s*$", re.I)


def _column_split(page) -> float | None:
    """x-coordinate where the main column starts, or None if the page is single-column."""
    words = page.extract_words()
    xs = [w["x0"] for w in words if w["text"].strip().lower() in _MAIN_HEADINGS]
    if not xs:
        return None
    split = min(xs)
    # A real sidebar leaves a clear margin on the left (>18% of page width).
    return split if split > 0.18 * page.width else None


def extract_pdf_text(data: bytes) -> str:
    """Return the text of the PDF in reading order (all main-column pages, then all sidebar pages)."""
    main_chunks: list[str] = []   # main column of every page, in order
    side_chunks: list[str] = []   # sidebar of every page, in order
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # Use page 1 to decide the layout, since later pages may have no headings at all.
        split = _column_split(pdf.pages[0]) if pdf.pages else None
        for page in pdf.pages:
            if split:
                main = page.crop((split - 6, 0, page.width, page.height)).extract_text() or ""
                side = page.crop((0, 0, split - 6, page.height)).extract_text() or ""
                main_chunks.append(main)
                side_chunks.append(side)
            else:
                main_chunks.append(page.extract_text() or "")
    # All main-column pages first, THEN the sidebar: otherwise page-2 experience text would be
    # appended to whichever sidebar section (e.g. Certifications) came last on page 1.
    text = "\n".join(main_chunks + side_chunks)
    lines = [ln.rstrip() for ln in text.splitlines() if not _FOOTER_RE.match(ln)]
    return "\n".join(lines).strip()
