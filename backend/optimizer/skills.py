"""Skill extraction: spaCy PhraseMatcher over a bundled skills taxonomy.

Why PhraseMatcher + taxonomy (and not NER)? Skill names ("scikit-learn", "CI/CD", "C++") are not
standard NER entities, a closed vocabulary gives deterministic, explainable output, and
PhraseMatcher is O(text length) regardless of vocabulary size. Trade-off: it only finds skills that
are in the taxonomy - that's why the taxonomy also carries aliases ("k8s" -> "Kubernetes").

Matching rules
  * default: case-insensitive on the lower-cased token (attr="LOWER")
  * names in taxonomy["case_sensitive"] ("R", "Go", "Spark"...) and aliases of <=3 letters ("ML", "AI")
    must match exactly - otherwise "r", "go", "ai"/"pm" in ordinary prose would create false hits
  * overlapping matches are resolved longest-first (so "C" is not reported inside "C++")
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache

from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

from ..config import SKILLS_TAXONOMY_PATH
from ..nlp import get_nlp
from .models import SkillHit

SOFT_CATEGORY = "soft_skills"
# "Go to market", "Go ahead" are English, not the language: skip the case-sensitive hit if followed by these
_NOT_A_SKILL_IF_NEXT = {"Go": {"to", "ahead", "through", "live", "for", "beyond", "-", "above"}}


class SkillExtractor:
    def __init__(self, taxonomy_path=SKILLS_TAXONOMY_PATH):
        tax = json.loads(open(taxonomy_path, encoding="utf-8").read())
        self.nlp = get_nlp()
        self.category: dict[str, str] = {}               # canonical skill -> category
        for cat, names in tax["categories"].items():
            for n in names:
                self.category.setdefault(n, cat)
        self.canonical_lower = {n.lower(): n for n in self.category}
        self.aliases = {a.lower(): c for a, c in tax["aliases"].items() if c in self.category}
        case_sensitive = set(tax.get("case_sensitive", []))

        self._ci = PhraseMatcher(self.nlp.vocab, attr="LOWER")   # case-insensitive
        self._cs = PhraseMatcher(self.nlp.vocab)                 # exact (ORTH)
        make = self.nlp.make_doc                                 # tokenizer only: very fast
        for name in self.category:
            (self._cs if name in case_sensitive else self._ci).add(name, [make(name)])
        for alias, target in self.aliases.items():
            if len(alias) <= 3 and alias.isalpha():              # "ml", "ai", "pm", "js" -> must be written "ML", "AI"...
                self._cs.add(target, [make(alias.upper())])
            else:
                self._ci.add(target, [make(alias)])

    # -------------------------------------------------------------- public
    def canonicalise(self, skill: str) -> str | None:
        """Map a user-written skill ("ML", "react.js", "Postgres") to its canonical taxonomy name."""
        s = skill.strip()
        low = s.lower()
        if low in self.canonical_lower:
            return self.canonical_lower[low]
        if low in self.aliases:
            return self.aliases[low]
        hits = self.extract(s)
        return hits[0].skill if len(hits) == 1 else None

    def extract(self, text: str, source: str = "") -> list[SkillHit]:
        """Distinct skills found in `text`, most frequent first (ties: order of first appearance)."""
        if not text.strip():
            return []
        doc = self.nlp.make_doc(text)
        spans = [doc[s:e] for _, s, e in list(self._ci(doc)) + list(self._cs(doc))]
        spans = filter_spans(spans)                              # longest, non-overlapping
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for sp in spans:
            name = self._name_of(sp)
            nxt = doc[sp.end].lower_ if sp.end < len(doc) else ""
            if nxt in _NOT_A_SKILL_IF_NEXT.get(name, ()):
                continue
            counts[name] += 1
            first_seen.setdefault(name, sp.start)
        ordered = sorted(counts, key=lambda n: (-counts[n], first_seen[n]))
        return [SkillHit(skill=n, category=self.category[n], count=counts[n], source=source) for n in ordered]

    def _name_of(self, span) -> str:
        """Canonical skill for a matched span (it is either a canonical name or an alias)."""
        low = span.text.lower()
        if low in self.canonical_lower:
            return self.canonical_lower[low]
        if low in self.aliases:
            return self.aliases[low]
        for alias, target in self.aliases.items():               # upper-cased short alias ("ML")
            if len(alias) <= 3 and alias.upper() == span.text:
                return target
        # multi-token span whose lower-case form differs only by whitespace (e.g. "Node . js")
        return self.canonical_lower.get(low.replace(" ", ""), span.text)


@lru_cache(maxsize=1)
def get_extractor() -> SkillExtractor:
    return SkillExtractor()
