"""Shared spaCy pipeline loader.

We prefer `en_core_web_sm` (gives POS tags + lemmas, used for action-verb detection). If the model
is not installed we fall back to a blank English pipeline with a rule-based sentence splitter, so
the app and the tests still run - just with slightly weaker verb detection.
"""
from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def has_pos_tagger() -> bool:
    return "tagger" in get_nlp().pipe_names
