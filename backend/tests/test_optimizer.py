"""Optimizer tests: skill extraction, rubric maths, scoring behaviour, rewrites and the LLM fallback."""
import json
import re
from pathlib import Path

import httpx
import pytest

from backend import config
from backend.optimizer import llm
from backend.optimizer.analyzer import analyze_profile, build_skill_report
from backend.optimizer.rewrite import rule_based_rewrite, years_of_experience
from backend.optimizer.scoring import (METRIC_WEIGHTS, SECTION_WEIGHTS, band, count_quantified,
                                       flesch_reading_ease, starts_with_action_verb)
from backend.optimizer.skills import get_extractor
from backend.parser.models import Experience, Profile
from backend.parser.section_parser import parse_profile_pdf

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "sample_profiles"


# ------------------------------------------------------------------ taxonomy & extraction
def test_taxonomy_integrity():
    tax = json.loads(config.SKILLS_TAXONOMY_PATH.read_text(encoding="utf-8"))
    names = {s for v in tax["categories"].values() for s in v}
    assert len(names) >= 500
    assert all(target in names for target in tax["aliases"].values()), "alias points to unknown skill"
    assert all(cs in names for cs in tax["case_sensitive"])


def names(text):
    return {h.skill for h in get_extractor().extract(text)}


def test_extracts_aliases_and_multiword():
    got = names("Deployed ML models with k8s, CI/CD and scikit-learn; built REST APIs in Node.js.")
    assert {"Machine Learning", "Kubernetes", "CI/CD", "scikit-learn", "REST APIs", "Node.js"} <= got


def test_cplusplus_does_not_also_report_c():
    assert names("Wrote C++ services") == {"C++"}


def test_ambiguous_words_are_not_skills():
    got = names("The rest of the team will go to market in spring. I like to express myself; pm meeting at 5 pm.")
    assert got == set()


def test_case_sensitive_short_names():
    assert "R" in names("Analysis in R and Python") and "R" not in names("the r key")
    assert "Go" not in names("Go to market strategy") and "Go" in names("Services written in Go and Rust")


def test_counts_and_order():
    hits = get_extractor().extract("SQL, Python, Python, python")
    assert [(h.skill, h.count) for h in hits] == [("Python", 3), ("SQL", 1)]


def test_canonicalise():
    ex = get_extractor()
    assert ex.canonicalise("ML") == "Machine Learning" and ex.canonicalise("postgres") == "PostgreSQL"
    assert ex.canonicalise("Underwater basket weaving") is None


# ------------------------------------------------------------------ rubric maths
def test_rubric_weights_sum_to_one():
    assert sum(SECTION_WEIGHTS.values()) == pytest.approx(1.0)
    for section, ws in METRIC_WEIGHTS.items():
        assert sum(ws.values()) == pytest.approx(1.0), section


def test_band_trapezoid():
    assert band(5, 10, 60, 150, 220) == 0 and band(60, 10, 60, 150, 220) == 1
    assert band(35, 10, 60, 150, 220) == pytest.approx(0.5) and band(185, 10, 60, 150, 220) == pytest.approx(0.5)
    assert band(300, 10, 60, 150, 220) == 0


def test_flesch_orders_easy_before_hard():
    easy = "The cat sat on the mat. It was a big cat."
    hard = "Notwithstanding organisational heterogeneity, multidisciplinary methodological considerations necessitate comprehensive evaluation."
    assert flesch_reading_ease(easy) > 80 > flesch_reading_ease(hard)


@pytest.mark.parametrize("text,expected", [
    ("Built a pipeline that cut costs", True), ("Led 3 engineers", True), ("I led a team", True),
    ("Successfully deployed models", True), ("Responsible for backend", False), ("Helped with reporting", False),
    ("Optimised queries", True), ("", False),
])
def test_action_verb_detection(text, expected):
    assert starts_with_action_verb(text) is expected


@pytest.mark.parametrize("text,n", [
    ("cut latency by 40%", 1), ("served $2M in revenue to 10k users", 2), ("worked 2019 - 2021", 0),
    ("no numbers here", 0), ("processing 2M documents per day", 1),
])
def test_count_quantified(text, n):
    assert count_quantified(text) == n


# ------------------------------------------------------------------ whole-profile behaviour
STRONG = Profile(
    headline="Machine Learning Engineer | Python | PyTorch | NLP",
    about=("I build production NLP systems. Built a semantic search service that served 2M queries per day and cut "
           "latency by 45%. Led a team of 4 engineers using Python, PyTorch, Docker, AWS, SQL, Kubernetes and Airflow. "
           "Reduced annotation cost by 30% with active learning.\n\nI am open to ML roles - reach out any time. "
           "I enjoy mentoring and sharing what I learn with the community through talks and writing."),
    experience=[Experience(title="ML Engineer", company="Acme", date_range="Jan 2021 - Present",
                           description="Built a ranking model that improved CTR by 12%.\nDeployed services on AWS with Docker and Kubernetes.\n"
                                       "Reduced inference cost by 40% using ONNX.")],
    skills=["Python", "PyTorch", "NLP", "Docker", "AWS", "SQL", "Kubernetes", "Airflow", "Machine Learning", "Leadership"],
)
WEAK = Profile(headline="student", about="I like computers.",
               experience=[Experience(title="Intern", description="Responsible for helping with stuff.")], skills=["stuff"])


def test_strong_profile_beats_weak_profile():
    assert analyze_profile(STRONG, use_llm=False).overall_score > analyze_profile(WEAK, use_llm=False).overall_score + 25


def test_empty_profile_scores_zero_without_crashing():
    a = analyze_profile(Profile(), use_llm=False)
    assert a.overall_score == 0 and a.grade == "E" and all(not s.present for s in a.sections)
    assert a.top_tips                      # still tells the user what to add


def test_scores_stay_in_range():
    for prof in (STRONG, WEAK, Profile()):
        a = analyze_profile(prof, use_llm=False)
        assert 0 <= a.overall_score <= 100
        assert all(0 <= m.score <= 100 for s in a.sections for m in s.metrics)


def test_missing_skills_flagged():
    prof = Profile(experience=[Experience(title="Dev", description="Built apps with React and Docker on AWS.")], skills=["React"])
    rep = build_skill_report(prof)
    assert {h.skill for h in rep.missing_from_skills_section} == {"Docker", "AWS"}
    assert "React" in rep.listed_recognised


def test_alias_in_skills_section_counts_as_listed():
    prof = Profile(experience=[Experience(description="Trained machine learning models.")], skills=["ML"])
    assert build_skill_report(prof).missing_from_skills_section == []


def test_headline_only_mention_is_not_flagged_missing():
    prof = Profile(headline="Python Developer", skills=[])
    assert build_skill_report(prof).missing_from_skills_section == []


def test_sample_pdf_pipeline_end_to_end():
    a = analyze_profile(parse_profile_pdf((SAMPLES / "sample_data_scientist.pdf").read_bytes()), use_llm=False)
    assert 50 < a.overall_score < 85
    assert {"Airflow", "PyTorch"} <= {h.skill for h in a.skills.missing_from_skills_section}


# ------------------------------------------------------------------ rewrites
def test_rule_based_rewrite_invents_no_numbers():
    prof = parse_profile_pdf((SAMPLES / "sample_data_scientist.pdf").read_bytes())
    rw = rule_based_rewrite(prof, build_skill_report(prof))
    source_numbers = set(re.findall(r"\d+", prof.full_text() + prof.raw_text))
    produced = set(re.findall(r"\d+", " ".join(rw.headline_options) + rw.about))
    years = {str(years_of_experience(prof))}
    assert produced <= source_numbers | years
    assert all(len(h) <= 220 for h in rw.headline_options)


def test_rewrite_works_on_empty_profile():
    rw = rule_based_rewrite(Profile(), build_skill_report(Profile()))
    assert rw.source == "rule-based" and rw.headline_options and rw.about


def test_years_of_experience_month_precision():
    prof = Profile(experience=[Experience(date_range="January 2020 - December 2021"), Experience(date_range="June 2021 - March 2023")])
    assert years_of_experience(prof) == 3          # Jan 2020 -> Mar 2023 = 3y 2m


# ------------------------------------------------------------------ LLM + fallback
def test_no_api_key_uses_rule_based(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    assert analyze_profile(STRONG, use_llm=True).rewrites.source == "rule-based"


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self.payload


def _gemini_payload(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}


def test_gemini_success_is_used(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["prompt"], sent["headers"] = json["contents"][0]["parts"][0]["text"], headers
        return _Resp(_gemini_payload({"headline_options": ["A", "B", "C"], "about": "Hello"}))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    rw = analyze_profile(STRONG, use_llm=True).rewrites
    assert rw.source == "gemini" and rw.headline_options == ["A", "B", "C"]
    assert sent["headers"]["x-goog-api-key"] == "fake"          # key goes in a header, never in the URL
    assert "ONLY facts" in sent["prompt"] or "ONLY" in sent["prompt"]


@pytest.mark.parametrize("failure", ["http_error", "bad_json", "network"])
def test_gemini_failures_fall_back(monkeypatch, failure):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")

    def fake_post(*a, **k):
        if failure == "network":
            raise httpx.ConnectError("offline")
        if failure == "http_error":
            return _Resp({}, status=429)
        return _Resp({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    assert analyze_profile(STRONG, use_llm=True).rewrites.source == "rule-based"


def test_prompt_never_contains_name_or_contact():
    prof = STRONG.model_copy(update={"name": "Jane Secret", "location": "Atlantis", "raw_text": "jane@secret.com"})
    prompt = llm.build_prompt(prof, build_skill_report(prof))
    assert "Jane" not in prompt and "Atlantis" not in prompt and "secret.com" not in prompt
