"""API tests with FastAPI's TestClient. The matcher is replaced by a small fake-encoder one (see test_matcher)."""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.app import app, matcher_dep
from backend.matcher.hybrid import HybridMatcher
from backend.optimizer.skills import get_extractor
from backend.tests.test_matcher import JOBS, fake_encode

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "sample_profiles"
PDF = (SAMPLES / "sample_data_scientist.pdf").read_bytes()


@pytest.fixture(scope="module")
def client():
    ex = get_extractor()
    df = pd.DataFrame([{"job_id": i, "title": t, "company_name": "Acme", "location": "Remote", "description": d,
                        "skills_desc": "", "job_posting_url": f"http://x/{i}", "text": f"{t}. {t}. {d}",
                        "skills": [h.skill for h in ex.extract(f"{t}. {d}")]} for i, t, d in JOBS])
    fake = HybridMatcher(df, job_embeddings=fake_encode(list(df["text"])), encode_fn=fake_encode)
    app.dependency_overrides[matcher_dep] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "GEMINI" not in r.text.upper().replace("GEMINI_CONFIGURED", "")


def test_analyze_pdf(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    r = client.post("/api/analyze", files={"file": ("p.pdf", PDF, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["name"] == "Aarav Mehta" and len(body["profile"]["experience"]) == 2
    assert 0 < body["analysis"]["overall_score"] <= 100
    assert [s["section"] for s in body["analysis"]["sections"]] == ["headline", "about", "experience", "education", "skills", "certifications"]
    assert body["analysis"]["rewrites"]["source"] == "rule-based"


def test_analyze_text_with_extra_skills(client):
    r = client.post("/api/analyze", data={"text": "Headline: Backend Dev\nSkills\nPython", "extra_skills": "Docker, python, AWS", "use_llm": "false"})
    assert r.status_code == 200
    assert r.json()["profile"]["skills"] == ["Python", "Docker", "AWS"]      # merged, case-insensitive de-dup


@pytest.mark.parametrize("kwargs,status", [
    (dict(data={}), 400),
    (dict(files={"file": ("x.pdf", b"not a pdf", "application/pdf")}), 415),
    (dict(files={"file": ("x.pdf", b"%PDF-1.4 garbage", "application/pdf")}), 422),
    (dict(files={"file": ("big.pdf", b"%PDF" + b"0" * (5 * 1024 * 1024 + 1), "application/pdf")}), 413),
    (dict(data={"text": "x" * 60_001}), 413),
])
def test_analyze_rejects_bad_input(client, kwargs, status):
    assert client.post("/api/analyze", **kwargs).status_code == status


def _profile(client):
    return client.post("/api/analyze", files={"file": ("p.pdf", PDF, "application/pdf")}).json()["profile"]


def test_match_dataset(client):
    r = client.post("/api/match", json={"profile": _profile(client), "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "dataset" and len(body["matches"]) == 3 and body["pool_size"] == len(JOBS)
    assert abs(sum(body["weights"].values()) - 1) < 1e-6
    scores = [m["score"] for m in body["matches"]]
    assert scores == sorted(scores, reverse=True) and all(len(m["tips"]) == 3 for m in body["matches"])


def test_match_custom_weights_are_normalised(client):
    r = client.post("/api/match", json={"profile": _profile(client), "top_k": 2, "weights": {"embedding": 2, "tfidf": 0, "skills": 2}})
    assert r.json()["weights"] == {"embedding": 0.5, "tfidf": 0.0, "skills": 0.5}


def test_match_pasted_jd(client):
    r = client.post("/api/match", json={"profile": _profile(client), "job_title": "ML Eng", "job_description": "Python, PyTorch, machine learning, SQL, Airflow"})
    body = r.json()
    assert body["mode"] == "job_description" and len(body["matches"]) == 1 and body["matches"][0]["title"] == "ML Eng"
    assert "Python" in body["matches"][0]["matched_skills"]


@pytest.mark.parametrize("payload,status", [
    ({"profile": {}}, 422),                                                    # empty profile
    ({"profile": {"headline": "x"}, "weights": {"embedding": 0, "tfidf": 0, "skills": 0}}, 422),
    ({"profile": {"headline": "x"}, "top_k": 0}, 422),
    ({"profile": {"headline": "x"}, "weights": {"embedding": -1}}, 422),
])
def test_match_validation(client, payload, status):
    assert client.post("/api/match", json=payload).status_code == status


def test_missing_job_dataset_gives_503(monkeypatch):
    """If the jobs CSV is absent, /api/match must answer 503 with a helpful message (not a 500 stack trace)."""
    from backend.matcher import service

    def no_csv():
        raise FileNotFoundError("data/jobs_sample.csv not found. Build it with: python data/build_jobs_sample.py")

    app.dependency_overrides.clear()
    service.get_matcher.cache_clear()
    monkeypatch.setattr(service, "load_jobs", no_csv)          # patch where it is USED (service imports it by name)
    r = TestClient(app).post("/api/match", json={"profile": {"headline": "x"}})
    service.get_matcher.cache_clear()
    assert r.status_code == 503 and "build_jobs_sample" in r.json()["detail"]
