"""Matcher tests. A deterministic fake encoder (hashed bag-of-words) keeps them fast and offline;
the real MiniLM model is exercised separately in test_embeddings_integration."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from backend.matcher.gap import make_tips
from backend.matcher.hybrid import HybridMatcher, Query, _minmax
from backend.matcher.models import Weights
from backend.optimizer.skills import get_extractor
from backend.parser.models import Experience, Profile


def fake_encode(texts):
    """Hash each word into one of 128 buckets, L2-normalise: cosine ~ word overlap."""
    out = np.zeros((len(texts), 128), dtype="float32")
    for r, t in enumerate(texts):
        for w in t.lower().split():
            out[r, int(hashlib.md5(w.strip(".,;:()").encode()).hexdigest(), 16) % 128] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


JOBS = [
    ("1", "Machine Learning Engineer", "Build machine learning models in Python with PyTorch and TensorFlow. Deploy to AWS with Docker. SQL required."),
    ("2", "Senior Machine Learning Engineer", "Machine learning, deep learning, Python, PyTorch, MLOps, Kubernetes and Docker experience required."),
    ("3", "Frontend Developer", "Build web apps with React, TypeScript, HTML and CSS. Jest testing, Redux, responsive design."),
    ("4", "React Developer", "React, JavaScript, CSS, HTML, Redux. Work with designers on web interfaces and REST APIs."),
    ("5", "Staff Accountant", "Prepare financial statements, reconciliations, month-end close, payroll and Excel reporting."),
    ("6", "Bookkeeper", "Accounts payable, accounts receivable, payroll, Excel, financial statements and reconciliations."),
    ("7", "Data Analyst", "Analyse data with SQL, Python and Tableau. Build dashboards and reports for stakeholders."),
]


@pytest.fixture(scope="module")
def matcher():
    ex = get_extractor()
    df = pd.DataFrame([{"job_id": i, "title": t, "company_name": "Acme", "location": "Remote", "description": d,
                        "skills_desc": "", "job_posting_url": f"http://x/{i}",
                        "text": f"{t}. {t}. {d}", "skills": [h.skill for h in ex.extract(f"{t}. {d}")]} for i, t, d in JOBS])
    return HybridMatcher(df, job_embeddings=fake_encode(list(df["text"])), encode_fn=fake_encode)


ML = Profile(headline="Machine Learning Engineer", skills=["Python", "PyTorch", "Docker"],
             experience=[Experience(title="ML Engineer", description="Trained deep learning models with PyTorch and deployed with Docker on AWS.")])
FE = Profile(headline="Frontend Developer", skills=["React", "CSS", "HTML"],
             experience=[Experience(title="Developer", description="Built React interfaces with Redux and Jest tests.")])


def top_titles(matcher, profile, **kw):
    return [m.title for m in matcher.rank(Query.from_profile(profile), **kw)]


def test_profiles_rank_their_own_domain_first(matcher):
    assert "Machine Learning" in top_titles(matcher, ML, top_k=2)[0]
    assert top_titles(matcher, FE, top_k=2)[0] in {"Frontend Developer", "React Developer"}
    assert not any("Accountant" in t or "Bookkeeper" in t for t in top_titles(matcher, ML, top_k=3))


def test_each_match_has_three_tips_and_consistent_skills(matcher):
    for m in matcher.rank(Query.from_profile(ML), top_k=7):
        assert len(m.tips) == 3 and all(t.strip() for t in m.tips)
        assert set(m.matched_skills).isdisjoint(m.missing_skills)
        assert 0 <= m.score <= 100 and m.rank >= 1


def test_matched_and_missing_partition_job_skills(matcher):
    q = Query.from_profile(ML)
    m = matcher.rank(q, top_k=1)[0]
    job = matcher.jobs[matcher.jobs["job_id"] == m.job_id].iloc[0]
    assert set(m.matched_skills) | set(m.missing_skills) == set(job["skills"])
    assert "Python" in m.matched_skills and "Kubernetes" in m.missing_skills or "TensorFlow" in m.missing_skills


def test_weights_change_the_ranking_signal(matcher):
    q = Query.from_profile(ML)
    skills_only = matcher.rank(q, top_k=7, weights=Weights(embedding=0, tfidf=0, skills=1))
    assert [m.components.skills for m in skills_only] == sorted([m.components.skills for m in skills_only], reverse=True)
    emb_only = matcher.rank(q, top_k=7, weights=Weights(embedding=1, tfidf=0, skills=0))
    assert [m.components.embedding for m in emb_only] == sorted([m.components.embedding for m in emb_only], reverse=True)


def test_weights_normalise_and_validate():
    w = Weights(embedding=2, tfidf=1, skills=1).normalised()
    assert (w.embedding, w.tfidf, w.skills) == (0.5, 0.25, 0.25)
    with pytest.raises(ValueError):
        Weights(embedding=0, tfidf=0, skills=0).normalised()
    with pytest.raises(Exception):
        Weights(embedding=-1)


def test_minmax_constant_vector_is_zero_not_nan():
    assert list(_minmax(np.array([0.3, 0.3, 0.3]))) == [0, 0, 0]


def test_pasted_jd_scores_relevant_above_irrelevant(matcher):
    q = Query.from_profile(ML)
    good = matcher.match_job_description(q, "Machine learning engineer: Python, PyTorch, Docker, AWS, MLOps.")
    bad = matcher.match_job_description(q, "Accountant: payroll, reconciliations, financial statements, Excel.")
    assert good.score > bad.score + 30 and 0 <= bad.score <= 100 and len(good.tips) == 3
    assert "Python" in good.matched_skills and "Excel" in bad.missing_skills


def test_empty_profile_does_not_crash(matcher):
    matches = matcher.rank(Query.from_profile(Profile()), top_k=3)
    assert len(matches) == 3 and all(0 <= m.score <= 100 for m in matches)


# ------------------------------------------------------------------ tips
def test_tips_name_the_missing_skills():
    t = make_tips("Data Engineer", ["Python"], ["Kafka", "Airflow", "Spark", "dbt"], "Student", ["pipelines"])
    assert "Kafka, Airflow and Spark" in t[0] and "dbt" not in t[0]
    assert "Data Engineer" in t[1]                       # headline lacks the title -> suggests aligning
    assert "pipelines" in t[2]


def test_tips_when_nothing_is_missing():
    t = make_tips("Data Analyst", ["SQL", "Tableau"], [], "Data Analyst | SQL", [])
    assert len(t) == 3 and "cover every skill" in t[0] and "SQL and Tableau" in t[1] and "numbers" in t[2]


# ------------------------------------------------------------------ real model (downloads ~90 MB on first run)
@pytest.mark.slow
def test_embeddings_integration():
    from backend.matcher import embeddings
    v = embeddings.encode(["machine learning engineer", "ML engineer building neural networks", "pastry chef baking bread"])
    assert v.shape == (3, 384)
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2             # semantic similarity beats unrelated text
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)


def test_title_words_are_not_suggested_as_missing_keywords(matcher):
    from backend.matcher.gap import distinctive_absent_terms
    row = matcher.tfidf.transform(["payroll reconciliations"])
    assert {"payroll", "reconciliations"} <= set(distinctive_absent_terms(row, matcher.vocab, ""))
    got = distinctive_absent_terms(row, matcher.vocab, "", exclude="Payroll Clerk")
    assert "payroll" not in got and "reconciliations" in got
