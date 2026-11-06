"""Evaluation tests: metric formulas (hand-computed), labelling rules and labelled-set construction."""
import math
import random

import pandas as pd
import pytest

from backend.eval import synthetic as syn
from backend.eval.metrics import all_metrics, dcg_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from backend.optimizer.skills import get_extractor


def test_precision_and_recall():
    g = [2, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert precision_at_k(g, 5) == pytest.approx(2 / 5)
    assert recall_at_k(g, 10, total_relevant=4) == pytest.approx(3 / 4)
    assert recall_at_k(g, 10, total_relevant=0) == 0


def test_reciprocal_rank():
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([2, 0]) == 1
    assert reciprocal_rank([0, 0, 0]) == 0


def test_ndcg_hand_computed():
    # gains: grade 2 -> 3, grade 1 -> 1.   ranking [1, 2]: DCG = 1/log2(2) + 3/log2(3)
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    assert dcg_at_k([1, 2], 10) == pytest.approx(dcg)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)           # best order is [2, 1]
    assert ndcg_at_k([1, 2], 10, all_grades=[2, 1, 0, 0]) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([2, 1], 10, all_grades=[2, 1, 0, 0]) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0], 10, all_grades=[0, 0]) == 0


def test_metrics_perfect_and_worst_ranking():
    pool = [2] * 5 + [1] * 3 + [0] * 92
    perfect = all_metrics(sorted(pool, reverse=True), pool)
    assert perfect == {"P@5": 1.0, "R@10": 1.0, "MRR": 1.0, "NDCG@10": pytest.approx(1.0)}
    worst = all_metrics(sorted(pool), pool)
    assert worst["P@5"] == 0 and worst["R@10"] == 0 and worst["NDCG@10"] == 0


def test_family_skills_exist_in_taxonomy():
    ex = get_extractor()
    for fam in syn.FAMILIES.values():
        unknown = [s for s in fam.skills if s not in ex.category]
        assert not unknown, f"{fam.key}: not in taxonomy {unknown}"
    assert len(syn.FAMILIES) == 10


@pytest.mark.parametrize("title,family,grade", [
    ("Senior Data Scientist", "data_scientist", 2), ("Data Analyst II", "data_scientist", 1),
    ("Registered Nurse", "data_scientist", 0), ("Full Stack Developer", "frontend", 1),
    ("Full Stack Developer", "fullstack", 2), ("Cloud Security Engineer", "devops", 2),
])
def test_grade_title(title, family, grade):
    assert syn.grade_title(title, family) == grade


def test_hard_negatives_are_tech_but_not_relevant():
    assert syn.is_tech_other("Marketing Manager", "backend")
    assert not syn.is_tech_other("Data Analyst", "data_scientist")     # adjacent, so not a negative
    assert not syn.is_tech_other("Registered Nurse", "backend")        # not tech


def test_profiles_are_synthetic_and_obfuscation_hides_role():
    rng = random.Random(0)
    fam = syn.FAMILIES["data_scientist"]
    explicit, obf = syn.make_profile(fam, rng, False), syn.make_profile(fam, rng, True)
    assert explicit.name == obf.name == "[SYNTHETIC]"
    assert any(r in explicit.headline for r in fam.roles)
    assert not any(r in obf.headline for r in fam.roles) and not any(e.title in fam.roles for e in obf.experience)


def _fake_jobs():
    rows = []
    for fam in syn.FAMILIES.values():
        rows += [f"{fam.roles[0]} {n}" for n in range(12)]
    rows += [f"Registered Nurse {n}" for n in range(80)] + [f"Barista {n}" for n in range(40)]
    return pd.DataFrame({"job_id": [str(i) for i in range(len(rows))], "title": rows})


def test_candidate_pool_composition_and_determinism():
    jobs = _fake_jobs()
    pool = syn.build_candidate_pool(jobs, "backend", random.Random(1))
    grades = list(pool.values())
    assert len(pool) == 100 and grades.count(2) == syn.N_G2 and grades.count(1) == syn.N_G1 and grades.count(0) == 92
    titles = dict(zip(jobs["job_id"], jobs["title"]))
    assert all(syn.grade_title(titles[j], "backend") == g for j, g in pool.items())     # labels agree with the rule
    assert pool == syn.build_candidate_pool(jobs, "backend", random.Random(1))


def test_pool_builder_complains_when_dataset_too_small():
    with pytest.raises(ValueError, match="too small"):
        syn.build_candidate_pool(_fake_jobs().head(30), "backend", random.Random(1))


def test_labelled_set_shape():
    data = syn.build_labelled_set(_fake_jobs())
    assert len(data) == 30 and sum(x["obfuscated"] for x in data) == 10 and all(x["synthetic"] for x in data)
    assert len({x["id"] for x in data}) == 30
