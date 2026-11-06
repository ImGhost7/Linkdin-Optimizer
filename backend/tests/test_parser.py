"""Parser tests: LinkedIn-style PDF, pasted text in several formats, and degenerate input."""
from pathlib import Path

import pytest

from backend.parser.section_parser import parse_profile_pdf, parse_profile_text

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "sample_profiles"


@pytest.fixture(scope="module")
def ds_profile():
    return parse_profile_pdf((SAMPLES / "sample_data_scientist.pdf").read_bytes())


def test_pdf_header_fields(ds_profile):
    assert ds_profile.name == "Aarav Mehta"
    assert ds_profile.headline == "Data Scientist | NLP | Machine Learning"
    assert ds_profile.location.startswith("Bengaluru")


def test_pdf_columns_not_interleaved(ds_profile):
    # sidebar content must not leak into the main column and vice versa
    assert ds_profile.skills == ["Python", "Machine Learning", "SQL"]
    assert "Top Skills" not in ds_profile.about and "aarav.mehta@example.com" not in ds_profile.raw_text.split("Summary")[1].split("Contact")[0]
    assert len(ds_profile.certifications) == 2
    assert ds_profile.certifications[0] == "Google Data Analytics Professional Certificate"


def test_pdf_experience_entries(ds_profile):
    assert [e.title for e in ds_profile.experience] == ["Data Scientist", "Data Analyst"]
    first = ds_profile.experience[0]
    assert first.company == "Fintrix Analytics"
    assert first.date_range == "March 2022 - Present"
    assert first.duration == "2 years 8 months"
    assert first.description.count("\n") == 2          # three separate statements
    assert first.description.splitlines()[0].startswith("Built a transformer")


def test_pdf_education(ds_profile):
    (edu,) = ds_profile.education
    assert edu.school == "Vellore Institute of Technology"
    assert "Computer Science" in edu.degree and edu.dates == "2016 - 2020"


def test_second_sample_has_no_certifications():
    p = parse_profile_pdf((SAMPLES / "sample_frontend_developer.pdf").read_bytes())
    assert p.certifications == [] and p.headline == "Frontend Developer"


PASTED = """Name: Rohan Gupta
Headline: Backend Engineer | Python | AWS
About
I build APIs.

Second paragraph here.

Experience
Software Engineer at Acme Corp (Jan 2021 - Present)
- Built REST APIs serving 1M requests/day
- Reduced latency by 30%
Intern, Beta Labs  Jun 2020 - Dec 2020
Wrote unit tests.
Education
B.Tech in Computer Science, NIT Trichy, 2017-2021
Skills: Python, Django; AWS | Docker
Certifications
AWS Certified Developer
"""


def test_pasted_labels_and_inline_experience():
    p = parse_profile_text(PASTED)
    assert p.name == "Rohan Gupta" and p.headline.startswith("Backend Engineer")
    assert p.about.count("\n\n") == 1                  # paragraph break preserved
    assert [(e.title, e.company) for e in p.experience] == [("Software Engineer", "Acme Corp"), ("Intern", "Beta Labs")]
    assert p.experience[0].description.splitlines() == ["Built REST APIs serving 1M requests/day", "Reduced latency by 30%"]
    assert p.skills == ["Python", "Django", "AWS", "Docker"]
    assert p.education[0].school == "NIT Trichy" and p.education[0].dates == "2017 - 2021"
    assert p.certifications == ["AWS Certified Developer"]


def test_no_headings_becomes_about():
    p = parse_profile_text("Just a paragraph about my work as an analyst with SQL.")
    assert p.about.startswith("Just a paragraph") and p.experience == [] and p.headline == ""


def test_empty_input_does_not_crash():
    p = parse_profile_text("")
    assert p.experience == [] and p.skills == [] and p.about == ""


def test_experience_without_dates_is_kept_as_one_entry():
    p = parse_profile_text("Experience\nI built things.\nI led teams.")
    assert len(p.experience) == 1 and "built things" in p.experience[0].description
