"""SYNTHETIC labelled evaluation set. Everything here is generated - no real person's data.

What is generated
  * 30 fictional profiles = 10 role families x 3 profiles. Two per family state their role
    explicitly ("Data Scientist ..."); one is "obfuscated": generic job titles and a headline that
    names no role, so a pure keyword matcher has less to latch onto.
  * For every profile, a 100-job CANDIDATE POOL drawn from the real job sample and labelled:
        grade 2  title matches the profile's own role family            (5 jobs)
        grade 1  title matches an ADJACENT family (e.g. DS <-> DA)      (3 jobs)
        grade 0  hard negatives: titles of OTHER tech families          (30 jobs)
        grade 0  random jobs (mostly non-tech)                          (62 jobs)

How relevance is judged: a rule (regex on the job TITLE per role family). This is a proxy for human
judgement, NOT ground truth - see the limitations section in docs/RESULTS.md.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import pandas as pd

from ..parser.models import Education, Experience, Profile

N_G2, N_G1, N_HARD, N_RANDOM = 5, 3, 30, 62      # pool composition (sums to 100)
SEED = 2024

DOMAINS = ["fraud detection", "customer churn", "demand forecasting", "search ranking", "ad targeting",
           "patient triage", "supply chain planning", "loan underwriting", "content moderation", "inventory management"]
METRICS = ["accuracy", "retention", "latency", "conversion", "operating cost", "delivery time", "uptime"]
VOLUMES = ["500K", "2M", "10M", "1.5M", "80K", "3M"]
COMPANIES = ["Nimbus Labs", "Orbit Systems", "Brightpath", "Kestrel Tech", "Lumen Works", "Vertex Group", "Quanta Soft", "Blue Harbor"]
GENERIC_TITLES = ["Associate", "Consultant", "Member of Technical Staff", "Specialist"]
SCHOOLS = ["Pune Institute of Technology", "Delhi Technical University", "Nirma University", "Anna University", "BITS Pilani"]
SOFT = ["Communication", "Teamwork", "Problem Solving", "Leadership"]


@dataclass(frozen=True)
class Family:
    key: str
    roles: list[str]
    title_regex: str
    adjacent: list[str]
    skills: list[str]
    obfuscated_headlines: list[str]
    bullets: list[str]


FAMILIES: dict[str, Family] = {f.key: f for f in [
    Family("data_scientist", ["Data Scientist", "Machine Learning Engineer", "NLP Engineer"],
           r"data scien|machine learning|\bml\b|\bai\b|\bnlp\b|deep learning|applied scien|research scien",
           ["data_analyst", "data_engineer"],
           ["Python", "Machine Learning", "Deep Learning", "scikit-learn", "PyTorch", "TensorFlow", "Pandas", "NumPy", "SQL",
            "Natural Language Processing", "Statistics", "Feature Engineering", "Model Deployment", "Hugging Face", "XGBoost", "A/B Testing"],
           ["Turning raw data into decisions", "Teaching machines to find signal in messy data", "Predictive models that ship"],
           ["Built {s1} models for {domain} that improved {metric} by {n}%.", "Trained and evaluated {s2} pipelines on {vol} records using {s3}.",
            "Deployed a {s1} service to production serving {vol} requests per day.", "Ran {s2} experiments with product teams to lift {metric} by {n}%."]),
    Family("data_analyst", ["Data Analyst", "Business Intelligence Analyst", "Analytics Associate"],
           r"data analy|business intelligence|\bbi\b|analytics|reporting analy|insights",
           ["data_scientist", "data_engineer"],
           ["SQL", "Excel", "Tableau", "Power BI", "Python", "Data Visualization", "Data Analysis", "Statistics", "Dashboarding",
            "Google Analytics", "A/B Testing", "Business Intelligence", "KPI Reporting", "Pandas", "Looker"],
           ["Making numbers tell the story", "Dashboards people actually open", "From spreadsheets to insight"],
           ["Built {s1} dashboards that cut weekly reporting time by {n}%.", "Wrote {s2} queries over {vol} rows to track {metric} for {domain}.",
            "Automated KPI reports in {s3}, saving {n} analyst hours a month.", "Analysed {domain} data with {s1} and presented findings to leadership."]),
    Family("data_engineer", ["Data Engineer", "Big Data Engineer", "Analytics Engineer"],
           r"data engineer|\betl\b|big data|data architect|analytics engineer|data platform",
           ["data_scientist", "data_analyst", "devops"],
           ["Python", "SQL", "Apache Spark", "Kafka", "Airflow", "ETL", "Data Pipelines", "Data Warehousing", "Snowflake", "AWS",
            "PySpark", "dbt", "Hadoop", "BigQuery", "Data Modeling", "Docker"],
           ["Plumbing for data teams", "Moving data reliably at scale", "Pipelines that don't wake you at 3am"],
           ["Designed {s1} pipelines processing {vol} events per day with {s2}.", "Migrated {domain} workloads to {s3}, reducing {metric} by {n}%.",
            "Built {s1} jobs that cut batch runtime by {n}%.", "Modelled a {s2} warehouse used by {n} analysts."]),
    Family("backend", ["Backend Engineer", "Software Engineer", "Python Developer"],
           r"back.?end|software engineer|software developer|java developer|python developer|\.net|api developer",
           ["fullstack", "devops"],
           ["Python", "Java", "REST APIs", "PostgreSQL", "Docker", "Microservices", "Django", "FastAPI", "Spring Boot", "Redis", "SQL",
            "Unit Testing", "System Design", "Git", "Kafka", "AWS"],
           ["Building the services behind the screens", "APIs that scale and stay up", "Clean code, reliable systems"],
           ["Built {s1} services handling {vol} requests per day using {s2}.", "Reduced API latency by {n}% by optimising {s3} queries.",
            "Designed {s1} microservices for {domain} with {s2}.", "Wrote unit tests covering {n}% of {s3} code."]),
    Family("frontend", ["Frontend Developer", "React Developer", "UI Engineer"],
           r"front.?end|ui developer|\breact\b|ui engineer|angular|javascript developer",
           ["fullstack"],
           ["JavaScript", "TypeScript", "React", "HTML", "CSS", "Redux", "Next.js", "Tailwind CSS", "Responsive Design", "Jest",
            "Web Accessibility", "Webpack", "Figma", "GraphQL", "Git", "Vue.js"],
           ["Crafting interfaces people enjoy", "Pixels, performance and accessibility", "Where design meets code"],
           ["Built {s1} components used across {n} product pages.", "Improved page load time by {n}% with {s2} code splitting.",
            "Migrated a legacy codebase to {s1} and {s2} for {domain}.", "Raised accessibility scores to {n}+ using {s3}."]),
    Family("fullstack", ["Full Stack Developer", "Full Stack Engineer", "Web Application Developer"],
           r"full.?stack",
           ["frontend", "backend"],
           ["JavaScript", "React", "Node.js", "Express.js", "MongoDB", "PostgreSQL", "REST APIs", "TypeScript", "Docker", "HTML", "CSS",
            "Git", "AWS", "Next.js", "GraphQL", "CI/CD"],
           ["End to end: from database to button", "Shipping whole features solo", "One developer, the whole stack"],
           ["Delivered {s1} and {s2} features for {domain} used by {vol} users.", "Built a {s3} API and a {s1} frontend in {n} weeks.",
            "Set up CI/CD with {s2}, cutting release time by {n}%.", "Designed the {s3} schema supporting {vol} records."]),
    Family("devops", ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"],
           r"devops|site reliability|\bsre\b|cloud|platform engineer|infrastructure|systems engineer",
           ["backend", "security", "data_engineer"],
           ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins", "Linux", "Ansible", "Prometheus", "Grafana", "Python",
            "Bash", "GitHub Actions", "Infrastructure as Code", "Azure", "Monitoring"],
           ["Keeping systems fast, safe and boring", "Automating everything that can be automated", "Infrastructure as code evangelist"],
           ["Automated {s1} deployments with {s2}, cutting release time by {n}%.", "Managed {s3} clusters serving {vol} requests per day.",
            "Reduced cloud spend by {n}% by right-sizing {s1} workloads.", "Built {s2} dashboards improving {metric} visibility."]),
    Family("security", ["Security Analyst", "Cybersecurity Engineer", "SOC Analyst"],
           r"security|cyber|soc analyst|penetration|infosec|information assurance",
           ["devops"],
           ["Cybersecurity", "Network Security", "SIEM", "Incident Response", "Penetration Testing", "Vulnerability Assessment", "Splunk",
            "Wireshark", "Linux", "Python", "OWASP", "Threat Modeling", "Firewall", "ISO 27001", "Risk Assessment", "Identity and Access Management"],
           ["Defending what others build", "Finding the weak spots before attackers do", "Security is a team sport"],
           ["Investigated {n}+ incidents using {s1} and {s2}, cutting response time by {n}%.", "Ran {s1} assessments across {n} applications.",
            "Tuned {s2} alerts, reducing false positives by {n}%.", "Wrote {s3} runbooks adopted by the on-call team."]),
    Family("product_manager", ["Product Manager", "Associate Product Manager", "Technical Program Manager"],
           r"product manager|product owner|program manager|project manager|technical program",
           ["marketing"],
           ["Product Management", "Product Roadmap", "User Stories", "Agile", "Scrum", "Stakeholder Management", "Jira", "Market Research",
            "Product Analytics", "A/B Testing", "Go-to-Market Strategy", "Prioritization", "SQL", "Roadmapping", "OKRs", "Customer Success"],
           ["Bridging users, design and engineering", "Shipping products people ask for", "Outcome over output"],
           ["Owned the {s1} for {domain}, lifting {metric} by {n}%.", "Ran {s2} with {n} stakeholders to prioritise a {s3} backlog.",
            "Launched {n} features using {s1}, reaching {vol} users.", "Defined {s2} KPIs and tracked them in {s3}."]),
    Family("marketing", ["Digital Marketing Specialist", "Content Marketing Manager", "Growth Marketer"],
           r"marketing|\bseo\b|content|social media|brand|growth|communications",
           ["product_manager"],
           ["Digital Marketing", "SEO", "Content Marketing", "Social Media Marketing", "Google Analytics", "Google Ads", "Email Marketing",
            "Copywriting", "Marketing Automation", "HubSpot", "Brand Management", "Campaign Management", "A/B Testing", "Canva", "Content Strategy", "Mailchimp"],
           ["Telling stories that convert", "Growth through great content", "Brands people remember"],
           ["Grew organic traffic by {n}% through {s1} and {s2}.", "Managed {s3} campaigns generating {vol} impressions.",
            "Launched an {s1} programme that raised {metric} by {n}%.", "Ran {s2} tests lifting conversion by {n}%."]),
]}


# ---------------------------------------------------------------- relevance rules
def matches_family(title: str, key: str) -> bool:
    return re.search(FAMILIES[key].title_regex, str(title), re.I) is not None


def grade_title(title: str, family_key: str) -> int:
    """2 = own family, 1 = adjacent family, 0 = otherwise."""
    if matches_family(title, family_key):
        return 2
    if any(matches_family(title, adj) for adj in FAMILIES[family_key].adjacent):
        return 1
    return 0


def is_tech_other(title: str, family_key: str) -> bool:
    """A job from a DIFFERENT family that is not adjacent (a hard negative)."""
    fam = FAMILIES[family_key]
    others = [k for k in FAMILIES if k != family_key and k not in fam.adjacent]
    return grade_title(title, family_key) == 0 and any(matches_family(title, k) for k in others)


# ---------------------------------------------------------------- profile generation
def make_profile(fam: Family, rng: random.Random, obfuscated: bool) -> Profile:
    skills = rng.sample(fam.skills, k=rng.randint(8, 11))
    foreign = rng.choice([f for f in FAMILIES.values() if f.key != fam.key])
    listed = skills + rng.sample(foreign.skills, 1) + rng.sample(SOFT, 1)      # 1 distractor + 1 soft skill
    rng.shuffle(listed)
    s1, s2, s3 = skills[:3]
    yrs = rng.randint(2, 7)
    fill = lambda t: t.format(s1=rng.choice(skills), s2=rng.choice(skills), s3=rng.choice(skills), n=rng.randint(8, 45),  # noqa: E731
                              domain=rng.choice(DOMAINS), metric=rng.choice(METRICS), vol=rng.choice(VOLUMES))
    role = rng.choice(fam.roles)
    title_now = rng.choice(GENERIC_TITLES) if obfuscated else role
    title_before = rng.choice(GENERIC_TITLES) if obfuscated else role.replace("Senior ", "")
    headline = f"{rng.choice(fam.obfuscated_headlines)} | {s1} | {s2}" if obfuscated else f"{role} | {s1} | {s2} | {s3}"
    lead = "Professional" if obfuscated else role
    about = (f"{lead} with {yrs} years of experience in {s1}, {s2} and {s3}. {fill(rng.choice(fam.bullets))} "
             f"{fill(rng.choice(fam.bullets))} I'm open to new opportunities - feel free to reach out.")
    bullets = lambda: "\n".join(fill(b) for b in rng.sample(fam.bullets, 3))  # noqa: E731
    company1, company2 = rng.sample(COMPANIES, 2)
    experience = [
        Experience(title=title_now, company=company1, date_range=f"{2026 - yrs + 2} - Present", description=bullets()),
        Experience(title=title_before, company=company2, date_range=f"{2026 - yrs} - {2026 - yrs + 2}", description=bullets()),
    ]
    return Profile(
        name="[SYNTHETIC]", headline=headline, about=about, experience=experience,
        education=[Education(school=rng.choice(SCHOOLS), degree="Bachelor of Technology - BTech, Computer Science", dates="2018 - 2022")],
        skills=listed, certifications=[])


def build_candidate_pool(jobs: pd.DataFrame, family_key: str, rng: random.Random) -> dict[str, int]:
    """Sample a labelled candidate pool {job_id: grade}. Raises if the dataset has too few jobs."""
    titles = jobs["title"].tolist()
    ids = jobs["job_id"].astype(str).tolist()
    g = [grade_title(t, family_key) for t in titles]
    own = [i for i, x in enumerate(g) if x == 2]
    adj = [i for i, x in enumerate(g) if x == 1]
    hard = [i for i, t in enumerate(titles) if is_tech_other(t, family_key)]
    rest = [i for i, x in enumerate(g) if x == 0 and i not in set(hard)]
    if len(own) < N_G2 or len(adj) < N_G1 or len(hard) < N_HARD or len(rest) < N_RANDOM:
        raise ValueError(f"job sample too small for family {family_key!r}: own={len(own)} adj={len(adj)} hard={len(hard)}")
    picks = ([(i, 2) for i in rng.sample(own, N_G2)] + [(i, 1) for i in rng.sample(adj, N_G1)]
             + [(i, 0) for i in rng.sample(hard, N_HARD)] + [(i, 0) for i in rng.sample(rest, N_RANDOM)])
    return {ids[i]: grade for i, grade in picks}


def build_labelled_set(jobs: pd.DataFrame, seed: int = SEED) -> list[dict]:
    """30 synthetic profiles (3 per family; the 3rd is obfuscated) + their labelled pools."""
    rng = random.Random(seed)
    out = []
    for fam in FAMILIES.values():
        for n in range(3):
            obfuscated = n == 2
            prof = make_profile(fam, rng, obfuscated)
            out.append({"id": f"{fam.key}-{n + 1}", "family": fam.key, "obfuscated": obfuscated, "synthetic": True,
                        "profile": prof.model_dump(exclude={"raw_text"}), "judgments": build_candidate_pool(jobs, fam.key, rng)})
    return out
