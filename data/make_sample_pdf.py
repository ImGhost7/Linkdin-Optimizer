"""Generate SYNTHETIC LinkedIn-style profile PDFs (two-column layout like LinkedIn's "Save to PDF").

These are fictional people, created only so the app, tests and README demo have an input file.
    python data/make_sample_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / "sample_profiles"

SAMPLES = {
    "sample_data_scientist.pdf": {
        "name": "Aarav Mehta",
        "headline": "Data Scientist | NLP | Machine Learning",
        "location": "Bengaluru, Karnataka, India",
        "contact": ["aarav.mehta@example.com", "www.linkedin.com/in/aarav-mehta-example (LinkedIn)"],
        "top_skills": ["Python", "Machine Learning", "SQL"],
        "certs": ["Google Data Analytics Professional", "Certificate",
                  "AWS Certified Cloud Practitioner"],
        "summary": [
            "Data scientist with 3 years of experience building NLP and recommendation systems.",
            "I enjoy turning messy text data into products. Worked with PyTorch, scikit-learn,",
            "Hugging Face and Airflow. Looking for ML roles.",
        ],
        "experience": [
            ("Fintrix Analytics", "Data Scientist", "March 2022 - Present (2 years 8 months)", "Bengaluru, Karnataka, India", [
                "Built a transformer-based complaint classifier that cut manual triage time by 40%.",
                "Developed an Airflow pipeline processing 2M documents per day using Python and Spark.",
                "Led a team of 3 analysts to deliver a churn prediction model, improving retention by 12%.",
            ]),
            ("Fintrix Analytics", "Data Analyst", "June 2020 - February 2022 (1 year 9 months)", "Pune, Maharashtra, India", [
                "Worked on dashboards in Tableau and SQL queries for the finance team.",
                "Helped with reporting.",
            ]),
        ],
        "education": [("Vellore Institute of Technology", "Bachelor of Technology - BTech, Computer Science", "2016 - 2020")],
    },
    "sample_frontend_developer.pdf": {
        "name": "Diya Kapoor",
        "headline": "Frontend Developer",
        "location": "Pune, Maharashtra, India",
        "contact": ["diya.kapoor@example.com"],
        "top_skills": ["JavaScript", "React", "CSS"],
        "certs": [],
        "summary": ["Frontend developer who builds responsive web apps with React and TypeScript."],
        "experience": [
            ("Pixel Forge", "Frontend Developer", "January 2021 - Present (3 years 10 months)", "Pune, Maharashtra, India", [
                "Responsible for building React components and fixing bugs.",
                "Migrated the legacy jQuery codebase to React and Redux, reducing page load time by 35%.",
                "Wrote Jest unit tests and set up GitHub Actions CI/CD.",
            ]),
        ],
        "education": [("Savitribai Phule Pune University", "Bachelor of Engineering - BE, Information Technology", "2017 - 2021")],
    },
}


def draw_profile(path: Path, p: dict) -> None:
    w, h = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    side_x, main_x, main_w = 30, 215, w - 215 - 30

    def wrap(text, font, size, width):
        words, line, lines = text.split(), "", []
        for word in words:
            trial = f"{line} {word}".strip()
            if c.stringWidth(trial, font, size) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        return lines + [line] if line else lines

    # ---- sidebar
    y = h - 60
    for title, items in (("Contact", p["contact"]), ("Top Skills", p["top_skills"]), ("Certifications", p["certs"])):
        if not items:
            continue
        c.setFont("Helvetica-Bold", 11); c.drawString(side_x, y, title); y -= 15
        c.setFont("Helvetica", 9)
        for it in items:
            for ln in wrap(it, "Helvetica", 9, 150):
                c.drawString(side_x, y, ln); y -= 12
        y -= 12

    # ---- main column
    y = h - 60
    c.setFont("Helvetica-Bold", 20); c.drawString(main_x, y, p["name"]); y -= 20
    c.setFont("Helvetica", 11)
    for ln in wrap(p["headline"], "Helvetica", 11, main_w):
        c.drawString(main_x, y, ln); y -= 14
    c.setFont("Helvetica", 9); c.drawString(main_x, y, p["location"]); y -= 28

    def heading(t):
        nonlocal y
        c.setFont("Helvetica-Bold", 13); c.drawString(main_x, y, t); y -= 18

    heading("Summary")
    c.setFont("Helvetica", 9.5)
    for ln in p["summary"]:
        c.drawString(main_x, y, ln); y -= 12.5
    y -= 14

    heading("Experience")
    for company, title, dates, loc, bullets in p["experience"]:
        if y < 150:                      # start a second page, like a long real profile
            c.showPage(); y = h - 60
        c.setFont("Helvetica-Bold", 11); c.drawString(main_x, y, company); y -= 14
        c.setFont("Helvetica-Bold", 10); c.drawString(main_x, y, title); y -= 13
        c.setFont("Helvetica", 9); c.drawString(main_x, y, dates); y -= 12
        c.drawString(main_x, y, loc); y -= 14
        for b in bullets:
            for i, ln in enumerate(wrap(b, "Helvetica", 9.5, main_w)):
                c.drawString(main_x, y, ln); y -= 12.5
        y -= 12

    heading("Education")
    for school, degree, yrs in p["education"]:
        c.setFont("Helvetica-Bold", 10.5); c.drawString(main_x, y, school); y -= 13
        c.setFont("Helvetica", 9.5); c.drawString(main_x, y, f"{degree} · ({yrs})"); y -= 16
    c.save()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for fname, prof in SAMPLES.items():
        draw_profile(OUT / fname, prof)
        print("wrote", OUT / fname)


if __name__ == "__main__":
    main()
