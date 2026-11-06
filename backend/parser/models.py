"""Pydantic models describing a parsed LinkedIn profile (also used as the API schema)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    date_range: str = ""      # e.g. "March 2022 - Present"
    duration: str = ""        # e.g. "1 year 8 months" (LinkedIn prints this in the PDF)
    location: str = ""
    description: str = ""


class Education(BaseModel):
    school: str = ""
    degree: str = ""          # "Bachelor of Technology - BTech, Computer Science"
    dates: str = ""


class Profile(BaseModel):
    name: str = ""
    headline: str = ""
    location: str = ""
    about: str = ""
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    raw_text: str = ""

    def experience_text(self) -> str:
        """All experience text (titles + descriptions) as one string - used for skill extraction."""
        return "\n".join(f"{e.title} {e.company}\n{e.description}" for e in self.experience)

    def full_text(self) -> str:
        """Everything a recruiter would read, used as the query document for job matching.

        Order matters: the embedding model only reads the first ~256 word-pieces, so the densest,
        most role-defining text (headline, skills) comes first and education comes last.
        """
        parts = [self.headline, ", ".join(self.skills), self.about, self.experience_text(),
                 ", ".join(self.certifications),
                 " ".join(f"{e.degree} {e.school}" for e in self.education)]
        return "\n".join(p for p in parts if p.strip())
