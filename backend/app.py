"""FastAPI application.

    POST /api/analyze   PDF upload or pasted text  -> parsed profile + scores + skills + rewrites
    POST /api/match     parsed profile (+ optional pasted JD) -> ranked job matches with skill gaps
    GET  /api/health    readiness + configuration (no secrets)

The server is stateless and keeps no user data: uploads are processed in memory and discarded.
Run:  uvicorn backend.app:app --reload
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .matcher.hybrid import HybridMatcher, Query
from .matcher.models import JobMatch, Weights
from .matcher.service import get_matcher, warm_up
from .optimizer.analyzer import analyze_profile
from .optimizer.models import Analysis
from .parser.models import Profile
from .parser.section_parser import parse_profile_pdf, parse_profile_text, parse_skills

log = logging.getLogger("linkedin_optimizer")
MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 60_000


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the matcher at start-up (loads jobs; builds the embedding cache on the very first run)."""
    if os.getenv("PRELOAD_MATCHER", "1") != "0":
        try:
            log.info("Loading job index (first run builds the embedding cache, ~1-2 min)...")
            warm_up()
            log.info("Job index ready.")
        except FileNotFoundError as exc:
            log.warning("%s - /api/match will return 503 until the jobs CSV exists.", exc)
    yield


app = FastAPI(title="LinkedIn Profile Optimizer & Job Matcher", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def matcher_dep() -> HybridMatcher:
    try:
        return get_matcher()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Job dataset missing: {exc}") from exc


# ------------------------------------------------------------------ schemas
class AnalyzeResponse(BaseModel):
    profile: Profile
    analysis: Analysis


class MatchRequest(BaseModel):
    profile: Profile
    job_description: str | None = Field(None, max_length=MAX_TEXT_CHARS, description="Paste one JD to score it instead of ranking the dataset")
    job_title: str = "Pasted job description"
    top_k: int = Field(10, ge=1, le=50)
    weights: Weights | None = None


class MatchResponse(BaseModel):
    mode: str                         # "dataset" | "job_description"
    weights: Weights                  # the (normalised) weights actually used
    pool_size: int
    matches: list[JobMatch]


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok", "gemini_configured": bool(config.GEMINI_API_KEY),
            "embedding_model": config.EMBEDDING_MODEL, "default_weights": config.DEFAULT_WEIGHTS,
            "jobs_csv_present": config.JOBS_CSV_PATH.exists()}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile | None = File(None), text: str = Form(""),
                  extra_skills: str = Form(""), use_llm: bool = Form(True)):
    """Parse a LinkedIn 'Save to PDF' export OR pasted profile text, then score and suggest improvements."""
    if file is None and not text.strip():
        raise HTTPException(400, "Provide a PDF file or paste your profile text.")
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(413, "PDF is larger than 5 MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "That file is not a PDF.")
        try:
            profile = parse_profile_pdf(data)
        except Exception as exc:                      # encrypted / corrupt / image-only PDFs
            log.warning("PDF parse failed: %s", exc)
            raise HTTPException(422, "Could not read this PDF (is it encrypted or a scanned image?). "
                                     "Try pasting the profile text instead.") from exc
        if not profile.raw_text.strip():
            raise HTTPException(422, "No text found in the PDF (scanned image?). Paste the profile text instead.")
    else:
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(413, "Pasted text is too long (limit 60,000 characters).")
        profile = parse_profile_text(text)

    if extra_skills.strip():                          # LinkedIn's PDF only exports the top 3 skills
        merged: dict[str, str] = {}
        for s in profile.skills + parse_skills([extra_skills]):
            merged.setdefault(s.lower(), s)           # first spelling wins ("Python" beats "python")
        profile.skills = list(merged.values())
    return AnalyzeResponse(profile=profile, analysis=analyze_profile(profile, use_llm=use_llm))


@app.post("/api/match", response_model=MatchResponse)
def match(req: MatchRequest, matcher: HybridMatcher = Depends(matcher_dep)):
    """Rank the bundled jobs for this profile, or score one pasted job description."""
    weights = (req.weights or Weights(**config.DEFAULT_WEIGHTS))
    try:
        weights = weights.normalised()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not req.profile.full_text().strip():
        raise HTTPException(422, "The profile is empty - nothing to match.")
    query = Query.from_profile(req.profile)
    if req.job_description and req.job_description.strip():
        m = matcher.match_job_description(query, req.job_description, req.job_title, weights)
        return MatchResponse(mode="job_description", weights=weights, pool_size=len(matcher.jobs), matches=[m])
    return MatchResponse(mode="dataset", weights=weights, pool_size=len(matcher.jobs),
                         matches=matcher.rank(query, req.top_k, weights))


# Serve the built React app when present (single-process deployment: `npm run build` in frontend/)
_dist = config.ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
