"""Capture the README screenshots from a RUNNING app (uses the sample, fictional profile).

    pip install playwright              # dev-only; uses your installed Edge/Chrome, no browser download
    uvicorn backend.app:app --port 8000 # in another terminal (after `npm run build` in frontend/)
    python scripts/take_screenshots.py  # -> docs/screenshots/*.png

Pass --url http://localhost:5173 to shoot the Vite dev server instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
SAMPLE_JD = ("Senior Machine Learning Engineer. Build NLP models in Python and PyTorch, deploy them with Docker and "
             "Kubernetes on AWS, and maintain MLOps pipelines with Airflow and MLflow. Strong SQL and Spark required.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:
            browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=1)
        page.goto(args.url)

        page.get_by_role("tab", name="Paste text").click()
        page.get_by_role("button", name="Use a sample (fictional) profile").click()
        page.screenshot(path=OUT / "01-upload.png")

        page.get_by_role("button", name="Analyse my profile").click()
        page.get_by_text("Section scores").wait_for()
        page.get_by_text("Show breakdown").nth(1).click()          # expand the About section's metrics
        page.wait_for_timeout(900)                                   # let the bars finish animating
        page.screenshot(path=OUT / "02-profile-report.png", full_page=True)

        page.get_by_role("button", name="Find matching jobs").click()
        page.get_by_text("Top 10 matches").wait_for(timeout=60_000)
        page.get_by_text("Show 3 tips to close the gap").first.click()
        page.wait_for_timeout(900)
        page.screenshot(path=OUT / "03-job-matches.png")

        page.get_by_role("tab", name="Paste a JD").click()
        page.get_by_label("Job description").fill(SAMPLE_JD)
        page.get_by_role("button", name="Score this job").click()
        page.get_by_text("Your job description").wait_for(timeout=30_000)
        page.wait_for_timeout(900)
        page.screenshot(path=OUT / "04-job-description-fit.png")
        browser.close()
    print("saved:", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
