# Data

| File | What it is |
|---|---|
| `skills_taxonomy.json` | ~590 hand-curated tech, business and soft skills in 15 categories, plus ~190 aliases (`k8s` → Kubernetes). `case_sensitive` lists names that must match exactly (`R`, `Go`, `Spark`…). |
| `jobs_sample.csv` | 1,979 **real** job postings sampled from the *LinkedIn Job Postings (2023–2024)* dataset (see below). |
| `build_jobs_sample.py` | Reproduces `jobs_sample.csv` (seeded). |
| `sample_profiles/*.pdf` | **Fictional** LinkedIn-style PDFs used by the tests and the demo. Made by `make_sample_pdf.py`. |
| `eval/labelled_set.json` | The **synthetic** evaluation set: 30 generated profiles, each with a 100-job labelled candidate pool. |
| `eval/results.json` | Machine-readable summary of the last evaluation run. |
| `.cache/` | Auto-generated embedding cache (git-ignored). |

## Job postings: provenance

`jobs_sample.csv` was built with `python data/build_jobs_sample.py` from the Hugging Face mirror
[`datastax/linkedin_job_listings`](https://huggingface.co/datasets/datastax/linkedin_job_listings)
of the Kaggle dataset *LinkedIn Job Postings (2023 – 2024)* (123,849 postings).

* The script reads ~40,000 randomly chosen rows through the Hugging Face datasets-server API (it does **not** download the 517 MB file), then stratifies: up to ~1,300 rows whose title matches a tech/business keyword, plus ~700 other rows so the pool also contains unrelated jobs.
* Descriptions are truncated to 4,000 characters; columns are limited to those the app uses.
* **Licence:** the Hugging Face mirror does not state one. Check the terms on the original Kaggle page before redistributing the data outside this coursework. The postings are public listings; this project does not scrape LinkedIn.

## Privacy

No real person's profile is stored in this repository. User profiles uploaded to the app are processed in memory and are not written to disk.
