# Sleep Doctor 🩺🌙

**AI4ALL Ignite portfolio project** — predicting sleep quality from lifestyle factors.

> **Research question:** Based on age, stress level, and activity level, can we predict sleep quality?

**Team:** Quang Doan · Shaili Halani · Nathanael Owusu · Prevailer Nchekwube · Sanskriti Poudel · Alex Saidov

## The project

We train two models on the [Sleep Health dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) (100,000 records, synthetic):

- **Path A — Linear Regression:** predicts the sleep quality score (1–10), then buckets it into Low / Medium / High
- **Path B — Random Forest:** classifies Low / Medium / High directly

We compare both paths, validate with k-fold cross-validation, interpret with permutation importance, and audit fairness across occupations and gender.

## Getting started

```bash
git clone https://github.com/Poudel-Sanskriti/Sleep_Doctor.git
cd Sleep_Doctor
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order — each one is a phase of the project:

| Notebook | Phase | What it does |
|---|---|---|
| `notebooks/01_setup_and_cleaning.ipynb` | 1 | Load the dataset, verify integrity, sanity-check values |
| `notebooks/02_statistics.ipynb` | 2 | Descriptive stats, correlations, t-test |
| `notebooks/03_visualization.ipynb` | 3 | The five key charts (saved to `figures/`) |
| *(coming)* `notebooks/04_models.ipynb` | 4 | Train, compare, and evaluate both models |

## How we work

- `main` is protected — all changes arrive by pull request
- One branch per phase: `feat/phase1-data-setup`, `feat/phase2-statistics`, …
- Commit style: `feat:` / `fix:` / `docs:` / `chore:`
- Every PR gets reviewed by a teammate before merge
