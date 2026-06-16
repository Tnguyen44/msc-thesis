# Master Thesis — ESG Rating Inconsistencies & Portfolio Outcomes

**Author:** Thomas Nguyen · **Programme:** MSc Big Data & Finance

Machine learning pipeline to identify ESG rating inconsistencies (vs. firm fundamentals) and evaluate portfolio implications.

## Repository layout

```
├── notebooks/          # Jupyter notebooks (run in order)
│   ├── 01_exploratory_eda_esg.ipynb
│   └── 02_intermediate_report_appendix.ipynb
├── src/                # Standalone Python scripts (mirror notebook logic)
│   ├── eda_esg_thesis.py
│   └── intermediate_analysis.py
├── data/
│   ├── raw/            # Local raw inputs (not committed)
│   └── external/       # Optional ESG CSV when Yahoo API is unavailable
├── outputs/
│   ├── eda/            # EDA figures + master_dataset.csv
│   └── intermediate/   # Hypothesis tests (H1–H4) + figs/
├── docs/
│   ├── reports/        # Thesis milestones (intro, preliminary, intermediate)
│   └── drafts/         # Submission notes
└── references/         # Academic papers (PDF)
```

## Setup

```bash
pip install yfinance pandas numpy matplotlib seaborn tqdm scikit-learn \
            xgboost shap statsmodels scipy
```

## Workflow

1. **Exploratory EDA** — open `notebooks/01_exploratory_eda_esg.ipynb` (or run `python src/eda_esg_thesis.py`).  
   Writes to `outputs/eda/`. If Yahoo ESG is unavailable, set `ESG_SOURCE = "csv"` and place a file under `data/external/`.

2. **Intermediate analysis** — ensure `outputs/eda/master_dataset.csv` exists, then run `python src/intermediate_analysis.py` or `notebooks/02_intermediate_report_appendix.ipynb`.  
   Writes to `outputs/intermediate/`.

## Reports

| Document | Location |
|----------|----------|
| Thesis introduction | `docs/reports/thesis_intro_thomas_nguyen.pdf` |
| Preliminary work (Mar 30) | `docs/reports/preliminary_work_march30_thomas_nguyen.pdf` |
| Intermediate report (Apr 30) | `docs/reports/intermediate_report_april30_thomas_nguyen.pdf` |

## License

Academic use — thesis work in progress.
