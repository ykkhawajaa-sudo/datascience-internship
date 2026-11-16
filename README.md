# Customer360 — Customer Lifetime Value & Churn Intelligence System

An end-to-end data science portfolio project for customer churn prediction, survival analysis, customer lifetime value estimation, customer segmentation, and retention-offer scenario analysis.

## Project status

This repository contains the initial project scaffold. Model performance and causal uplift are **not yet validated** on the target dataset.

## Scope

- Churn classification with LightGBM
- Survival analysis with Kaplan–Meier and Cox proportional hazards
- CLV estimation with clearly documented assumptions
- Customer segmentation with K-Means
- Retention-offer uplift workflow using **simulated treatment data only** until real randomized campaign data is available
- Streamlit dashboard and optional PostgreSQL persistence

## Important methodology note

The IBM Telco Customer Churn dataset does not provide randomized retention-offer assignment and observed offer outcomes. Therefore, simulated treatment data must not be presented as causal evidence or real campaign uplift. Reliable causal uplift evaluation requires randomized campaign data or a suitable observational design with defensible assumptions.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run dashboard/app.py
```

See `docs/` for architecture, data dictionary, and presentation outline.

## Disclaimer

CLV and ROI outputs are decision-support estimates, not guaranteed future revenue. Review all assumptions before using results for business decisions.
