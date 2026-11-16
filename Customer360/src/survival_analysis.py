"""Kaplan-Meier and Cox proportional-hazards analysis."""
from pathlib import Path
import joblib
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from .features import clean_customers

ROOT = Path(__file__).resolve().parents[1]


def fit_survival_models(raw: pd.DataFrame):
    df = clean_customers(raw)
    duration = df["tenure"].clip(lower=1)
    event = df["Churn"].eq("Yes").astype(int)
    kmf = KaplanMeierFitter().fit(duration, event_observed=event, label="All customers")
    cox_data = pd.DataFrame({"duration": duration, "event": event, "MonthlyCharges": df["MonthlyCharges"], "SeniorCitizen": pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0), "PaperlessBilling": df["PaperlessBilling"].eq("Yes").astype(int)})
    cph = CoxPHFitter(penalizer=0.1).fit(cox_data, duration_col="duration", event_col="event")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(kmf, ROOT / "models" / "kaplan_meier.joblib")
    joblib.dump(cph, ROOT / "models" / "cox_model.joblib")
    return kmf, cph
