"""Optional SHAP explanations for the fitted churn pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import shap
from .features import clean_customers, feature_columns

ROOT = Path(__file__).resolve().parents[1]


def explain_churn(raw: pd.DataFrame, max_rows: int = 100):
    pipeline = joblib.load(ROOT / 'models' / 'churn_pipeline.joblib')
    cleaned = clean_customers(raw)
    X = cleaned[feature_columns(cleaned)].sample(min(max_rows, len(cleaned)), random_state=42)
    explainer = shap.Explainer(pipeline.predict_proba, X)
    return explainer, explainer(X), X
