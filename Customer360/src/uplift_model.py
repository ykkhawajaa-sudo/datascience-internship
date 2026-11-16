"""Simulated randomized-offer uplift demo; not causal evidence for real customers."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def make_randomized_offer_demo(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0)
    risk = (out["Churn"].eq("Yes").astype(float) * 0.25 + (out["tenure"] < 12) * 0.15).clip(0, 0.8)
    out["treatment"] = rng.binomial(1, 0.5, len(out))
    uplift = 0.04 + 0.12 * (out["tenure"] < 24) + 0.08 * (out["MonthlyCharges"] > out["MonthlyCharges"].median())
    base_retention = (1 - risk).clip(0.05, 0.98)
    out["retained_after_offer"] = rng.binomial(1, (base_retention + out["treatment"] * uplift).clip(0.01, 0.99))
    return out


def fit_uplift_demo(df: pd.DataFrame) -> pd.DataFrame:
    data = make_randomized_offer_demo(df)
    X = data[FEATURES].copy()
    X["SeniorCitizen"] = pd.to_numeric(X["SeniorCitizen"], errors="coerce").fillna(0)
    w, y = data["treatment"], data["retained_after_offer"]
    X_train, _, w_train, _, y_train, _ = train_test_split(X, w, y, test_size=0.25, random_state=42, stratify=w)
    treated = LGBMRegressor(n_estimators=120, verbosity=-1, random_state=42)
    control = LGBMRegressor(n_estimators=120, verbosity=-1, random_state=43)
    treated.fit(X_train[w_train == 1], y_train[w_train == 1])
    control.fit(X_train[w_train == 0], y_train[w_train == 0])
    scored = data[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
    scored["estimated_uplift"] = treated.predict(X) - control.predict(X)
    scored["uplift_data_status"] = "SIMULATED — not causal evidence"
    return scored
