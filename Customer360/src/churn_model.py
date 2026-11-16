"""Train and evaluate a LightGBM churn classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .features import clean_customers, feature_columns, TARGET, ID_COL

ROOT = Path(__file__).resolve().parents[1]


def train_churn_model(raw: pd.DataFrame):
    df = clean_customers(raw)
    X = df[feature_columns(df)]
    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    pipeline = Pipeline([("prep", prep), ("model", LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=20, class_weight="balanced", random_state=42, verbosity=-1))])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    metrics = {"roc_auc": float(roc_auc_score(y_test, probabilities)), "pr_auc": float(average_precision_score(y_test, probabilities)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "confusion_matrix": confusion_matrix(y_test, predictions).tolist(), "test_rows": int(len(y_test))}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ROOT / "models" / "churn_pipeline.joblib")
    (ROOT / "models" / "churn_metrics.json").write_text(json.dumps(metrics, indent=2))
    scored = df[[ID_COL, "tenure", "MonthlyCharges", "TotalCharges", TARGET]].copy()
    scored["churn_probability"] = pipeline.predict_proba(X)[:, 1]
    scored.to_csv(ROOT / "data" / "processed" / "customer_scores.csv", index=False)
    return pipeline, metrics, scored
