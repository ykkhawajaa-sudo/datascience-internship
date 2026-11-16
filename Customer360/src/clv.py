"""Transparent baseline customer lifetime value estimate."""
import pandas as pd


def calculate_clv(df: pd.DataFrame, gross_margin: float = 0.35, annual_discount_rate: float = 0.10) -> pd.DataFrame:
    if not 0 < gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1")
    out = df.copy()
    monthly_margin = out["MonthlyCharges"].fillna(0) * gross_margin
    monthly_churn = (out["Churn"].eq("Yes").mean() / 12) if "Churn" in out else 0.03
    retention = max(0.01, min(0.99, 1 - monthly_churn))
    monthly_discount = (1 + annual_discount_rate) ** (1 / 12) - 1
    denominator = max(0.001, 1 + monthly_discount - retention)
    out["estimated_clv"] = monthly_margin * retention / denominator
    out["clv_assumption"] = f"{gross_margin:.0%} illustrative gross margin; portfolio-average churn proxy"
    return out
