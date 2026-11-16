import pandas as pd
from src.features import clean_customers


def test_total_charges_are_numeric_and_missing_values_filled():
    raw = pd.DataFrame({
        "customerID": ["A", "B"], "tenure": [0, 12], "Churn": ["No", "Yes"],
        "TotalCharges": [" ", "120"], "MonthlyCharges": [50, 10], "SeniorCitizen": [0, 1],
        "PhoneService": ["Yes", "No"], "MultipleLines": ["No", "No"],
    })
    cleaned = clean_customers(raw)
    assert cleaned["TotalCharges"].notna().all()
    assert cleaned["service_count"].tolist() == [1, 0]
