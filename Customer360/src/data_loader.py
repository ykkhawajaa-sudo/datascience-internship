"""Download and load the public IBM Telco Customer Churn dataset."""
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "Telco-Customer-Churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    DATA_PATH.write_bytes(response.content)
    return DATA_PATH


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(download_dataset())
    required = {"customerID", "tenure", "Churn", "MonthlyCharges", "TotalCharges"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset is missing columns: {sorted(required - set(df.columns))}")
    return df
