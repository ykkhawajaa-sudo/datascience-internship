"""Customer value segmentation using K-Means."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    if n_clusters < 2 or n_clusters >= len(df):
        raise ValueError("n_clusters must be >=2 and smaller than row count")
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X = df[cols].fillna(df[cols].median())
    scaled = StandardScaler().fit_transform(X)
    out = df.copy()
    out["segment"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled).astype(str)
    return out
