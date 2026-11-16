"""Run the end-to-end Customer360 training workflow."""
from pathlib import Path

from src.data_loader import load_dataset
from src.features import clean_customers
from src.churn_model import train_churn_model
from src.survival_analysis import fit_survival_models
from src.uplift_model import fit_uplift_demo
from src.clv import calculate_clv
from src.segmentation import segment_customers

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = load_dataset()
    clean = clean_customers(raw)
    _, metrics, _ = train_churn_model(raw)
    fit_survival_models(raw)
    uplift = fit_uplift_demo(clean)
    segmented = segment_customers(calculate_clv(clean))
    output = segmented.merge(
        uplift[["customerID", "estimated_uplift", "uplift_data_status"]],
        on="customerID",
        how="left",
    )
    destination = ROOT / "data" / "processed" / "customer360_master.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print("Churn metrics:", metrics)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
