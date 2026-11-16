"""Run the end-to-end local training workflow."""
from src.data_loader import load_dataset
from src.features import clean_customers
from src.churn_model import train_churn_model
from src.survival_analysis import fit_survival_models
from src.uplift_model import fit_uplift_demo
from src.clv import calculate_clv
from src.segmentation import segment_customers


def main():
    raw = load_dataset()
    clean = clean_customers(raw)
    _, metrics, _ = train_churn_model(raw)
    fit_survival_models(raw)
    uplift = fit_uplift_demo(clean)
    segmented = segment_customers(calculate_clv(clean))
    output = segmented.merge(uplift[["customerID", "estimated_uplift", "uplift_data_status"]], on="customerID", how="left")
    output.to_csv("data/processed/customer360_master.csv", index=False)
    print("Churn metrics:", metrics)
    print("Saved data/processed/customer360_master.csv")


if __name__ == "__main__":
    main()
