# Architecture

IBM Telco CSV → validation and cleaning → feature engineering → churn model, survival analysis, CLV, segmentation, and a clearly labelled synthetic uplift demonstration → persisted outputs → Streamlit dashboard.

The churn classifier uses a stratified holdout and a scikit-learn preprocessing pipeline. IDs and target labels are excluded from predictors. Survival analysis treats observed tenure as duration and churn as the event; this is a simplified snapshot-based demonstration. The dataset has no retention campaign assignment, so synthetic uplift must not be interpreted as causal evidence.
