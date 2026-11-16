# Data and methodology

**Dataset:** IBM Telco Customer Churn sample (7,043 customer records; source schema includes 21 columns).

**Target:** `Churn` (Yes/No). `customerID` is an identifier and must not be used as a predictor.

**Data quality:** `TotalCharges` contains blank strings for some zero-tenure customers. The baseline converts it to numeric and median-imputes missing values; assess a missingness indicator or zero-charge treatment in a deeper analysis.

**Survival limitation:** this is a one-row-per-customer snapshot, not longitudinal event history. Kaplan–Meier/Cox analysis is educational and should not be presented as precise remaining-lifetime forecasts.

**Uplift limitation:** the dataset has no randomized retention-offer assignment or post-offer outcomes. Synthetic treatment outcomes are demonstration data only, not causal evidence. Use randomized campaign records for decision-grade uplift evaluation.

**CLV limitation:** baseline CLV uses an illustrative gross-margin assumption and portfolio churn proxy. Replace with observed margin, cohort retention, and discounting assumptions for business use.
