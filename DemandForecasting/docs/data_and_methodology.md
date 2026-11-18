# Data and methodology

## Dataset
Use the M5 Forecasting competition sales, calendar and sell-price CSV files. Sales begin in wide format (one column per day); the loader melts them into item-store-date observations and joins calendar and weekly prices.

## Features
Calendar fields, holiday/event flags, SNAP indicator, sell price, 7/14/28-day lags and shifted 7/28-day rolling averages. Historical rolling features are shifted to prevent target leakage.

## Evaluation
Chronological 28-day holdout. WAPE = sum absolute error / sum actual demand. RMSSE scales forecast RMSE by the mean squared first difference of the training series. Compare with seasonal naive.

## Uncertainty and hierarchy
Empirical absolute validation residual quantiles form symmetric prediction intervals. A final benchmark should reserve a distinct calibration period and report achieved coverage and interval width. Bottom-up aggregation sums item-store forecasts to store, state and national totals, ensuring arithmetic coherence.

## Inventory
OR-Tools minimizes leftover-stock holding cost plus unmet-demand stockout cost, with integer pack sizes and optional budget. Lead-time demand can be supplied separately.

## Extensions
Prophet is an optional per-series baseline. TFT (PyTorch Forecasting) and Darts hierarchical reconciliation are extension tracks; do not report them as completed until trained and evaluated on a defined time split.
