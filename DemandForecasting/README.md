# Nationwide Demand Forecasting & Inventory Optimization

An end-to-end retail analytics project using the Walmart M5 Forecasting dataset. It forecasts item demand across stores, evaluates forecast quality, builds uncertainty intervals, reconciles bottom-level forecasts to store/state totals, and recommends replenishment quantities under holding and stockout costs.

## Included
- M5 wide-to-long loader and clearly labelled synthetic demo fallback
- Calendar, price, holiday, lag and rolling-demand features
- Seasonal-naive baseline, LightGBM, optional Prophet
- WAPE and RMSSE evaluation; empirical residual prediction intervals
- Bottom-up store/state/national aggregation
- OR-Tools integer replenishment optimization
- Plotly Dash dashboard, tests and project documentation

## Quick start (Windows)
```powershell
cd DemandForecasting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.prepare_demo
python -m src.train
python -m dashboard.app
```
Open http://127.0.0.1:8050.

## Official dataset
Download `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` from [Kaggle M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/raw/`. Do not commit the dataset. Without these files, the app uses generated synthetic demo data; demo metrics are not M5 benchmark results.

## Methodology and limitations
See `docs/`. Intervals are empirical validation-residual quantiles and require coverage checks on a separate calibration window. Bottom-up aggregation guarantees arithmetic consistency, not statistical optimality. The optimizer assumes linear holding/shortage costs and optional pack-size/budget constraints. TFT/PyTorch Forecasting and Darts are documented extension tracks, not claimed as trained results in this baseline implementation.
