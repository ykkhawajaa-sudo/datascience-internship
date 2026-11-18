from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DEMAND_DATA_DIR", ROOT / "data"))

def load_m5(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    sales_path, cal_path, price_path = (data_dir / "raw" / n for n in
        ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv"))
    if not all(p.exists() for p in (sales_path, cal_path, price_path)):
        raise FileNotFoundError("M5 files missing. See README or run python -m src.prepare_demo.")
    sales, calendar, prices = pd.read_csv(sales_path), pd.read_csv(cal_path), pd.read_csv(price_path)
    id_cols = ["id","item_id","dept_id","cat_id","store_id","state_id"]
    days = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=id_cols, value_vars=days, var_name="d", value_name="sales")
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    long = long.merge(prices, on=["store_id","item_id","wm_yr_wk"], how="left")
    long["date"] = pd.to_datetime(long["date"])
    long["sales"] = long["sales"].clip(lower=0)
    long["sell_price"] = long["sell_price"].fillna(0.0)
    return long.sort_values(["store_id","item_id","date"]).reset_index(drop=True)

def load_or_demo():
    try:
        return load_m5()
    except FileNotFoundError:
        from src.prepare_demo import make_demo_data
        return make_demo_data()
