import numpy as np
import pandas as pd

def make_demo_data(days=420, stores=3, items=8, seed=17):
    """Reproducible synthetic M5-shaped sample; never represents actual Walmart data."""
    rng, dates, rows = np.random.default_rng(seed), pd.date_range("2024-01-01", periods=days), []
    for s in range(stores):
        for i in range(items):
            base, trend = rng.uniform(2,18), rng.uniform(-.004,.006)
            weekly = [.9,.95,1,1,1.1,1.35,1.2]
            noise = rng.normal(0,1.5,days)
            for t,date in enumerate(dates):
                holiday = int(date.month==12 and date.day in (24,25,26))
                promo = int((t+i*3+s)%31<4)
                price = round(rng.uniform(4,35)*(0.9 if promo else 1),2)
                y = max(0,round((base+trend*t+holiday*5+promo*2)*weekly[date.dayofweek]+noise[t]))
                rows.append((date,f"ITEM_{i+1:02d}",f"STORE_{s+1:02d}",["CA","TX","WI"][s%3],y,price,promo,"Holiday" if holiday else ""))
    return pd.DataFrame(rows,columns=["date","item_id","store_id","state_id","sales","sell_price","snap_CA","event_name_1"])

if __name__ == "__main__":
    from pathlib import Path
    out=Path(__file__).resolve().parents[1]/"data"/"processed"/"demo_demand.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    make_demo_data().to_csv(out,index=False)
    print(f"Created synthetic demo data: {out}")
