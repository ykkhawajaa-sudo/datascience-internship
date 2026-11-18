import pandas as pd

FEATURES=["dayofweek","month","dayofyear","weekofyear","is_weekend","is_holiday","sell_price","snap_CA",
          "lag_7","lag_14","lag_28","rolling_mean_7","rolling_mean_28"]

def add_features(frame):
    required={"date","item_id","store_id","sales"}
    if required-set(frame): raise ValueError(f"Missing columns: {sorted(required-set(frame))}")
    df=frame.copy()
    df["date"]=pd.to_datetime(df.date)
    df=df.sort_values(["store_id","item_id","date"])
    df["dayofweek"]=df.date.dt.dayofweek
    df["month"]=df.date.dt.month
    df["dayofyear"]=df.date.dt.dayofyear
    df["weekofyear"]=df.date.dt.isocalendar().week.astype(int)
    df["is_weekend"]=(df.dayofweek>=5).astype(int)
    df["is_holiday"]=df.event_name_1.fillna("").ne("").astype(int) if "event_name_1" in df else 0
    if "sell_price" not in df: df["sell_price"]=0.0
    if "snap_CA" not in df: df["snap_CA"]=0
    grouped=df.groupby(["store_id","item_id"],sort=False).sales
    for lag in (7,14,28): df[f"lag_{lag}"]=grouped.shift(lag)
    shifted=grouped.shift(1)
    for window in (7,28):
        df[f"rolling_mean_{window}"]=shifted.groupby([df.store_id,df.item_id]).rolling(window).mean().reset_index(level=[0,1],drop=True)
    return df.replace([float("inf"),-float("inf")],0).fillna(0)
