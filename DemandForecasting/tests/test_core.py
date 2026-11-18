import pandas as pd
from src.features import add_features
from src.evaluation import wape
from src.intervals import add_prediction_intervals
from src.hierarchy import reconcile_bottom_up

def data():
 dates=pd.date_range("2025-01-01",periods=40)
 return pd.DataFrame({"date":list(dates)*2,"item_id":["A"]*40+["B"]*40,"store_id":["S1"]*80,
 "state_id":["CA"]*80,"sales":[3]*40+[5]*40,"sell_price":[2.0]*80,"event_name_1":[""]*80})

def test_features():
 out=add_features(data())
 assert {"lag_7","lag_28","rolling_mean_7","is_weekend"}.issubset(out.columns)

def test_wape(): assert round(wape([10,20],[8,22]),4)==.1

def test_intervals():
 out=add_prediction_intervals(pd.DataFrame({"sales":[10,12,8],"prediction":[9,11,9]}))
 assert (out.lower>=0).all() and (out.upper>=out.lower).all()

def test_hierarchy():
 frame=pd.DataFrame({"date":pd.to_datetime(["2025-01-01"]*2),"state_id":["CA"]*2,"store_id":["S1"]*2,
 "item_id":["A","B"],"prediction":[2,3]})
 assert reconcile_bottom_up(frame)["national"].prediction.iloc[0]==5
