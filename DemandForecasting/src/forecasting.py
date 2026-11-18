import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from src.features import add_features, FEATURES

def time_split(frame,test_days=28):
    cutoff=pd.to_datetime(frame.date).max()-pd.Timedelta(days=test_days-1)
    return frame[pd.to_datetime(frame.date)<cutoff].copy(),frame[pd.to_datetime(frame.date)>=cutoff].copy()

def seasonal_naive(train,test):
    lookup=train.set_index(["store_id","item_id","date"]).sales
    preds=[]
    for r in test.itertuples():
        key=(r.store_id,r.item_id,pd.Timestamp(r.date)-pd.Timedelta(days=7))
        history=train[(train.store_id==r.store_id)&(train.item_id==r.item_id)].sales
        preds.append(float(lookup.get(key,history.tail(28).mean())))
    return np.maximum(0,np.asarray(preds))

def fit_lightgbm(train):
    model=LGBMRegressor(n_estimators=300,learning_rate=.04,num_leaves=31,random_state=42,verbosity=-1)
    model.fit(train[FEATURES],train.sales)
    return model

def train_and_predict(frame,test_days=28):
    featured=add_features(frame)
    train_raw,test_raw=time_split(featured,test_days)
    train=train_raw[train_raw.lag_28>0].copy()
    if train.empty: train=train_raw.copy()
    model=fit_lightgbm(train)
    test=test_raw.copy()
    test["prediction"]=np.maximum(0,model.predict(test[FEATURES]))
    test["naive_prediction"]=seasonal_naive(train_raw,test)
    return model,train,test

def prophet_forecast(group,periods=28):
    try: from prophet import Prophet
    except ImportError as exc: raise RuntimeError("Install prophet for this optional baseline.") from exc
    data=group[["date","sales"]].rename(columns={"date":"ds","sales":"y"})
    model=Prophet(weekly_seasonality=True,yearly_seasonality=True,daily_seasonality=False)
    model.fit(data)
    future=model.make_future_dataframe(periods=periods)
    return model.predict(future).tail(periods)[["ds","yhat","yhat_lower","yhat_upper"]]
