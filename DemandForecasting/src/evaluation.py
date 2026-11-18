import numpy as np

def wape(actual,predicted):
    a,p=np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    return float(np.abs(a-p).sum()/np.abs(a).sum()) if np.abs(a).sum() else float("nan")

def rmsse(actual,predicted,training_series):
    a,p,tr=map(lambda x:np.asarray(x,dtype=float),(actual,predicted,training_series))
    if len(tr)<2:return float("nan")
    scale=np.mean(np.diff(tr)**2)
    return float(np.sqrt(np.mean((a-p)**2)/scale)) if scale>0 else float("nan")

def metric_summary(test,train):
    return {"WAPE":wape(test.sales,test.prediction),"RMSSE":rmsse(test.sales,test.prediction,train.sales),
            "Naive WAPE":wape(test.sales,test.naive_prediction)}
