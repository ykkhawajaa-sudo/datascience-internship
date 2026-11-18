from pathlib import Path
import joblib
from src.data_loader import load_or_demo
from src.forecasting import train_and_predict
from src.evaluation import metric_summary
from src.intervals import add_prediction_intervals
from src.hierarchy import reconcile_bottom_up

def main():
    root=Path(__file__).resolve().parents[1]
    frame=load_or_demo()
    model,train,test=train_and_predict(frame)
    test=add_prediction_intervals(test)
    metrics=metric_summary(test,train)
    out=root/"data"/"processed"; models=root/"models"
    out.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,models/"lightgbm_demand.joblib")
    test.to_csv(out/"forecast_validation.csv",index=False)
    levels=reconcile_bottom_up(test)
    for level in ("store","state","national"): levels[level].to_csv(out/f"{level}_forecast.csv",index=False)
    print("Validation metrics:",metrics)
    if str(frame.item_id.iloc[0]).startswith("ITEM_"): print("Metrics are from synthetic demo data, not official M5 data.")
if __name__=="__main__": main()
