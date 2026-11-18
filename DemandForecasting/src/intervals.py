import numpy as np

def add_prediction_intervals(predictions,actual_col="sales",pred_col="prediction",coverage=.90):
    """Empirical absolute-residual interval; use a separate calibration window in final evaluation."""
    if not 0<coverage<1: raise ValueError("coverage must be between 0 and 1")
    radius=float(np.quantile(np.abs(predictions[actual_col]-predictions[pred_col]),coverage))
    out=predictions.copy()
    out["lower"]=np.maximum(0,out[pred_col]-radius)
    out["upper"]=out[pred_col]+radius
    out.attrs.update(nominal_coverage=coverage,calibration_radius=radius)
    return out
