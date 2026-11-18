def reconcile_bottom_up(frame,value_col="prediction"):
    required={"state_id","store_id","item_id","date",value_col}
    if required-set(frame): raise ValueError(f"Missing columns: {sorted(required-set(frame))}")
    store=frame.groupby(["state_id","store_id","date"],as_index=False)[value_col].sum()
    state=store.groupby(["state_id","date"],as_index=False)[value_col].sum()
    national=state.groupby("date",as_index=False)[value_col].sum()
    return {"bottom":frame.copy(),"store":store,"state":state,"national":national}
