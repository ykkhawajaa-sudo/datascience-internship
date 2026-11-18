from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import plotly.express as px
from dash import Dash,dcc,html,Input,Output
from src.data_loader import load_or_demo
from src.forecasting import train_and_predict
from src.evaluation import metric_summary
from src.intervals import add_prediction_intervals
from src.inventory_optimizer import optimize_order_quantity

frame=load_or_demo()
model,train,test=train_and_predict(frame)
test=add_prediction_intervals(test)
stores=sorted(test.store_id.unique())
app=Dash(__name__,title="Demand & Inventory Intelligence")
app.layout=html.Div([
 html.H1("Demand & Inventory Intelligence"),
 html.P("Nationwide retail forecasting and replenishment decision support"),
 html.P("Synthetic demo data" if str(frame.item_id.iloc[0]).startswith("ITEM_") else "Official M5 data loaded"),
 dcc.Dropdown(id="store",options=[{"label":s,"value":s} for s in stores],value=stores[0],clearable=False),
 html.Div(id="metrics"),dcc.Graph(id="forecast"),
 html.H2("Inventory recommendation"),
 html.Label("On-hand units"),dcc.Input(id="stock",type="number",value=15,min=0),
 html.Label("Holding cost per unit"),dcc.Input(id="holding",type="number",value=1,min=0),
 html.Label("Stockout cost per unit"),dcc.Input(id="stockout",type="number",value=8,min=0),
 html.Div(id="order"),
 html.Footer("Forecasts are estimates. Validate on the official M5 dataset before operational use.")
],style={"maxWidth":"1200px","margin":"auto","fontFamily":"Arial","padding":"24px"})
@app.callback(Output("metrics","children"),Output("forecast","figure"),Input("store","value"))
def update_view(store):
 view=test[test.store_id==store]
 m=metric_summary(view,train)
 cards=[html.Div([html.Small(k),html.H3(f"{v:.1%}" if "WAPE" in k else f"{v:.3f}")]) for k,v in m.items()]
 fig=px.line(view,x="date",y=["sales","prediction","lower","upper"],title=f"Forecast — {store}")
 return cards,fig
@app.callback(Output("order","children"),Input("stock","value"),Input("holding","value"),Input("stockout","value"),Input("store","value"))
def order_view(stock,holding,stockout,store):
 if None in (stock,holding,stockout): return "Enter inventory costs and stock."
 demand=float(test[test.store_id==store].prediction.sum())
 result=optimize_order_quantity(demand,max(0,stock),max(0,holding),max(0,stockout))
 return f"Recommended order: {result['order_quantity']} units | Expected shortage: {result['expected_shortage']} | {result['status']}"
if __name__=="__main__": app.run(debug=False)
