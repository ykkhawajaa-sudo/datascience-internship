from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Customer360', page_icon='📊', layout='wide')
st.title('Customer360')
st.caption('Customer Lifetime Value & Churn Intelligence System')
master = ROOT / 'data' / 'processed' / 'customer360_master.csv'
metrics_file = ROOT / 'models' / 'churn_metrics.json'
if not master.exists():
    st.warning('No trained outputs found. Run python -m src.train from the Customer360 folder first.')
    st.stop()
df = pd.read_csv(master)
metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
a, b, c, d = st.columns(4)
a.metric('Customers', f'{len(df):,}')
b.metric('Historical churn', f"{df['Churn'].eq('Yes').mean():.1%}")
c.metric('Avg monthly charge', f"${df['MonthlyCharges'].mean():,.2f}")
d.metric('Avg estimated CLV', f"${df['estimated_clv'].mean():,.0f}")
left, right = st.columns(2)
with left:
    st.subheader('Churn by contract')
    st.plotly_chart(px.histogram(df, x='Contract', color='Churn', barmode='group'), use_container_width=True)
with right:
    st.subheader('Estimated CLV by segment')
    st.plotly_chart(px.box(df, x='segment', y='estimated_clv', color='segment'), use_container_width=True)
st.subheader('Churn model evaluation')
if metrics:
    cols = st.columns(4)
    for col, key in zip(cols, ['roc_auc', 'pr_auc', 'precision', 'recall']):
        col.metric(key.upper().replace('_', ' '), f'{metrics[key]:.3f}')
st.subheader('Retention offer ROI scenario')
x, y, z = st.columns(3)
discount = x.slider('Monthly discount ($)', 0, 30, 10)
months = y.slider('Offer duration (months)', 1, 12, 3)
increment = z.slider('Assumed incremental retention (%)', 0, 50, 10)
targets = df[df['Churn'].eq('Yes')]
saved = len(targets) * increment / 100
cost = len(targets) * discount * months
revenue = saved * targets['MonthlyCharges'].mean() * months if len(targets) else 0
roi = (revenue - cost) / cost * 100 if cost else 0
p, q, r = st.columns(3)
p.metric('Customers in scenario', f'{len(targets):,}')
q.metric('Estimated offer cost', f'${cost:,.2f}')
r.metric('Illustrative net ROI', f'{roi:,.1f}%')
st.caption('Scenario only, not a causal forecast. Use randomized campaign outcomes before making live retention decisions.')
st.subheader('Customer records')
st.dataframe(df.head(100), use_container_width=True)
