# Fraud Radar

A Streamlit dashboard for analyzing credit card fraud data: it surfaces historical fraud patterns and uses a lightweight machine-learning risk model to flag currently-unflagged transactions worth a second look. Built on a 10,000-row synthetic transaction dataset (`credit_card_fraud_10k.csv`, 151 fraud cases / 1.51% fraud rate) with a sleek, Apple-glass-inspired dark UI.

## What it does

- **KPI summary** — live transaction count, fraud count/rate, average amount, and model ROC-AUC, all responsive to the active filters.
- **Risk model & early warning** — a class-balanced logistic regression trained on the dataset scores every transaction's fraud probability. A feature-importance chart shows what drives the score, and a "top unflagged high-risk transactions" watchlist highlights the transactions the model would escalate for review — this is the "foresee fraud" angle, since the dataset has no timestamp column to support real time-series forecasting.
- **Visual analysis** — fraud rate by hour (a strong overnight 0–3am spike), fraud rate by merchant category (a weak/uniform signal, called out honestly), amount/device-trust/velocity distributions split by fraud vs. legitimate, and fraud rate conditional on foreign-transaction and location-mismatch flags.
- **Transaction explorer** — a fully filterable and sortable table (sidebar filters for category, amount, age, hour, foreign/mismatch flags, fraud status, risk-score threshold, and ID search; sort by any column via header click or the explicit sort control).

## Running it locally

```bash
cd "dashboard excersize"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`).

## Stack

- [Streamlit](https://streamlit.io/) — app framework and UI
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data loading and feature engineering
- [scikit-learn](https://scikit-learn.org/) — logistic regression risk model (`StandardScaler` + `LogisticRegression(class_weight="balanced")`)
- [Plotly](https://plotly.com/python/) — interactive charts
- Custom CSS (`backdrop-filter` blur, translucent cards) for the glass aesthetic; dark theme pinned in `.streamlit/config.toml`

## Status

Complete and working end-to-end — verified in-browser (styling, filters, sorting, KPIs, charts, risk model). This is an educational/portfolio exercise, not a production fraud system: with only 151 fraud examples in the data, the reported ROC-AUC/average-precision metrics (evaluated on a held-out stratified split) should be read as directional, not a guarantee of real-world performance.
