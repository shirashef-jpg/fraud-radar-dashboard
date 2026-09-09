"""Fraud analysis & early-warning dashboard for credit_card_fraud_10k.csv."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "credit_card_fraud_10k.csv"

NUMERIC_COLS = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]
CATEGORICAL_COLS = ["merchant_category"]

PRETTY_NAMES = {
    "amount": "Amount",
    "transaction_hour": "Hour of day",
    "foreign_transaction": "Foreign transaction",
    "location_mismatch": "Location mismatch",
    "device_trust_score": "Device trust score",
    "velocity_last_24h": "Velocity (24h)",
    "cardholder_age": "Cardholder age",
}

ACCENT = "#5eead4"
ACCENT2 = "#818cf8"
MUTED = "#9aa3b8"

GLASS_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at 15% 8%, #1b2140 0%, #0b0d17 55%, #05060b 100%);
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
[data-testid="stSidebar"] > div:first-child {{
    background: rgba(20, 22, 34, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.08);
}}
[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
[data-testid="stMetricValue"] {{ color: {ACCENT}; }}

div[class*="st-key-"] {{
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 1.4rem 1.6rem 1.1rem 1.6rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 1.3rem;
}}

[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

h1, h2, h3 {{ color: #f5f6fa; letter-spacing: 0.2px; }}
.hero-title {{
    font-size: 2.3rem; font-weight: 700; margin-bottom: 0.1rem;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0; }}
.section-title {{ font-size: 1.15rem; font-weight: 600; color: #f5f6fa; margin-bottom: 0.2rem; }}
.section-caption {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 0.8rem; }}
.accent {{ color: {ACCENT}; }}
</style>
"""

TRANSPARENT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e6e8ef",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def style_fig(fig):
    fig.update_layout(**TRANSPARENT_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


def pretty_feature_name(col: str) -> str:
    if col in PRETTY_NAMES:
        return PRETTY_NAMES[col]
    if col.startswith("merchant_category_"):
        return f"Merchant: {col.split('merchant_category_', 1)[1]}"
    return col


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-1, 3, 7, 11, 15, 19, 23]
    labels = ["Late night (0-3)", "Early morning (4-7)", "Morning (8-11)",
              "Afternoon (12-15)", "Evening (16-19)", "Night (20-23)"]
    df["hour_bucket"] = pd.cut(df["transaction_hour"], bins=bins, labels=labels)
    return df


@st.cache_resource
def train_risk_model(_df: pd.DataFrame):
    df = _df
    X = pd.get_dummies(df[NUMERIC_COLS + CATEGORICAL_COLS], columns=CATEGORICAL_COLS, drop_first=True)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba_test),
        "avg_precision": average_precision_score(y_test, y_proba_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_fraud_test": int(y_test.sum()),
    }

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    all_proba = pipeline.predict_proba(X)[:, 1]
    X_scaled = scaler.transform(X)
    contributions = X_scaled * clf.coef_[0]
    top_idx = np.argmax(np.abs(contributions), axis=1)
    top_driver = np.array(X.columns)[top_idx]
    top_driver = [pretty_feature_name(c) for c in top_driver]

    coef_df = pd.DataFrame({
        "feature": [pretty_feature_name(c) for c in X.columns],
        "coefficient": clf.coef_[0],
    }).sort_values("coefficient", key=np.abs, ascending=False)

    scored = df.copy()
    scored["risk_score"] = all_proba
    scored["top_risk_driver"] = top_driver

    return metrics, coef_df, scored


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if f["categories"]:
        mask &= df["merchant_category"].isin(f["categories"])

    mask &= df["amount"].between(f["amount_range"][0], f["amount_range"][1])
    mask &= df["cardholder_age"].between(f["age_range"][0], f["age_range"][1])

    if f["night_only"]:
        mask &= df["transaction_hour"].between(0, 3)
    else:
        mask &= df["transaction_hour"].between(f["hour_range"][0], f["hour_range"][1])

    if f["foreign"] == "Foreign only":
        mask &= df["foreign_transaction"] == 1
    elif f["foreign"] == "Domestic only":
        mask &= df["foreign_transaction"] == 0

    if f["mismatch"] == "Mismatch only":
        mask &= df["location_mismatch"] == 1
    elif f["mismatch"] == "No mismatch only":
        mask &= df["location_mismatch"] == 0

    if f["fraud_status"] == "Fraud only":
        mask &= df["is_fraud"] == 1
    elif f["fraud_status"] == "Legitimate only":
        mask &= df["is_fraud"] == 0

    mask &= df["risk_score"] >= f["risk_threshold"]

    if f["search"]:
        mask &= df["transaction_id"].astype(str).str.contains(f["search"].strip(), na=False)

    return df[mask]


def glass_section(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Fraud Radar", page_icon="🛰️", layout="wide")
    st.markdown(GLASS_CSS, unsafe_allow_html=True)

    raw_df = engineer_features(load_data())
    metrics, coef_df, scored_df = train_risk_model(raw_df)

    st.markdown('<div class="hero-title">🛰️ Fraud Radar</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Detect historical fraud patterns and surface currently-unflagged '
        'transactions the model would escalate for review. Educational dataset — 10,000 transactions, '
        '151 confirmed fraud cases.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # ---------------- Sidebar filters ----------------
    with st.sidebar:
        st.markdown("### Filters")
        categories = st.multiselect(
            "Merchant category",
            sorted(raw_df["merchant_category"].unique()),
            default=[],
            help="Leave empty to include all categories.",
        )
        amount_min, amount_max = float(raw_df["amount"].min()), float(raw_df["amount"].max())
        amount_range = st.slider("Amount range ($)", amount_min, amount_max, (amount_min, amount_max))

        age_min, age_max = int(raw_df["cardholder_age"].min()), int(raw_df["cardholder_age"].max())
        age_range = st.slider("Cardholder age", age_min, age_max, (age_min, age_max))

        night_only = st.checkbox("Night hours only (0–3am)", value=False,
                                  help="Overrides the hour slider below with the highest-risk window.")
        hour_range = st.slider("Transaction hour", 0, 23, (0, 23), disabled=night_only)

        foreign = st.selectbox("Foreign transaction", ["All", "Foreign only", "Domestic only"])
        mismatch = st.selectbox("Location mismatch", ["All", "Mismatch only", "No mismatch only"])
        fraud_status = st.selectbox("Fraud status", ["All", "Fraud only", "Legitimate only"])

        risk_threshold = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)
        search = st.text_input("Search transaction ID")

        filters = dict(
            categories=categories, amount_range=amount_range, age_range=age_range,
            night_only=night_only, hour_range=hour_range, foreign=foreign,
            mismatch=mismatch, fraud_status=fraud_status, risk_threshold=risk_threshold,
            search=search,
        )

    filtered_df = apply_filters(scored_df, filters)

    # ---------------- KPI row ----------------
    total_txns = len(filtered_df)
    fraud_count = int(filtered_df["is_fraud"].sum())
    fraud_rate = (fraud_count / total_txns * 100) if total_txns else 0.0
    avg_amount = filtered_df["amount"].mean() if total_txns else 0.0

    with st.container(key="kpi-row"):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Transactions", f"{total_txns:,}")
        c2.metric("Fraud cases", f"{fraud_count:,}")
        c3.metric("Fraud rate", f"{fraud_rate:.2f}%")
        c4.metric("Avg. amount", f"${avg_amount:,.2f}")
        c5.metric("Model ROC-AUC (holdout)", f"{metrics['roc_auc']:.3f}")

    # ---------------- Risk & early-warning section ----------------
    with st.container(key="risk-section"):
        glass_section(
            "🔮 Risk model & early warning",
            f"Logistic regression trained on all {len(raw_df):,} transactions with class-balanced "
            f"weighting (only {int(raw_df['is_fraud'].sum())} historical fraud cases — treat metrics as "
            f"directional, not production-grade). Held-out performance: ROC-AUC {metrics['roc_auc']:.3f}, "
            f"average precision {metrics['avg_precision']:.3f}, evaluated on a {metrics['n_test']:,}-row "
            f"stratified test split ({metrics['n_fraud_test']} fraud cases) never seen during training.",
        )

        rc1, rc2 = st.columns([2, 3])
        with rc1:
            st.markdown("**What drives the risk score**")
            fig_coef = px.bar(
                coef_df, x="coefficient", y="feature", orientation="h",
                color="coefficient", color_continuous_scale=["#818cf8", "#1b2140", ACCENT],
                color_continuous_midpoint=0,
            )
            fig_coef.update_layout(showlegend=False, coloraxis_showscale=False, height=380,
                                    yaxis_title="", xaxis_title="Standardized coefficient")
            st.plotly_chart(style_fig(fig_coef), use_container_width=True)

        with rc2:
            st.markdown("**Top unflagged high-risk transactions**")
            st.caption(
                "Transactions not historically labeled as fraud, ranked by predicted risk. "
                "Scored on the full dataset for exploration — held-out metrics above are the honest number."
            )
            watchlist_cols = ["transaction_id", "amount", "merchant_category", "transaction_hour",
                               "device_trust_score", "velocity_last_24h", "risk_score", "top_risk_driver"]
            watchlist = (
                filtered_df[filtered_df["is_fraud"] == 0]
                .sort_values("risk_score", ascending=False)
                .head(15)[watchlist_cols]
            )
            st.dataframe(
                watchlist,
                use_container_width=True,
                hide_index=True,
                height=380,
                column_config={
                    "transaction_id": "ID",
                    "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "merchant_category": "Category",
                    "transaction_hour": "Hour",
                    "device_trust_score": "Trust score",
                    "velocity_last_24h": "Velocity",
                    "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
                    "top_risk_driver": "Top driver",
                },
            )

    # ---------------- Visual analysis ----------------
    with st.container(key="analysis-section"):
        glass_section("📊 Visual analysis", "All charts reflect the current sidebar filters.")

        hour_stats = (
            filtered_df.groupby("transaction_hour")
            .agg(fraud_rate=("is_fraud", "mean"), count=("is_fraud", "size"))
            .reindex(range(24), fill_value=0)
            .reset_index()
        )
        hour_stats["fraud_rate"] *= 100

        cat_stats = (
            filtered_df.groupby("merchant_category")
            .agg(fraud_rate=("is_fraud", "mean"), count=("is_fraud", "size"))
            .reset_index()
        )
        cat_stats["fraud_rate"] *= 100

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("**Fraud rate by hour of day** — overnight hours stand out")
            fig_hour = px.bar(hour_stats, x="transaction_hour", y="fraud_rate",
                               color="fraud_rate", color_continuous_scale=["#1b2140", ACCENT])
            fig_hour.update_layout(coloraxis_showscale=False, xaxis_title="Hour", yaxis_title="Fraud rate (%)")
            st.plotly_chart(style_fig(fig_hour), use_container_width=True)
        with r1c2:
            st.markdown("**Fraud rate by merchant category**")
            st.caption("Signal is weak here — rates are roughly uniform across categories.")
            fig_cat = px.bar(cat_stats.sort_values("fraud_rate", ascending=False),
                              x="merchant_category", y="fraud_rate",
                              color="fraud_rate", color_continuous_scale=["#1b2140", ACCENT2])
            fig_cat.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Fraud rate (%)")
            st.plotly_chart(style_fig(fig_cat), use_container_width=True)

        label_map = {0: "Legitimate", 1: "Fraud"}
        dist_df = filtered_df.assign(status=filtered_df["is_fraud"].map(label_map))
        color_map = {"Legitimate": ACCENT2, "Fraud": ACCENT}

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown("**Amount: fraud vs. legitimate**")
            fig_amt = px.box(dist_df, x="status", y="amount", color="status", color_discrete_map=color_map)
            fig_amt.update_layout(showlegend=False, xaxis_title="", yaxis_title="Amount ($)")
            st.plotly_chart(style_fig(fig_amt), use_container_width=True)
        with r2c2:
            st.markdown("**Device trust score**")
            fig_trust = px.histogram(dist_df, x="device_trust_score", color="status",
                                      histnorm="percent", barmode="overlay", opacity=0.65,
                                      color_discrete_map=color_map)
            fig_trust.update_layout(xaxis_title="Trust score", yaxis_title="Share (%)",
                                     legend_title="")
            st.plotly_chart(style_fig(fig_trust), use_container_width=True)
        with r2c3:
            st.markdown("**Velocity (txns / 24h)**")
            fig_vel = px.histogram(dist_df, x="velocity_last_24h", color="status",
                                    histnorm="percent", barmode="overlay", opacity=0.65,
                                    color_discrete_map=color_map)
            fig_vel.update_layout(xaxis_title="Velocity", yaxis_title="Share (%)", legend_title="")
            st.plotly_chart(style_fig(fig_vel), use_container_width=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.markdown("**Fraud rate: foreign vs. domestic**")
            fx = (filtered_df.groupby("foreign_transaction")["is_fraud"].mean() * 100).reindex([0, 1], fill_value=0)
            fig_fx = px.bar(x=["Domestic", "Foreign"], y=fx.values,
                             color=["Domestic", "Foreign"], color_discrete_map={"Domestic": ACCENT2, "Foreign": ACCENT})
            fig_fx.update_layout(showlegend=False, xaxis_title="", yaxis_title="Fraud rate (%)")
            st.plotly_chart(style_fig(fig_fx), use_container_width=True)
        with r3c2:
            st.markdown("**Fraud rate: location mismatch**")
            lm = (filtered_df.groupby("location_mismatch")["is_fraud"].mean() * 100).reindex([0, 1], fill_value=0)
            fig_lm = px.bar(x=["No mismatch", "Mismatch"], y=lm.values,
                             color=["No mismatch", "Mismatch"], color_discrete_map={"No mismatch": ACCENT2, "Mismatch": ACCENT})
            fig_lm.update_layout(showlegend=False, xaxis_title="", yaxis_title="Fraud rate (%)")
            st.plotly_chart(style_fig(fig_lm), use_container_width=True)

    # ---------------- Transaction explorer ----------------
    with st.container(key="explorer-section"):
        glass_section("🔍 Transaction explorer", "Click any column header to sort, or use the controls below.")

        sort_cols = ["risk_score", "amount", "transaction_hour", "device_trust_score",
                     "velocity_last_24h", "cardholder_age", "transaction_id"]
        sc1, sc2, sc3 = st.columns([2, 1, 3])
        with sc1:
            sort_by = st.selectbox("Sort by", sort_cols, index=0)
        with sc2:
            ascending = st.toggle("Ascending", value=False)

        table_cols = ["transaction_id", "amount", "merchant_category", "transaction_hour",
                      "foreign_transaction", "location_mismatch", "device_trust_score",
                      "velocity_last_24h", "cardholder_age", "is_fraud", "risk_score"]
        table_df = filtered_df.sort_values(sort_by, ascending=ascending)[table_cols]

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=420,
            column_config={
                "transaction_id": "ID",
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "merchant_category": "Category",
                "transaction_hour": "Hour",
                "foreign_transaction": st.column_config.CheckboxColumn("Foreign"),
                "location_mismatch": st.column_config.CheckboxColumn("Mismatch"),
                "device_trust_score": "Trust score",
                "velocity_last_24h": "Velocity",
                "cardholder_age": "Age",
                "is_fraud": st.column_config.CheckboxColumn("Fraud"),
                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        st.caption(f"Showing {len(table_df):,} of {len(raw_df):,} total transactions.")


if __name__ == "__main__":
    main()
