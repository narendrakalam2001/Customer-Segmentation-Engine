# ============================================================
# MONITORING DASHBOARD — Customer Segmentation Engine (Streamlit)
# ============================================================
# Sidebar: 🔮 Try a Prediction (manual RFM input, calls /predict live)
#          🎮 Simulate Live Traffic (scenario picker, calls /predict N times)
#          ⚙️ API URL (configurable — defaults to http://localhost:8000)
#
# 5 tabs:
#   1. Alerts               — PSI drift + noise-ratio + stability alerts
#   2. Champion-Challenger   — current champion + promotion history
#   3. KPIs + charts         — segment sizes, revenue by persona, RFM scatter
#   4. PSI drift             — feature-level drift vs reference distribution
#   5. Recent predictions    — live /predict traffic log
# ============================================================

import os
import json
import sys
import time

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODEL_DIR, PSI_MODERATE, PSI_HIGH
from src.metrics import psi

st.set_page_config(page_title="Customer Segmentation Monitor", layout="wide")
st.title("🛍️ Customer Segmentation Engine — Monitoring Dashboard")

LOG_PATH = "logs/prediction_logs.csv"
RFM_COLS = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "UniqueProducts", "Tenure"]


# ── Data loaders ────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_registry():
    path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_model_card(card_path):
    if not card_path or not os.path.exists(card_path):
        return None
    with open(card_path) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_challenger_log():
    path = os.path.join(MODEL_DIR, "challenger_log.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_profile():
    path = os.path.join(MODEL_DIR, "cluster_profile.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data(ttl=30)
def load_impact():
    path = os.path.join(MODEL_DIR, "business_impact.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data(ttl=30)
def load_reference_rfm():
    path = os.path.join(MODEL_DIR, "reference_rfm_distribution.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_predictions():
    # Not cached: needs to reflect predictions sent seconds ago from the sidebar.
    return pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()


# ============================================================
# SIDEBAR — API config, manual prediction form, live simulation
# ============================================================

st.sidebar.header("⚙️ API Connection")
api_base = st.sidebar.text_input(
    "Segmentation API URL", value=st.session_state.get("api_base", "https://customer-segmentation-engine-4njy.onrender.com"),
    help="Base URL of `scripts/run_api.py` (FastAPI). Change this if the API runs on a different host/port.",
)
st.session_state["api_base"] = api_base

st.sidebar.markdown("---")
st.sidebar.header("🔮 Try a Prediction")
st.sidebar.caption("Enter a customer's RFM profile and get a live segment assignment from the API.")

with st.sidebar.form("predict_form"):
    recency         = st.number_input("Recency (days since last purchase)", min_value=0, value=15)
    frequency       = st.number_input("Frequency (distinct invoices)", min_value=0, value=8)
    monetary        = st.number_input("Monetary (total spend)", min_value=0.0, value=2450.50)
    avg_basket      = st.number_input("Avg Basket Value", min_value=0.0, value=306.31)
    unique_products = st.number_input("Unique Products", min_value=0, value=34)
    tenure          = st.number_input("Tenure (days as customer)", min_value=0, value=210)
    submitted = st.form_submit_button("Predict Segment")

if submitted:
    payload = {
        "Recency": float(recency), "Frequency": float(frequency), "Monetary": float(monetary),
        "AvgBasketValue": float(avg_basket), "UniqueProducts": float(unique_products), "Tenure": float(tenure),
    }
    try:
        r = requests.post(f"{api_base}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            result = r.json()
            st.sidebar.success(f"Persona: **{result['persona']}**")
            st.sidebar.write(f"Cluster: `{result['cluster']}` | Confidence: `{result['confidence']:.2%}` | Algorithm: `{result['algorithm']}`")
        else:
            st.sidebar.error(f"API returned {r.status_code}: {r.text}")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Could not reach API at {api_base}. Is `python scripts/run_api.py` running? ({e})")

st.sidebar.markdown("---")
st.sidebar.header("🎮 Simulate Live Traffic")
st.sidebar.caption("Sends N synthetic customers straight to /predict — no need to leave the dashboard or use the CLI.")

sim_scenario = st.sidebar.selectbox(
    "Scenario", ["normal_traffic", "high_value_surge", "churn_risk_drift"],
    help="churn_risk_drift is designed to trip the PSI Drift tab's alerts.",
)
sim_n = st.sidebar.slider("Customers to simulate", min_value=10, max_value=300, value=50, step=10)

if st.sidebar.button("Send Simulated Traffic"):
    try:
        from simulation.customer_simulator import simulate_batch
        batch = simulate_batch(sim_scenario, sim_n)
        progress = st.sidebar.progress(0, text="Sending requests...")
        ok, fail = 0, 0
        for i, (_, row) in enumerate(batch.iterrows()):
            payload = {c: float(row[c]) for c in RFM_COLS}
            try:
                r = requests.post(f"{api_base}/predict", json=payload, timeout=5)
                ok += 1 if r.status_code == 200 else 0
                fail += 0 if r.status_code == 200 else 1
            except requests.exceptions.RequestException:
                fail += 1
            progress.progress((i + 1) / sim_n, text=f"Sending requests... {i+1}/{sim_n}")
        progress.empty()
        st.sidebar.success(f"Sent {ok} succeeded, {fail} failed to {api_base}/predict")
        st.cache_data.clear()
    except Exception as e:
        st.sidebar.error(f"Simulation failed: {e}")


# ============================================================
# LOAD DATA FOR TABS
# ============================================================

registry = load_registry()
card = load_model_card(registry.get("model_card_path")) if registry else None
challenger_log = load_challenger_log()
profile = load_profile()
impact = load_impact()
reference_rfm = load_reference_rfm()
predictions = load_predictions()

tabs = st.tabs(["🚨 Alerts", "🏆 Champion vs Challenger", "📊 KPIs & Charts", "📉 PSI Drift", "🕒 Recent Predictions"])


# ============================================================
# 1. ALERTS
# ============================================================
with tabs[0]:
    st.subheader("Alerts")
    if not card:
        st.warning("No trained model found yet. Run `python scripts/train_model.py` first.")
    else:
        metrics = card.get("metrics", {})
        noise_ratio = metrics.get("noise_ratio", 0)
        stability = metrics.get("bootstrap_stability_silhouette", None)
        shares = metrics.get("cluster_shares", {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{metrics.get('silhouette', 0):.3f}")
        col2.metric("Davies-Bouldin (lower better)", f"{metrics.get('davies_bouldin', 0):.3f}")
        col3.metric("Bootstrap Stability", f"{stability:.3f}" if stability is not None else "n/a")

        if noise_ratio and noise_ratio > 0.15:
            st.error(f"⚠️ High noise ratio: {noise_ratio:.1%} of customers unassigned (DBSCAN). Consider retraining.")
        if shares and min(shares.values()) < 0.03:
            st.warning(f"⚠️ A segment holds only {min(shares.values()):.1%} of customers — below the 3% usability floor.")
        if not reference_rfm.empty and not predictions.empty:
            for feat in ["Recency", "Frequency", "Monetary"]:
                if feat in predictions.columns:
                    p = psi(reference_rfm[feat].values, predictions[feat].values)
                    if p >= PSI_HIGH:
                        st.error(f"🔴 CRITICAL drift on **{feat}** — PSI={p:.3f} (>= {PSI_HIGH}). Retrain recommended.")
                    elif p >= PSI_MODERATE:
                        st.warning(f"🟡 Moderate drift on **{feat}** — PSI={p:.3f} (>= {PSI_MODERATE}). Monitor.")
        if noise_ratio <= 0.15 and (not shares or min(shares.values()) >= 0.03):
            st.success("✅ No active alerts — segmentation is healthy.")


# ============================================================
# 2. CHAMPION VS CHALLENGER
# ============================================================
with tabs[1]:
    st.subheader("Current Champion")
    if registry and card:
        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm", card.get("algorithm", "n/a"))
        c2.metric("Promoted At", registry.get("promoted_at", "n/a"))
        c3.metric("# Segments", len(card.get("personas", {})))
        st.json(card.get("metrics", {}), expanded=False)
    else:
        st.info("No champion model registered yet.")

    st.subheader("Promotion History")
    if challenger_log:
        hist_df = pd.DataFrame(challenger_log)
        st.dataframe(hist_df, width="stretch")
    else:
        st.info("No challenger comparisons logged yet.")


# ============================================================
# 3. KPIs & CHARTS
# ============================================================
with tabs[2]:
    st.subheader("Segment KPIs")
    if profile.empty:
        st.info("Train a model to populate segment KPIs.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Customers", int(profile["customer_count"].sum()))
        k2.metric("Total Revenue", f"₹{profile['total_revenue'].sum():,.0f}")
        k3.metric("Segments", len(profile))

        fig1 = px.bar(profile, x="persona", y="customer_count", color="persona",
                      title="Customer Count by Persona")
        st.plotly_chart(fig1, width="stretch")

        fig2 = px.bar(profile, x="persona", y="total_revenue", color="persona",
                      title="Revenue by Persona")
        st.plotly_chart(fig2, width="stretch")

        fig3 = px.scatter(profile, x="Recency", y="Monetary", size="customer_count",
                          color="persona", hover_name="persona",
                          title="Segment Centroids — Recency vs Monetary")
        st.plotly_chart(fig3, width="stretch")

        if not impact.empty:
            st.subheader("Business Impact — Campaign ROI by Segment")
            st.dataframe(impact, width="stretch")
            fig4 = px.bar(impact, x="persona", y="estimated_roi_x", color="persona",
                          title="Estimated Campaign ROI (x) by Persona")
            st.plotly_chart(fig4, width="stretch")

        dendrogram_path = os.path.join(MODEL_DIR, "dendrogram.png")
        if os.path.exists(dendrogram_path):
            st.subheader("Hierarchical Clustering — Dendrogram")
            st.caption("Only rendered when the champion algorithm is Hierarchical (Agglomerative).")
            st.image(dendrogram_path, width="stretch")


# ============================================================
# 4. PSI DRIFT
# ============================================================
with tabs[3]:
    st.subheader("Feature Drift (PSI) — Live Predictions vs Training Reference")
    if reference_rfm.empty:
        st.info("No reference RFM distribution found. Train a model first.")
    elif predictions.empty:
        st.info("No live predictions logged yet. Use the sidebar's 'Simulate Live Traffic' button, or call /predict.")
    else:
        rows = []
        for feat in RFM_COLS:
            if feat in predictions.columns:
                p = psi(reference_rfm[feat].values, predictions[feat].values)
                status = "🔴 Critical" if p >= PSI_HIGH else ("🟡 Moderate" if p >= PSI_MODERATE else "🟢 Stable")
                rows.append({"feature": feat, "psi": round(p, 4), "status": status})
        psi_df = pd.DataFrame(rows)
        st.dataframe(psi_df, width="stretch")
        fig = px.bar(psi_df, x="feature", y="psi", color="status", title="PSI by Feature")
        fig.add_hline(y=PSI_MODERATE, line_dash="dot", annotation_text="moderate")
        fig.add_hline(y=PSI_HIGH, line_dash="dash", annotation_text="critical")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "PSI compares the *shape* of live traffic to the training distribution. "
            "0.0-0.1 = stable, 0.1-0.2 = moderate drift (monitor), 0.2+ = critical (retrain)."
        )


# ============================================================
# 5. RECENT PREDICTIONS
# ============================================================
with tabs[4]:
    st.subheader("Recent Predictions")
    if predictions.empty:
        st.info("No predictions logged yet. Use the sidebar to try one, or simulate a batch.")
    else:
        st.dataframe(predictions.tail(50).sort_index(ascending=False), width="stretch")
        if "persona" in predictions.columns:
            fig = px.pie(predictions, names="persona", title="Live Prediction Mix by Persona")
            st.plotly_chart(fig, width="stretch")