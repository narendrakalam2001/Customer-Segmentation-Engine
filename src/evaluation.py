# ============================================================
# EVALUATION — Customer Segmentation Engine
# ============================================================
# Cluster profiling, RFM-quantile persona naming, and cost-sensitive
# business-impact evaluation (the "so what" layer every stakeholder
# demo / consulting deck needs — mirrors the cost-sensitive evaluation
# standard used in the Credit Risk & Fraud projects).
# ============================================================

import logging
import numpy as np
import pandas as pd

from src.config import (
    RFM_QUANTILES, AVG_CAMPAIGN_COST_PER_CUSTOMER,
    CHAMPIONS_UPLIFT_RATE, AT_RISK_RETENTION_RATE,
)

logger = logging.getLogger(__name__)


# ============================================================
# RFM SCORING (1-5 quantile scores, standard RFM convention)
# ============================================================

def score_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns 1-5 quantile scores per RFM dimension.
    Recency is inverted (lower recency = more recent = higher score).
    """
    out = rfm_df.copy()

    out["R_score"] = pd.qcut(out["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F_score"] = pd.qcut(out["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_score"] = pd.qcut(out["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_score"] = out["R_score"] + out["F_score"] + out["M_score"]

    return out


# ============================================================
# CLUSTER PROFILING
# ============================================================

def profile_clusters(rfm_df: pd.DataFrame, labels: np.ndarray, feature_cols) -> pd.DataFrame:
    """
    Returns one row per cluster with mean feature values, size,
    and share of total customers — the raw material for persona
    naming and the monitoring dashboard's KPI section.
    """
    df = rfm_df.copy()
    df["cluster"] = labels

    profile = df.groupby("cluster")[feature_cols].mean().round(2)
    profile["customer_count"] = df.groupby("cluster").size()
    profile["share_of_base"]  = (profile["customer_count"] / len(df)).round(4)
    profile["total_revenue"]  = df.groupby("cluster")["Monetary"].sum().round(2)

    return profile.reset_index().sort_values("total_revenue", ascending=False)


PERSONA_RULES = [
    # (name, condition_fn(row)) evaluated in priority order on cluster-mean row
    ("Champions",        lambda r: r["Recency"] <= r["_r_med"] and r["Frequency"] >= r["_f_med"] and r["Monetary"] >= r["_m_med"]),
    ("Loyal Customers",  lambda r: r["Frequency"] >= r["_f_med"] and r["Recency"] <= r["_r_med"] * 1.5),
    ("Big Spenders",     lambda r: r["Monetary"] >= r["_m_med"] * 1.5),
    ("At Risk",          lambda r: r["Recency"] > r["_r_med"] * 1.5 and r["Frequency"] >= r["_f_med"]),
    ("New Customers",    lambda r: r["Tenure"] <= r["_tenure_med"] and r["Frequency"] <= 2),
    ("Hibernating/Lost", lambda r: r["Recency"] > r["_r_med"] * 2),
]


def assign_personas(cluster_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Maps each cluster's mean RFM profile to a business persona label
    using rule-based logic on the cluster centroids (Champions, Loyal
    Customers, At Risk, New Customers, Big Spenders, Hibernating/Lost)
    — the standard segmentation vocabulary used by Myntra/HDFC-style
    marketing/CRM teams.
    """
    out = cluster_profile.copy()
    out["_r_med"] = out["Recency"].median()
    out["_f_med"] = out["Frequency"].median()
    out["_m_med"] = out["Monetary"].median()
    out["_tenure_med"] = out["Tenure"].median() if "Tenure" in out.columns else 0

    personas = []
    for _, row in out.iterrows():
        label = "Regular Customers"
        for name, cond in PERSONA_RULES:
            try:
                if cond(row):
                    label = name
                    break
            except Exception:
                continue
        personas.append(label)

    out["persona"] = personas
    return out.drop(columns=["_r_med", "_f_med", "_m_med", "_tenure_med"])


# ============================================================
# BUSINESS IMPACT (cost-sensitive evaluation)
# ============================================================

def business_impact_summary(persona_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Translates each segment into an estimated campaign ROI:
      Champions   -> upsell/loyalty campaign, expected incremental revenue
      At Risk     -> win-back campaign, expected retained revenue
      Others      -> standard broadcast, cost only

    This is the "cost-sensitive evaluation" layer — same philosophy as
    the $-impact tables in the Fraud/Credit Risk projects, adapted to
    marketing ROI instead of loss-avoidance.
    """
    rows = []
    for _, r in persona_profile.iterrows():
        n = int(r["customer_count"])
        cost = round(n * AVG_CAMPAIGN_COST_PER_CUSTOMER, 2)

        if r["persona"] == "Champions":
            expected_gain = round(r["total_revenue"] * CHAMPIONS_UPLIFT_RATE, 2)
            action = "Loyalty / early-access campaign"
        elif r["persona"] == "At Risk":
            expected_gain = round(r["total_revenue"] * AT_RISK_RETENTION_RATE, 2)
            action = "Win-back discount campaign"
        elif r["persona"] == "Hibernating/Lost":
            expected_gain = round(r["total_revenue"] * 0.05, 2)
            action = "Low-cost reactivation email"
        else:
            expected_gain = 0.0
            action = "Standard broadcast / no targeted spend"

        rows.append({
            "persona": r["persona"], "customer_count": n,
            "campaign_cost": cost, "expected_incremental_revenue": expected_gain,
            "estimated_roi_x": round(expected_gain / cost, 2) if cost > 0 else 0.0,
            "recommended_action": action,
        })

    return pd.DataFrame(rows).sort_values("estimated_roi_x", ascending=False)
