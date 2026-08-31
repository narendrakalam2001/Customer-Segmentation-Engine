# ============================================================
# SIMULATION — Customer Segmentation Engine
# ============================================================
# Generates synthetic incoming customer RFM records for 3 scenarios,
# used to demo /predict traffic and drift monitoring without needing
# live production data.
#
# IMPORTANT (fix): the first version of this module drew from fixed
# gamma/poisson parameters that were NOT calibrated to the real
# trained dataset (e.g. it could generate Tenure > 1000 days on a
# dataset that only spans ~1 year of transactions). That mismatch
# made even "normal_traffic" look like severe drift versus the
# training reference distribution, producing nonsensical PSI values
# (7-9+) instead of the expected 0-1 range.
#
# Fix: whenever a trained model's reference RFM distribution exists
# (segmentation_models/reference_rfm_distribution.csv, written by
# training_pipeline.py), the simulator BOOTSTRAPS real customer rows
# from it and applies small jitter — so "normal_traffic" stays close
# to the real distribution (near-zero PSI) and the drift scenarios
# (high_value_surge / churn_risk_drift) apply realistic, bounded
# multipliers on top of real customers instead of synthesizing values
# out of thin air. Falls back to the old fixed-parameter synthetic
# generator only when no trained model/reference exists yet (e.g.
# CI/unit tests, or before the first training run).
# ============================================================

import os
import time
import logging
import numpy as np
import pandas as pd

from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)

REFERENCE_PATH = os.path.join(MODEL_DIR, "reference_rfm_distribution.csv")
RFM_COLS = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "UniqueProducts", "Tenure"]


def _load_reference():
    if os.path.exists(REFERENCE_PATH):
        try:
            return pd.read_csv(REFERENCE_PATH)
        except Exception as e:
            logger.warning("Could not read reference distribution (%s) — using fallback synthetic generator: %s",
                           REFERENCE_PATH, e)
    return None


def _bootstrap_from_reference(ref_df: pd.DataFrame, n: int, jitter_pct: float = 0.05) -> pd.DataFrame:
    """
    Resamples n real customer rows (with replacement) from the training
    reference distribution and adds small Gaussian jitter (5% of each
    column's std by default) so simulated traffic isn't literally
    duplicating real customer records, while staying statistically
    close to the true distribution.
    """
    idx = RNG.choice(len(ref_df), size=n, replace=True)
    sample = ref_df.iloc[idx][RFM_COLS].reset_index(drop=True).copy()
    for col in RFM_COLS:
        std = ref_df[col].std()
        jitter = RNG.normal(0, max(std * jitter_pct, 1e-6), n)
        sample[col] = (sample[col] + jitter).clip(lower=0)
    return sample


# ── Fallback synthetic generator (used only when no trained model exists yet) ──

def _fallback_normal_traffic(n: int) -> pd.DataFrame:
    logger.warning(
        "No trained model reference distribution found at %s — using rough "
        "fallback synthetic ranges. Run scripts/train_model.py first for "
        "realistic simulation (accurate PSI drift checks require it).",
        REFERENCE_PATH,
    )
    return pd.DataFrame({
        "Recency":        RNG.gamma(2.0, 40, n).round(0).clip(0, 365),
        "Frequency":      RNG.poisson(4, n) + 1,
        "Monetary":       RNG.gamma(2.5, 250, n).round(2),
        "AvgBasketValue": RNG.gamma(2.0, 60, n).round(2),
        "UniqueProducts": RNG.poisson(15, n) + 1,
        "Tenure":         RNG.gamma(3.0, 90, n).round(0).clip(0, 365),
    })


def _base_traffic(n: int) -> pd.DataFrame:
    ref = _load_reference()
    if ref is not None and len(ref) >= 10:
        return _bootstrap_from_reference(ref, n)
    return _fallback_normal_traffic(n)


def _sample_normal_traffic(n: int) -> pd.DataFrame:
    """Business-as-usual customer mix — bootstrapped from the real training distribution."""
    return _base_traffic(n)


def _sample_high_value_surge(n: int) -> pd.DataFrame:
    """Scenario: a marketing push drives a spike of high-frequency, high-spend customers."""
    df = _base_traffic(n)
    df["Frequency"]  = (df["Frequency"] * RNG.uniform(1.8, 2.8, n)).round(0)
    df["Monetary"]   = (df["Monetary"] * RNG.uniform(1.8, 2.8, n)).round(2)
    df["Recency"]    = (df["Recency"] * RNG.uniform(0.3, 0.6, n)).round(0)
    return df


def _sample_churn_risk_drift(n: int) -> pd.DataFrame:
    """Scenario: rising Recency + falling Frequency — a churn-risk drift wave (should trip PSI moderately, not extremely)."""
    df = _base_traffic(n)
    df["Recency"]    = (df["Recency"] * RNG.uniform(1.5, 2.2, n)).round(0).clip(upper=400)
    df["Frequency"]  = np.maximum(1, (df["Frequency"] * RNG.uniform(0.3, 0.6, n))).round(0)
    df["Monetary"]   = (df["Monetary"] * RNG.uniform(0.4, 0.7, n)).round(2)
    return df


SCENARIOS = {
    "normal_traffic":     _sample_normal_traffic,
    "high_value_surge":   _sample_high_value_surge,
    "churn_risk_drift":   _sample_churn_risk_drift,
}


def simulate_batch(scenario: str = "normal_traffic", n: int = 50) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}")
    df = SCENARIOS[scenario](n)
    df["scenario"] = scenario
    df["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Simulated %d customers for scenario '%s'", n, scenario)
    return df