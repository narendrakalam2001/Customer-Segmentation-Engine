# ============================================================
# TESTS — Customer Segmentation Engine
# ============================================================
# 30 tests covering data loading, RFM engineering, leakage checks,
# preprocessing, clustering/model_tuning, metrics, evaluation,
# champion-challenger logic, and the segmentation inference engine.
# Run: pytest tests/ -v
# ============================================================

import os
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.data_loader import load_raw_data, clean_transactions, load_and_clean
from src.preprocessing import (
    Clipper, build_rfm_table, build_clustering_preprocessor,
    fit_transform_rfm, RFM_FEATURES,
)
from src.leakage_check import (
    check_snapshot_leakage, check_duplicate_customers,
    check_negative_or_null_rfm, run_all_checks,
)
from src.model_tuning import (
    tune_kmeans, tune_dbscan, fit_best_kmeans,
    tune_hierarchical, fit_best_hierarchical, compute_linkage_matrix, labels_from_linkage,
)
from src.metrics import (
    cluster_validity_metrics, cluster_size_shares, psi, bootstrap_stability,
)
from src.evaluation import score_rfm, profile_clusters, assign_personas, business_impact_summary
from src.model_loader import run_challenger_comparison, _load_champion_metrics
from src.segmentation_engine import assign_segment

FIXTURE_PATH = "tests/fixtures/sample_retail.csv"
TEST_MODEL_DIR = "tests/_tmp_models"


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(FIXTURE_PATH)


@pytest.fixture(scope="module")
def clean_df():
    return load_and_clean(FIXTURE_PATH)


@pytest.fixture(scope="module")
def rfm_data(clean_df):
    rfm_df, snapshot = build_rfm_table(clean_df)
    return rfm_df, snapshot


@pytest.fixture(scope="module")
def scaled_data(rfm_data):
    rfm_df, _ = rfm_data
    X_scaled, pipeline = fit_transform_rfm(rfm_df)
    return X_scaled, pipeline, rfm_df


# ============================================================
# 1. DATA LOADER
# ============================================================

def test_load_raw_data_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) > 0


def test_load_raw_data_has_required_columns(raw_df):
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    for col in required:
        assert col in raw_df.columns


def test_load_raw_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_data("does/not/exist.csv")


def test_clean_transactions_drops_null_customers(raw_df):
    cleaned = clean_transactions(raw_df)
    assert cleaned["CustomerID"].isnull().sum() == 0


def test_clean_transactions_drops_cancelled_invoices(raw_df):
    cleaned = clean_transactions(raw_df)
    assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()


def test_clean_transactions_drops_nonpositive_qty_price(raw_df):
    cleaned = clean_transactions(raw_df)
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["UnitPrice"] > 0).all()


def test_clean_transactions_row_count_reduced(raw_df):
    cleaned = clean_transactions(raw_df)
    assert len(cleaned) < len(raw_df)


# ============================================================
# 2. RFM ENGINEERING
# ============================================================

def test_rfm_table_one_row_per_customer(rfm_data, clean_df):
    rfm_df, _ = rfm_data
    assert rfm_df["CustomerID"].is_unique


def test_rfm_table_has_all_features(rfm_data):
    rfm_df, _ = rfm_data
    for col in RFM_FEATURES:
        assert col in rfm_df.columns


def test_rfm_recency_non_negative(rfm_data):
    rfm_df, _ = rfm_data
    assert (rfm_df["Recency"] >= 0).all()


def test_rfm_monetary_positive(rfm_data):
    rfm_df, _ = rfm_data
    assert (rfm_df["Monetary"] > 0).all()


def test_rfm_snapshot_date_after_max_invoice(rfm_data, clean_df):
    _, snapshot = rfm_data
    assert snapshot > clean_df["InvoiceDate"].max()


# ============================================================
# 3. LEAKAGE CHECKS
# ============================================================

def test_snapshot_leakage_passes_on_clean_data(clean_df, rfm_data):
    _, snapshot = rfm_data
    result = check_snapshot_leakage(clean_df, snapshot)
    assert result["passed"] is True


def test_snapshot_leakage_fails_with_future_data(clean_df):
    bad_snapshot = clean_df["InvoiceDate"].min()
    result = check_snapshot_leakage(clean_df, bad_snapshot)
    assert result["passed"] is False


def test_duplicate_customers_check_passes(rfm_data):
    rfm_df, _ = rfm_data
    result = check_duplicate_customers(rfm_df)
    assert bool(result["passed"]) is True


def test_duplicate_customers_check_fails_on_dupes(rfm_data):
    rfm_df, _ = rfm_data
    dup_df = pd.concat([rfm_df, rfm_df.iloc[[0]]], ignore_index=True)
    result = check_duplicate_customers(dup_df)
    assert bool(result["passed"]) is False


def test_negative_or_null_rfm_passes(rfm_data):
    rfm_df, _ = rfm_data
    result = check_negative_or_null_rfm(rfm_df, RFM_FEATURES)
    assert bool(result["passed"]) is True


def test_run_all_checks_raises_on_bad_data(clean_df, rfm_data):
    rfm_df, snapshot = rfm_data
    bad_rfm = rfm_df.copy()
    bad_rfm.loc[0, "Recency"] = -5
    with pytest.raises(AssertionError):
        run_all_checks(clean_df, bad_rfm, snapshot, RFM_FEATURES)


# ============================================================
# 4. PREPROCESSING
# ============================================================

def test_clipper_fits_and_transforms():
    X = np.array([[1], [2], [3], [4], [1000]])
    clipper = Clipper(fold=1.5)
    clipper.fit(X)
    Xt = clipper.transform(X)
    assert Xt.max() < 1000


def test_clipper_clone_independence():
    base = build_clustering_preprocessor()
    p1 = clone(base)
    p2 = clone(base)
    assert p1 is not p2


def test_fit_transform_rfm_shape(rfm_data):
    rfm_df, _ = rfm_data
    X_scaled, pipeline = fit_transform_rfm(rfm_df)
    assert X_scaled.shape[0] == len(rfm_df)
    assert X_scaled.shape[1] == len(RFM_FEATURES)


def test_fit_transform_rfm_scaled_mean_near_zero(scaled_data):
    X_scaled, _, _ = scaled_data
    assert abs(X_scaled.mean()) < 1.0


# ============================================================
# 5. MODEL TUNING & METRICS
# ============================================================

def test_tune_kmeans_returns_sorted_results(scaled_data):
    X_scaled, _, _ = scaled_data
    results = tune_kmeans(X_scaled, k_range=[2, 3, 4])
    assert len(results) == 3
    assert results["silhouette"].is_monotonic_decreasing


def test_tune_dbscan_returns_results(scaled_data):
    X_scaled, _, _ = scaled_data
    results = tune_dbscan(X_scaled, eps_grid=[0.5, 1.0], min_samples_grid=[5])
    assert len(results) == 2


def test_tune_hierarchical_returns_sorted_results(scaled_data):
    X_scaled, _, _ = scaled_data
    results = tune_hierarchical(X_scaled, k_range=[2, 3, 4], linkage_methods=["ward"])
    assert len(results) == 3
    assert results["silhouette"].is_monotonic_decreasing


def test_fit_best_hierarchical_label_count(scaled_data):
    X_scaled, _, _ = scaled_data
    model = fit_best_hierarchical(X_scaled, k=3, method="ward")
    assert len(set(model.labels_)) == 3


def test_compute_linkage_matrix_shape(scaled_data):
    X_scaled, _, _ = scaled_data
    Z = compute_linkage_matrix(X_scaled, method="ward")
    # SciPy linkage matrix always has shape (n_samples - 1, 4)
    assert Z.shape == (X_scaled.shape[0] - 1, 4)


def test_labels_from_linkage_matches_k(scaled_data):
    X_scaled, _, _ = scaled_data
    Z = compute_linkage_matrix(X_scaled, method="ward")
    labels = labels_from_linkage(Z, k=3)
    assert len(set(labels)) == 3
    assert len(labels) == X_scaled.shape[0]


def test_cluster_validity_metrics_valid_range(scaled_data):
    X_scaled, _, _ = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    metrics = cluster_validity_metrics(X_scaled, model.labels_)
    assert -1.0 <= metrics["silhouette"] <= 1.0
    assert metrics["davies_bouldin"] >= 0


def test_cluster_size_shares_sum_to_one(scaled_data):
    X_scaled, _, _ = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    shares = cluster_size_shares(model.labels_)
    assert abs(sum(shares.values()) - 1.0) < 1e-3


def test_psi_zero_for_identical_distributions():
    x = np.random.normal(0, 1, 500)
    assert psi(x, x) < 1e-6


def test_psi_positive_for_shifted_distribution():
    x = np.random.normal(0, 1, 500)
    y = np.random.normal(3, 1, 500)
    assert psi(x, y) > 0.2


def test_bootstrap_stability_returns_float(scaled_data):
    X_scaled, _, _ = scaled_data
    stability = bootstrap_stability(lambda Xs: fit_best_kmeans(Xs, k=3).labels_, X_scaled, n_iter=3)
    assert isinstance(stability, float)


# ============================================================
# 6. EVALUATION
# ============================================================

def test_score_rfm_scores_in_range(rfm_data):
    rfm_df, _ = rfm_data
    scored = score_rfm(rfm_df)
    for col in ["R_score", "F_score", "M_score"]:
        assert scored[col].between(1, 5).all()


def test_profile_clusters_shape(scaled_data):
    X_scaled, _, rfm_df = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    profile = profile_clusters(rfm_df, model.labels_, RFM_FEATURES)
    assert len(profile) == 3
    assert "customer_count" in profile.columns


def test_assign_personas_adds_persona_column(scaled_data):
    X_scaled, _, rfm_df = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    profile = profile_clusters(rfm_df, model.labels_, RFM_FEATURES)
    personas = assign_personas(profile)
    assert "persona" in personas.columns
    assert personas["persona"].notnull().all()


def test_business_impact_summary_has_roi_column(scaled_data):
    X_scaled, _, rfm_df = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    profile = assign_personas(profile_clusters(rfm_df, model.labels_, RFM_FEATURES))
    impact = business_impact_summary(profile)
    assert "estimated_roi_x" in impact.columns


# ============================================================
# 7. CHAMPION-CHALLENGER SYSTEM
# ============================================================

@pytest.fixture
def temp_model_dir(monkeypatch):
    if os.path.exists(TEST_MODEL_DIR):
        shutil.rmtree(TEST_MODEL_DIR)
    os.makedirs(TEST_MODEL_DIR, exist_ok=True)
    monkeypatch.setattr("src.model_loader.MODEL_DIR", TEST_MODEL_DIR)
    monkeypatch.setattr("src.model_loader.CHALLENGER_LOG", os.path.join(TEST_MODEL_DIR, "challenger_log.json"))
    yield TEST_MODEL_DIR
    shutil.rmtree(TEST_MODEL_DIR, ignore_errors=True)


def test_first_model_auto_promoted(temp_model_dir):
    dummy_path = os.path.join(temp_model_dir, "dummy_model.joblib")
    open(dummy_path, "w").close()
    result = run_challenger_comparison(
        challenger_name="dummy_model.joblib", challenger_algorithm="KMeans",
        challenger_silhouette=0.5, challenger_db_index=0.8,
        challenger_min_cluster_share=0.10, challenger_model_path=dummy_path,
    )
    assert result["decision"] == "PROMOTED"


def test_challenger_rejected_on_worse_silhouette(temp_model_dir):
    # promote a strong first champion
    p1 = os.path.join(temp_model_dir, "champion.joblib")
    open(p1, "w").close()
    card1 = os.path.join(temp_model_dir, "card1.json")
    with open(card1, "w") as f:
        json.dump({"metrics": {"silhouette": 0.6, "davies_bouldin": 0.5, "cluster_shares": {"0": 0.5, "1": 0.5}}}, f)
    run_challenger_comparison("champion.joblib", "KMeans", 0.6, 0.5, 0.5, p1, card1)

    p2 = os.path.join(temp_model_dir, "challenger.joblib")
    open(p2, "w").close()
    result = run_challenger_comparison("challenger.joblib", "KMeans", 0.3, 0.9, 0.4, p2)
    assert result["decision"] == "REJECTED"
    assert result["gates"]["silhouette_improvement_passed"] is False


def test_challenger_log_persists(temp_model_dir):
    p1 = os.path.join(temp_model_dir, "m.joblib")
    open(p1, "w").close()
    run_challenger_comparison("m.joblib", "KMeans", 0.4, 0.7, 0.2, p1)
    log_path = os.path.join(temp_model_dir, "challenger_log.json")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        history = json.load(f)
    assert len(history) == 1


# ============================================================
# 8. SEGMENTATION ENGINE (inference)
# ============================================================

def test_assign_segment_kmeans(scaled_data):
    X_scaled, pipeline, rfm_df = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    bundle = {
        "model": model, "preprocessor": pipeline, "feature_cols": RFM_FEATURES,
        "persona_map": {"0": "Champions", "1": "At Risk", "2": "New Customers"},
        "algorithm": "KMeans",
    }
    sample = rfm_df.iloc[0][RFM_FEATURES].to_dict()
    result = assign_segment(bundle, sample)
    assert result["cluster"] in [0, 1, 2]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["persona"] in ["Champions", "At Risk", "New Customers"]


def test_assign_segment_hierarchical(scaled_data):
    X_scaled, pipeline, rfm_df = scaled_data
    model = fit_best_hierarchical(X_scaled, k=3, method="ward")
    centroids = np.array([X_scaled[model.labels_ == c].mean(axis=0) for c in sorted(set(model.labels_))])
    bundle = {
        "model": model, "preprocessor": pipeline, "feature_cols": RFM_FEATURES,
        "persona_map": {"0": "Champions", "1": "At Risk", "2": "New Customers"},
        "algorithm": "Hierarchical", "centroids": centroids.tolist(),
    }
    sample = rfm_df.iloc[0][RFM_FEATURES].to_dict()
    result = assign_segment(bundle, sample)
    assert result["cluster"] in [0, 1, 2]
    assert 0.0 <= result["confidence"] <= 1.0


def test_assign_segment_confidence_is_float(scaled_data):
    X_scaled, pipeline, rfm_df = scaled_data
    model = fit_best_kmeans(X_scaled, k=3)
    bundle = {
        "model": model, "preprocessor": pipeline, "feature_cols": RFM_FEATURES,
        "persona_map": {}, "algorithm": "KMeans",
    }
    sample = rfm_df.iloc[1][RFM_FEATURES].to_dict()
    result = assign_segment(bundle, sample)
    assert isinstance(result["confidence"], float)
