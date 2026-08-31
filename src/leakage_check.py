# ============================================================
# LEAKAGE CHECK — Customer Segmentation Engine
# ============================================================
# For unsupervised RFM segmentation the leakage risks are different
# from a supervised model, but just as real:
#   1. Snapshot leakage — RFM Recency must be computed relative to a
#      single, fixed snapshot_date. Recomputing "today" per-row lets
#      future data leak into a "training" cut.
#   2. Temporal train/holdout leakage — when validating stability,
#      the holdout period's transactions must not be included in the
#      snapshot used to build the training RFM table.
#   3. Duplicate customer leakage — a CustomerID must not appear more
#      than once in the RFM table (would silently double-weight that
#      customer in clustering).
#   4. Scaling leakage — the Clipper / PowerTransformer / StandardScaler
#      must be fit on the training split only, never on the full
#      dataset before splitting (checked at the pipeline level).
# ============================================================

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def check_snapshot_leakage(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> dict:
    """Fails if any transaction occurs AFTER the snapshot_date used for RFM."""
    future_rows = df[df["InvoiceDate"] > snapshot_date]
    passed = len(future_rows) == 0
    result = {
        "check": "snapshot_leakage",
        "passed": passed,
        "future_rows_found": int(len(future_rows)),
    }
    if not passed:
        logger.error("LEAKAGE: %d transactions occur after snapshot_date=%s",
                      len(future_rows), snapshot_date)
    return result


def check_duplicate_customers(rfm_df: pd.DataFrame) -> dict:
    dupes = rfm_df["CustomerID"].duplicated().sum()
    passed = dupes == 0
    result = {"check": "duplicate_customers", "passed": passed, "duplicates_found": int(dupes)}
    if not passed:
        logger.error("LEAKAGE: %d duplicate CustomerID rows in RFM table", dupes)
    return result


def check_train_holdout_overlap(train_ids: pd.Series, holdout_ids: pd.Series) -> dict:
    """Ensures the bootstrap-stability holdout customers don't overlap train fit customers."""
    overlap = set(train_ids) & set(holdout_ids)
    passed = len(overlap) == 0
    result = {"check": "train_holdout_overlap", "passed": passed, "overlap_count": len(overlap)}
    if not passed:
        logger.error("LEAKAGE: %d customers present in both train and holdout", len(overlap))
    return result


def check_negative_or_null_rfm(rfm_df: pd.DataFrame, feature_cols) -> dict:
    """RFM features must be non-null and Recency/Frequency/Monetary >= 0."""
    nulls = rfm_df[feature_cols].isnull().sum().sum()
    negatives = (rfm_df[["Recency", "Frequency", "Monetary"]] < 0).sum().sum()
    passed = (nulls == 0) and (negatives == 0)
    result = {
        "check": "negative_or_null_rfm",
        "passed": passed,
        "nulls_found": int(nulls),
        "negatives_found": int(negatives),
    }
    if not passed:
        logger.error("LEAKAGE/QUALITY: nulls=%d negatives=%d in RFM table", nulls, negatives)
    return result


def run_all_checks(df: pd.DataFrame, rfm_df: pd.DataFrame, snapshot_date, feature_cols) -> list:
    """Runs the full leakage-check suite; raises AssertionError if any check fails."""
    checks = [
        check_snapshot_leakage(df, snapshot_date),
        check_duplicate_customers(rfm_df),
        check_negative_or_null_rfm(rfm_df, feature_cols),
    ]
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise AssertionError(f"Leakage checks FAILED: {failed}")
    logger.info("All %d leakage checks PASSED", len(checks))
    return checks
