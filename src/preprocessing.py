# ============================================================
# PREPROCESSING — Customer Segmentation Engine
# ============================================================
# RFM (Recency, Frequency, Monetary) feature engineering +
# outlier-robust scaling pipeline for distance-based clustering
# (KMeans / DBSCAN / GMM all require scaled, non-skewed input).
# ============================================================

import logging
import numpy as np
import pandas as pd

from sklearn.base          import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler, PowerTransformer

from src.config import CLIP_FOLD, SNAPSHOT_BUFFER_DAYS

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER — IQR-based outlier clipping transformer
# (identical standard used across all portfolio projects)
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips values to [Q1 - fold*IQR, Q3 + fold*IQR].
    Fitted on train set only, applied to train + inference data.
    Prevents extreme whale-customers from destabilizing centroids.
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q1, q3 = np.quantile(X, 0.25, axis=0), np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr

        eps = 1e-9
        self.upper_ = np.where(self.upper_ == self.lower_, self.upper_ + eps, self.upper_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# RFM ENGINEERING
# ============================================================

def build_rfm_table(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Aggregates a cleaned transaction log into one row per CustomerID:
        Recency   = days since last purchase (relative to snapshot_date)
        Frequency = number of distinct invoices
        Monetary  = total spend (sum of Quantity * UnitPrice)

    snapshot_date defaults to (max InvoiceDate + SNAPSHOT_BUFFER_DAYS),
    the standard convention for RFM on a static historical extract.
    """
    df = df.copy()
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=SNAPSHOT_BUFFER_DAYS)

    rfm = (
        df.groupby("CustomerID")
          .agg(
              Recency   = ("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
              Frequency = ("InvoiceNo",   "nunique"),
              Monetary  = ("LineTotal",   "sum"),
          )
          .reset_index()
    )

    # Extra behavioral features (industry-standard extensions to base RFM)
    extra = (
        df.groupby("CustomerID")
          .agg(
              AvgBasketValue = ("LineTotal", "mean"),
              UniqueProducts = ("StockCode", "nunique"),
              Tenure         = ("InvoiceDate", lambda s: (s.max() - s.min()).days + 1),
          )
          .reset_index()
    )

    rfm = rfm.merge(extra, on="CustomerID", how="left")
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)

    logger.info("RFM table built: %d customers | snapshot_date=%s", len(rfm), snapshot_date.date())
    return rfm, snapshot_date


RFM_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "UniqueProducts", "Tenure"]


# ============================================================
# CLUSTERING PREPROCESSOR BUILDER
# ============================================================

def build_clustering_preprocessor(feature_cols=None) -> Pipeline:
    """
    Builds the scaling pipeline applied before KMeans/DBSCAN/GMM:
        Clip outliers (IQR)  ->  Yeo-Johnson power transform (fix skew)
        ->  StandardScaler (zero mean, unit variance — required for
            any distance-based algorithm, incl. Euclidean k-means).

    Returns an UNFITTED sklearn Pipeline. Use clone() before each
    independent fit to avoid shared fitted state across champion/
    challenger runs (same standard as other portfolio projects).
    """
    pipeline = Pipeline([
        ("clip",   Clipper(fold=CLIP_FOLD)),
        ("power",  PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale",  StandardScaler()),
    ])
    return pipeline


def fit_transform_rfm(rfm_df: pd.DataFrame, feature_cols=None):
    """
    Fits a fresh (cloned) preprocessing pipeline on RFM features
    and returns (X_scaled: np.ndarray, fitted_pipeline).
    """
    feature_cols = feature_cols or RFM_FEATURES
    pipeline = clone(build_clustering_preprocessor())
    X = rfm_df[feature_cols].values
    X_scaled = pipeline.fit_transform(X)
    return X_scaled, pipeline
