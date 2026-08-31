# ============================================================
# METRICS — Customer Segmentation Engine
# ============================================================
# Unsupervised clustering has no ground-truth labels, so evaluation
# relies on internal validity indices + bootstrap stability, not
# F1/ROC-AUC. These are the industry-standard metrics for KMeans/
# DBSCAN/GMM model selection.
# ============================================================

import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.utils import resample

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def cluster_validity_metrics(X: np.ndarray, labels: np.ndarray, context: str = "") -> dict:
    """
    Computes the three standard internal cluster-validity indices.
    DBSCAN noise points (label == -1) are excluded from silhouette/
    CH/DB computation since those indices are undefined for noise.

    context: optional label (e.g. "DBSCAN eps=0.3, min_samples=5") used
             only to make the "not enough clusters" warning identifiable
             during a large grid search — purely cosmetic, doesn't affect
             the returned metrics.

    Returns dict with:
        silhouette        (higher is better, range [-1, 1])
        davies_bouldin     (lower is better, >= 0)
        calinski_harabasz  (higher is better)
        n_clusters
        noise_ratio        (fraction of points labelled -1, DBSCAN only)
    """
    mask = labels != -1
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = float((~mask).sum()) / len(labels) if len(labels) else 0.0
    min_cluster_share = min(cluster_size_shares(labels[mask]).values()) if mask.any() and n_clusters else 0.0

    if n_clusters < 2 or mask.sum() < n_clusters + 1:
        tag = f" [{context}]" if context else ""
        logger.debug(
            "Skipped invalid config%s -> n_clusters=%d, usable_points=%d, noise_ratio=%.2f "
            "(penalized with silhouette=-1, will not be selected)",
            tag, n_clusters, int(mask.sum()), noise_ratio,
        )
        return {
            "silhouette": -1.0, "davies_bouldin": np.inf,
            "calinski_harabasz": 0.0, "n_clusters": n_clusters,
            "noise_ratio": round(noise_ratio, 4),
            "min_cluster_share": round(float(min_cluster_share), 4),
        }

    Xc, lc = X[mask], labels[mask]

    sil = silhouette_score(Xc, lc)
    db  = davies_bouldin_score(Xc, lc)
    ch  = calinski_harabasz_score(Xc, lc)

    return {
        "silhouette":        round(float(sil), 4),
        "davies_bouldin":    round(float(db), 4),
        "calinski_harabasz": round(float(ch), 2),
        "n_clusters":        n_clusters,
        "noise_ratio":       round(noise_ratio, 4),
        "min_cluster_share": round(float(min_cluster_share), 4),
    }


def per_sample_silhouette(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-customer silhouette width — used to flag weakly-assigned customers."""
    mask = labels != -1
    scores = np.full(len(labels), np.nan)
    if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
        scores[mask] = silhouette_samples(X[mask], labels[mask])
    return scores


def cluster_size_shares(labels: np.ndarray) -> dict:
    """Returns {cluster_id: share_of_total} excluding noise (-1) label."""
    s = pd.Series(labels)
    total = len(s)
    shares = (s.value_counts() / total).to_dict()
    return {int(k): round(float(v), 4) for k, v in shares.items()}


def bootstrap_stability(fit_predict_fn, X: np.ndarray, n_iter: int = 10, sample_frac: float = 0.8) -> float:
    """
    Bootstrap cluster stability check: refits the model on random
    subsamples and measures mean silhouette across resamples.
    A low variance / consistently positive silhouette across
    resamples indicates the segmentation is not an artifact of
    a single lucky centroid initialization.

    fit_predict_fn: callable(X_subsample) -> labels
    Returns: mean silhouette across bootstrap resamples.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    scores = []
    n = X.shape[0]
    for i in range(n_iter):
        idx = resample(np.arange(n), replace=False,
                        n_samples=int(n * sample_frac),
                        random_state=rng.randint(0, 1_000_000))
        X_sub = X[idx]
        labels = fit_predict_fn(X_sub)
        m = cluster_validity_metrics(X_sub, labels)
        scores.append(m["silhouette"])
    mean_sil = float(np.mean(scores))
    logger.info("Bootstrap stability (%d iters): mean silhouette=%.4f (std=%.4f)",
                n_iter, mean_sil, float(np.std(scores)))
    return mean_sil


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index — edge-based implementation.
    Bin edges are computed from the EXPECTED (reference/training)
    distribution only, then reused for the ACTUAL (current/production)
    distribution — this is the correct PSI convention (same standard
    used in the Credit Risk project's PSI drift monitor).
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual,   bins=edges)[0].astype(float)

    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, None)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))