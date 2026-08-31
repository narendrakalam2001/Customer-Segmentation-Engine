# ============================================================
# MODEL TUNING — Customer Segmentation Engine
# ============================================================
# Grid search across KMeans (k), DBSCAN (eps, min_samples),
# Agglomerative/Hierarchical Clustering (k, linkage) with dendrogram
# support, and GaussianMixture (k) — all evaluated using internal
# validity indices, since there is no ground truth to cross-validate
# against.
# ============================================================

import logging
import numpy as np
import pandas as pd

from sklearn.cluster  import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture   import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from src.config import (
    RANDOM_STATE, KMEANS_K_RANGE, GMM_K_RANGE,
    DBSCAN_EPS_GRID, DBSCAN_MIN_SAMPLES,
    HIERARCHICAL_K_RANGE, HIERARCHICAL_LINKAGE_METHODS,
)
from src.metrics import cluster_validity_metrics

logger = logging.getLogger(__name__)


def tune_kmeans(X: np.ndarray, k_range=None) -> pd.DataFrame:
    """
    Fits KMeans across k_range, returns a results table sorted by
    silhouette score (descending). Uses k-means++ init and n_init=10
    for stable centroid initialization.
    """
    k_range = k_range or KMEANS_K_RANGE
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        m = cluster_validity_metrics(X, labels, context=f"KMeans k={k}")
        rows.append({"algorithm": "KMeans", "params": f"k={k}", "k": k, "inertia": model.inertia_, **m})
        logger.info("KMeans k=%d -> silhouette=%.4f, DB=%.4f", k, m["silhouette"], m["davies_bouldin"])

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    return df


def tune_dbscan(X: np.ndarray, eps_grid=None, min_samples_grid=None) -> pd.DataFrame:
    """
    Grid search DBSCAN over (eps, min_samples). Configs producing a
    single cluster or >40% noise points are penalized (silhouette=-1)
    since they are not usable production segmentations.
    """
    eps_grid = eps_grid or DBSCAN_EPS_GRID
    min_samples_grid = min_samples_grid or DBSCAN_MIN_SAMPLES
    rows = []
    for eps in eps_grid:
        for ms in min_samples_grid:
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X)
            m = cluster_validity_metrics(X, labels, context=f"DBSCAN eps={eps}, min_samples={ms}")
            if m["n_clusters"] < 2 or m["noise_ratio"] > 0.40:
                m["silhouette"] = -1.0
            rows.append({"algorithm": "DBSCAN", "params": f"eps={eps}, min_samples={ms}",
                         "eps": eps, "min_samples": ms, "inertia": np.nan, **m})

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    logger.info("DBSCAN best config: %s", df.iloc[0]["params"] if len(df) else "none")
    return df


def tune_hierarchical(X: np.ndarray, k_range=None, linkage_methods=None) -> pd.DataFrame:
    """
    Grid search Agglomerative (Hierarchical) Clustering over
    (n_clusters, linkage method). 'ward' linkage requires euclidean
    affinity (default), which matches our scaled RFM features.
    """
    k_range = k_range or HIERARCHICAL_K_RANGE
    linkage_methods = linkage_methods or HIERARCHICAL_LINKAGE_METHODS
    rows = []
    for method in linkage_methods:
        for k in k_range:
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage=method)
                labels = model.fit_predict(X)
            except Exception as e:
                logger.warning("Hierarchical (%s, k=%d) failed: %s", method, k, e)
                continue
            m = cluster_validity_metrics(X, labels, context=f"Hierarchical {method} k={k}")
            rows.append({"algorithm": "Hierarchical", "params": f"k={k}, linkage={method}",
                         "k": k, "linkage": method, "inertia": np.nan, **m})
            logger.info("Hierarchical(%s) k=%d -> silhouette=%.4f, DB=%.4f",
                        method, k, m["silhouette"], m["davies_bouldin"])

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    return df


def compute_linkage_matrix(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """
    Computes the SciPy linkage matrix used to render a dendrogram.
    method: 'ward' | 'average' | 'complete' | 'single'
    """
    return linkage(X, method=method)


def plot_dendrogram(X: np.ndarray, method: str = "ward", truncate_last_p: int = 15,
                     save_path: str = None):
    """
    Renders (and optionally saves) a dendrogram for the given linkage
    method. truncate_last_p shows only the last p merged clusters
    for readability on large customer bases (541K txns -> thousands
    of customers would otherwise be unreadable).
    """
    import matplotlib.pyplot as plt

    Z = compute_linkage_matrix(X, method=method)
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=truncate_last_p, leaf_rotation=90,
               show_contracted=True, ax=ax)
    ax.set_title(f"Hierarchical Clustering Dendrogram (linkage={method}, truncated to last {truncate_last_p} merges)")
    ax.set_xlabel("Customer cluster / sample index")
    ax.set_ylabel("Distance (Ward linkage)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Dendrogram saved -> %s", save_path)

    return fig, Z


def labels_from_linkage(Z: np.ndarray, k: int) -> np.ndarray:
    """Cuts the dendrogram at k clusters using SciPy's fcluster (maxclust)."""
    return fcluster(Z, t=k, criterion="maxclust") - 1  # zero-index to match sklearn convention


def tune_gmm(X: np.ndarray, k_range=None) -> pd.DataFrame:
    """Gaussian Mixture Model search — soft-clustering challenger."""
    k_range = k_range or GMM_K_RANGE
    rows = []
    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3)
        labels = model.fit_predict(X)
        m = cluster_validity_metrics(X, labels, context=f"GMM k={k}")
        rows.append({"algorithm": "GMM", "params": f"k={k}", "k": k,
                     "inertia": np.nan, "bic": model.bic(X), **m})
    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    return df


def run_full_search(X: np.ndarray) -> pd.DataFrame:
    """
    Runs KMeans + DBSCAN + Hierarchical (Agglomerative) + GMM search
    and returns one combined, ranked leaderboard (mirrors
    model_experiment_results.csv convention from the other portfolio
    projects).
    """
    results = pd.concat(
        [tune_kmeans(X), tune_dbscan(X), tune_hierarchical(X), tune_gmm(X)],
        ignore_index=True, sort=False,
    )
    results = results.sort_values("silhouette", ascending=False).reset_index(drop=True)
    logger.info("Full search complete: %d configs evaluated across 4 algorithms", len(results))
    return results


def fit_best_kmeans(X: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    model.fit(X)
    return model


def fit_best_dbscan(X: np.ndarray, eps: float, min_samples: int) -> DBSCAN:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model


def fit_best_hierarchical(X: np.ndarray, k: int, method: str = "ward") -> AgglomerativeClustering:
    model = AgglomerativeClustering(n_clusters=k, linkage=method)
    model.fit(X)
    return model