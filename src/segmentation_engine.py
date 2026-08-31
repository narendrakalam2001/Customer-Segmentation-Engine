# ============================================================
# SEGMENTATION ENGINE — Customer Segmentation Engine
# ============================================================
# Inference-time logic: given a new/existing customer's RFM values,
# assigns the nearest cluster + persona + a confidence score based
# on normalized distance to centroid (KMeans) or trained density
# region (DBSCAN via nearest-core-point heuristic).
# ============================================================

import logging
import numpy as np

from src.preprocessing import RFM_FEATURES

logger = logging.getLogger(__name__)


def assign_segment(bundle: dict, rfm_input: dict) -> dict:
    """
    bundle: loaded via model_loader.load_latest_model() ->
            {'model', 'preprocessor', 'feature_cols', 'persona_map', 'algorithm'}
    rfm_input: dict with keys Recency, Frequency, Monetary, AvgBasketValue,
               UniqueProducts, Tenure

    Returns: {'cluster': int, 'persona': str, 'confidence': float}
    """
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle.get("feature_cols", RFM_FEATURES)
    persona_map  = bundle.get("persona_map", {})
    algorithm    = bundle.get("algorithm", "KMeans")

    x_raw = np.array([[rfm_input[c] for c in feature_cols]], dtype=float)
    x_scaled = preprocessor.transform(x_raw)

    if algorithm == "KMeans":
        cluster = int(model.predict(x_scaled)[0])
        centers = model.cluster_centers_
        dists = np.linalg.norm(centers - x_scaled, axis=1)
        d_to_own = dists[cluster]
        d_sorted = np.sort(dists)
        # confidence: how much closer to own centroid vs next-nearest (0-1)
        confidence = float(1.0 - (d_to_own / (d_sorted[1] + 1e-9))) if len(d_sorted) > 1 else 1.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

    elif algorithm == "GMM":
        cluster = int(model.predict(x_scaled)[0])
        proba = model.predict_proba(x_scaled)[0]
        confidence = float(np.max(proba))

    elif algorithm == "Hierarchical":
        # AgglomerativeClustering has no predict() for new points — assign
        # to the nearest centroid, computed once at training time and
        # stored in the bundle (bundle['centroids']).
        centers = bundle.get("centroids")
        if centers is None:
            raise ValueError("Hierarchical model bundle missing 'centroids' — retrain to regenerate.")
        centers = np.asarray(centers)
        dists = np.linalg.norm(centers - x_scaled, axis=1)
        cluster = int(np.argmin(dists))
        d_to_own = dists[cluster]
        d_sorted = np.sort(dists)
        confidence = float(1.0 - (d_to_own / (d_sorted[1] + 1e-9))) if len(d_sorted) > 1 else 1.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

    else:  # DBSCAN — assign to nearest core-sample's cluster
        core_samples = model.components_
        core_labels  = model.labels_[model.core_sample_indices_]
        dists = np.linalg.norm(core_samples - x_scaled, axis=1)
        nearest_idx = int(np.argmin(dists))
        cluster = int(core_labels[nearest_idx])
        d_to_nearest = float(dists[nearest_idx])
        confidence = float(np.clip(1.0 - min(d_to_nearest / (model.eps * 2), 1.0), 0.0, 1.0))
        if d_to_nearest > model.eps * 3:
            cluster = -1  # too far from any core point -> noise

    persona = persona_map.get(str(cluster), "Unclassified" if cluster == -1 else "Unknown")

    return {
        "cluster": cluster,
        "persona": persona,
        "confidence": round(confidence, 4),
        "algorithm": algorithm,
    }
