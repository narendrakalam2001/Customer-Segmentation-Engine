# ============================================================
# TRAINING PIPELINE — Customer Segmentation Engine
# ============================================================
# End-to-end: load -> clean -> RFM -> leakage checks -> preprocess
#   -> model search (KMeans/DBSCAN/GMM) -> pick champion algorithm
#   -> bootstrap stability -> persona assignment -> model card
#   -> champion-challenger promotion -> save joblib bundle + logs
# ============================================================

import os
import json
import logging
import joblib
import time
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, RAW_DATA_PATH, MIN_CLUSTER_SHARE
from src.data_loader import load_and_clean
from src.preprocessing import build_rfm_table, build_clustering_preprocessor, RFM_FEATURES
from src.leakage_check import run_all_checks
from src.model_tuning import (
    run_full_search, fit_best_kmeans, fit_best_dbscan, fit_best_hierarchical, plot_dendrogram,
)
from src.metrics import cluster_validity_metrics, bootstrap_stability, cluster_size_shares
from src.evaluation import profile_clusters, assign_personas, business_impact_summary
from src.model_card import build_model_card, save_model_card
from src.model_loader import run_challenger_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(data_path: str = RAW_DATA_PATH) -> dict:
    logger.info("=" * 60)
    logger.info("CUSTOMER SEGMENTATION ENGINE — TRAINING PIPELINE START")
    logger.info("=" * 60)

    # ── 1. Load & clean ────────────────────────────────────────
    df = load_and_clean(data_path)

    # ── 2. RFM engineering ─────────────────────────────────────
    rfm_df, snapshot_date = build_rfm_table(df)

    # ── 3. Leakage checks ───────────────────────────────────────
    run_all_checks(df, rfm_df, snapshot_date, RFM_FEATURES)

    # ── 4. Preprocess (fit fresh pipeline) ──────────────────────
    from sklearn.base import clone
    preprocessor = clone(build_clustering_preprocessor())
    X = rfm_df[RFM_FEATURES].values
    X_scaled = preprocessor.fit_transform(X)

    # ── 5. Model search across KMeans / DBSCAN / GMM ────────────
    leaderboard = run_full_search(X_scaled)
    os.makedirs(MODEL_DIR, exist_ok=True)
    leaderboard.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)
    logger.info("Leaderboard saved. Top config:\n%s", leaderboard.head(5).to_string())

    # Business usability gate: reject configs where the smallest segment
    # is too small to act on (noise_ratio already penalizes DBSCAN configs
    # inside tune_dbscan; this additionally guards KMeans/Hierarchical/GMM
    # configs that silhouette alone would otherwise pick, e.g. a 99%/1% split).
    usable = leaderboard[leaderboard["noise_ratio"].fillna(0) <= 0.40].copy()
    filtered = usable[usable["min_cluster_share"].fillna(0) >= MIN_CLUSTER_SHARE]

    if len(filtered) == 0:
        logger.warning(
            "No config meets the %.0f%% min-cluster-share business gate — "
            "falling back to the top silhouette config anyway (flagged in model card).",
            MIN_CLUSTER_SHARE * 100,
        )
        best = leaderboard.iloc[0]
    else:
        best = filtered.sort_values("silhouette", ascending=False).iloc[0]

    algorithm = best["algorithm"]
    logger.info("Best algorithm selected: %s (%s)", algorithm, best["params"])

    if algorithm == "DBSCAN" and float(best.get("noise_ratio", 0)) > 0:
        logger.warning(
            "Champion is DBSCAN with %.1f%% noise points — these customers will NOT "
            "belong to any segment (cluster=-1 -> persona='Unclassified') and will be "
            "excluded from targeted campaigns. Review model_experiment_results.csv if "
            "every customer needs a persona (e.g. prefer the next-best KMeans/Hierarchical "
            "config instead).",
            float(best["noise_ratio"]) * 100,
        )

    # ── 6. Fit the winning model on full data ───────────────────
    centroids = None
    if algorithm == "KMeans":
        model = fit_best_kmeans(X_scaled, int(best["k"]))
        labels = model.labels_
    elif algorithm == "DBSCAN":
        model = fit_best_dbscan(X_scaled, float(best["eps"]), int(best["min_samples"]))
        labels = model.labels_
    elif algorithm == "Hierarchical":
        model = fit_best_hierarchical(X_scaled, int(best["k"]), method=best["linkage"])
        labels = model.labels_
        # AgglomerativeClustering has no predict() for new points -> precompute
        # centroids so segmentation_engine can nearest-centroid-assign at inference.
        centroids = np.array([X_scaled[labels == c].mean(axis=0) for c in sorted(set(labels))])
        # Save a dendrogram artifact for the monitoring dashboard / portfolio README
        try:
            _, _ = plot_dendrogram(X_scaled, method=best["linkage"],
                                   save_path=os.path.join(MODEL_DIR, "dendrogram.png"))
        except Exception as e:
            logger.warning("Dendrogram plot failed (non-fatal): %s", e)
    else:  # GMM
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=int(best["k"]), random_state=42, n_init=3)
        labels = model.fit_predict(X_scaled)

    metrics = cluster_validity_metrics(X_scaled, labels)
    metrics["cluster_shares"] = cluster_size_shares(labels)

    # ── 7. Bootstrap stability check ─────────────────────────────
    def _fit_predict(X_sub):
        if algorithm == "KMeans":
            from src.model_tuning import fit_best_kmeans as _f
            return _f(X_sub, int(best["k"])).labels_
        elif algorithm == "DBSCAN":
            from src.model_tuning import fit_best_dbscan as _f
            return _f(X_sub, float(best["eps"]), int(best["min_samples"])).labels_
        elif algorithm == "Hierarchical":
            from src.model_tuning import fit_best_hierarchical as _f
            return _f(X_sub, int(best["k"]), method=best["linkage"]).labels_
        else:
            from sklearn.mixture import GaussianMixture
            return GaussianMixture(n_components=int(best["k"]), random_state=42, n_init=3).fit_predict(X_sub)

    stability = bootstrap_stability(_fit_predict, X_scaled, n_iter=10)
    metrics["bootstrap_stability_silhouette"] = round(stability, 4)

    # ── 8. Cluster profiling + persona naming ────────────────────
    profile = profile_clusters(rfm_df, labels, RFM_FEATURES)
    profile = assign_personas(profile)
    persona_map = dict(zip(profile["cluster"].astype(str), profile["persona"]))

    impact = business_impact_summary(profile)
    profile.to_csv(os.path.join(MODEL_DIR, "cluster_profile.csv"), index=False)
    impact.to_csv(os.path.join(MODEL_DIR, "business_impact.csv"), index=False)

    # ── 9. Save model bundle ──────────────────────────────────────
    # IMPORTANT: filename includes a timestamp so this run NEVER overwrites
    # the current champion's file on disk, even when the algorithm name is
    # identical to the existing champion. Without this, a rejected challenger
    # would still have silently clobbered the champion's .joblib file before
    # the promotion gates were even checked (the file save always happens
    # here, ahead of the champion-challenger comparison below).
    run_id = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"segmentation_model_{algorithm}_{run_id}.joblib"
    model_path = os.path.join(MODEL_DIR, model_name)
    bundle = {
        "model": model, "preprocessor": preprocessor,
        "feature_cols": RFM_FEATURES, "persona_map": persona_map,
        "algorithm": algorithm, "snapshot_date": str(snapshot_date),
        "centroids": centroids.tolist() if centroids is not None else None,
    }
    joblib.dump(bundle, model_path)

    # ── 10. Model card ──────────────────────────────────────────
    card = build_model_card(
        model_name=model_name, algorithm=algorithm,
        params=best.get("params"), metrics=metrics,
        feature_cols=RFM_FEATURES, n_customers=len(rfm_df),
        persona_map=persona_map,
    )
    card_path = os.path.join(MODEL_DIR, f"model_card_{algorithm}_{run_id}.json")
    save_model_card(card, card_path)

    # Save reference RFM distribution for future PSI drift comparisons
    rfm_df[RFM_FEATURES].to_csv(os.path.join(MODEL_DIR, "reference_rfm_distribution.csv"), index=False)

    # ── 11. Champion-Challenger promotion ────────────────────────
    min_share = min(metrics["cluster_shares"].values()) if metrics["cluster_shares"] else 0.0
    decision = run_challenger_comparison(
        challenger_name=model_name, challenger_algorithm=algorithm,
        challenger_silhouette=metrics["silhouette"], challenger_db_index=metrics["davies_bouldin"],
        challenger_min_cluster_share=min_share, challenger_model_path=model_path,
        challenger_card_path=card_path,
    )

    # Housekeeping: a REJECTED challenger's artifacts aren't needed on disk —
    # its metrics are already preserved in model_experiment_results.csv and
    # challenger_log.json. Removing them keeps segmentation_models/ from
    # accumulating clutter across repeated retrains, and makes it visually
    # obvious which .joblib file is the actual current champion.
    if decision["decision"] == "REJECTED":
        for f in (model_path, card_path):
            try:
                os.remove(f)
            except OSError:
                pass
        logger.info("Rejected challenger artifacts removed from disk (champion file untouched).")

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE — decision: %s", decision["decision"])
    logger.info("=" * 60)

    return {
        "metrics": metrics, "leaderboard": leaderboard, "profile": profile,
        "business_impact": impact, "decision": decision, "model_path": model_path,
        "card_path": card_path,
    }


if __name__ == "__main__":
    run_training_pipeline()