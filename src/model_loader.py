# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Customer Segmentation Engine
# ============================================================
#
# CHALLENGER MODEL SYSTEM (adapted for UNSUPERVISED clustering):
#   Champion   = current production segmentation (latest_model.json)
#   Challenger = newly trained segmentation (best config from model_tuning)
#
#   Since there is no ground-truth label to compute F1/ROC-AUC on,
#   the 3 promotion gates use internal cluster-validity indices:
#
#     Gate 1: Silhouette improvement >= MIN_SILHOUETTE_IMPROVEMENT
#     Gate 2: Davies-Bouldin index   <= MAX_DB_INDEX   (lower = tighter, better-separated clusters)
#     Gate 3: Smallest segment share >= MIN_CLUSTER_SHARE (no unusable micro-segments)
#
#   Challenger promoted ONLY if it beats champion on ALL 3 gates.
#   Full comparison saved to segmentation_models/challenger_log.json
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import MODEL_DIR, MIN_SILHOUETTE_IMPROVEMENT, MAX_DB_INDEX, MIN_CLUSTER_SHARE

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads segmentation_models/latest_model.json -> loads the fitted
    clustering model + preprocessing pipeline (joblib bundle).
    Returns: (bundle_dict) with keys 'model', 'preprocessor',
    'feature_cols', 'persona_map', 'algorithm'.
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    logger.info("Champion model loaded: %s | algorithm=%s", model_path, registry.get("algorithm"))
    return bundle, registry


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    card_path = registry.get("model_card_path", "")
    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card missing — treating as no champion")
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})
    return {
        "model_name":     registry.get("model_name", "unknown"),
        "algorithm":      card.get("algorithm", "unknown"),
        "silhouette":     float(metrics.get("silhouette", -1.0)),
        "davies_bouldin": float(metrics.get("davies_bouldin", float("inf"))),
        "min_cluster_share": float(min(metrics.get("cluster_shares", {"0": 0}).values())) if metrics.get("cluster_shares") else 0.0,
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC (clustering version)
# ============================================================

def run_challenger_comparison(
    challenger_name:        str,
    challenger_algorithm:    str,
    challenger_silhouette:   float,
    challenger_db_index:     float,
    challenger_min_cluster_share: float,
    challenger_model_path:   str,
    challenger_card_path:    str = "",
) -> dict:
    """
    Compares challenger segmentation vs current champion using the
    3-gate silhouette / Davies-Bouldin / min-cluster-share system.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    champion = _load_champion_metrics()

    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_algorithm, challenger_card_path)
        result = {
            "decision": "PROMOTED",
            "reason": "No existing champion — first model auto-promoted",
            "challenger_name": challenger_name, "challenger_algorithm": challenger_algorithm,
            "challenger_silhouette": round(challenger_silhouette, 4),
            "challenger_db_index": round(challenger_db_index, 4),
            "champion_name": None,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champ_sil   = champion.get("silhouette", -1.0)
    champ_db    = champion.get("davies_bouldin", float("inf"))
    champ_name  = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-25s  sil=%.4f  DB=%.4f", champ_name, champ_sil, champ_db)
    logger.info("  Challenger: %-25s  sil=%.4f  DB=%.4f", challenger_name, challenger_silhouette, challenger_db_index)
    logger.info("=" * 55)

    gate1_silhouette   = (challenger_silhouette - champ_sil) >= MIN_SILHOUETTE_IMPROVEMENT
    gate2_db_index     = challenger_db_index <= MAX_DB_INDEX
    gate3_cluster_size = challenger_min_cluster_share >= MIN_CLUSTER_SHARE

    gates_passed = gate1_silhouette and gate2_db_index and gate3_cluster_size

    if gates_passed:
        decision = "PROMOTED"
        reason = (f"Challenger beats champion: silhouette {champ_sil:.4f} -> "
                  f"{challenger_silhouette:.4f} (+{challenger_silhouette - champ_sil:.4f})")
        logger.info("CHALLENGER PROMOTED -> new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_algorithm, challenger_card_path)
    else:
        decision = "REJECTED"
        failed = []
        if not gate1_silhouette:
            failed.append(f"silhouette improvement {challenger_silhouette - champ_sil:+.4f} < {MIN_SILHOUETTE_IMPROVEMENT}")
        if not gate2_db_index:
            failed.append(f"Davies-Bouldin {challenger_db_index:.4f} > {MAX_DB_INDEX}")
        if not gate3_cluster_size:
            failed.append(f"min cluster share {challenger_min_cluster_share:.4f} < {MIN_CLUSTER_SHARE}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("CHALLENGER REJECTED — champion '%s' retained", champ_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision": decision, "reason": reason,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name": challenger_name, "challenger_algorithm": challenger_algorithm,
        "challenger_silhouette": round(challenger_silhouette, 4),
        "challenger_db_index": round(challenger_db_index, 4),
        "challenger_min_cluster_share": round(challenger_min_cluster_share, 4),
        "champion_name": champ_name, "champion_silhouette": round(champ_sil, 4),
        "champion_db_index": round(champ_db, 4),
        "gates": {
            "silhouette_improvement_passed": gate1_silhouette,
            "db_index_passed": gate2_db_index,
            "cluster_size_passed": gate3_cluster_size,
        },
    }
    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, algorithm: str, model_card_path: str = None):
    registry = {
        "model_name": os.path.basename(model_path),
        "algorithm": algorithm,
        "model_card_path": model_card_path or "",
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated -> %s", registry["model_name"])


def _save_challenger_log(result: dict):
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(result)
    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Challenger log saved -> %s", CHALLENGER_LOG)
