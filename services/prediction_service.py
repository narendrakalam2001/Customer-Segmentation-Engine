# ============================================================
# PREDICTION SERVICE — Customer Segmentation Engine
# ============================================================
# Thin service layer between the FastAPI routes and the core
# segmentation engine / model registry. Caches the loaded champion
# model bundle in-process to avoid a joblib.load() per request.
# ============================================================

import os
import csv
import time
import logging

from src.model_loader import load_latest_model
from src.segmentation_engine import assign_segment

logger = logging.getLogger(__name__)

_CACHE = {"bundle": None, "registry": None}
LOG_PATH = "logs/prediction_logs.csv"


def _get_bundle():
    if _CACHE["bundle"] is None:
        bundle, registry = load_latest_model()
        _CACHE["bundle"] = bundle
        _CACHE["registry"] = registry
    return _CACHE["bundle"], _CACHE["registry"]


def get_model_info() -> dict:
    bundle, registry = _get_bundle()
    return {
        "algorithm": bundle.get("algorithm"),
        "model_name": registry.get("model_name"),
        "promoted_at": registry.get("promoted_at"),
        "n_segments": len(bundle.get("persona_map", {})),
        "personas": list(set(bundle.get("persona_map", {}).values())),
        "features_required": bundle.get("feature_cols"),
    }


def predict_segment(rfm_input: dict) -> dict:
    bundle, _ = _get_bundle()
    result = assign_segment(bundle, rfm_input)
    _log_prediction(rfm_input, result)
    return result


def _log_prediction(rfm_input: dict, result: dict):
    """Appends every prediction to logs/prediction_logs.csv for the monitoring dashboard."""
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **rfm_input,
        "cluster": result["cluster"], "persona": result["persona"],
        "confidence": result["confidence"], "algorithm": result["algorithm"],
    }
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
