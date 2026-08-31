# ============================================================
# MODEL CARD — Customer Segmentation Engine
# ============================================================
# Google Model Cards standard (https://modelcards.withgoogle.com/)
# adapted for an unsupervised clustering model.
# ============================================================

import json
import time
import logging

logger = logging.getLogger(__name__)


def build_model_card(
    model_name: str,
    algorithm: str,
    params: dict,
    metrics: dict,
    feature_cols: list,
    n_customers: int,
    persona_map: dict,
) -> dict:
    card = {
        "model_name": model_name,
        "model_type": "unsupervised_clustering",
        "algorithm": algorithm,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_details": {
            "owner": "Narendra Kalam",
            "version": "v1",
            "description": (
                "RFM-based customer segmentation engine for e-commerce/BFSI "
                "marketing teams. Clusters customers on Recency, Frequency, "
                "Monetary + behavioral features to power targeted campaigns."
            ),
            "hyperparameters": params,
        },
        "intended_use": {
            "primary_uses": [
                "Targeted marketing campaign segmentation",
                "Customer lifetime value tiering",
                "Churn / win-back campaign targeting",
            ],
            "out_of_scope": [
                "Individual credit decisioning",
                "Real-time fraud detection",
            ],
        },
        "training_data": {
            "source": "Kaggle — UK Online Retail (carrie1/ecommerce-data), 541K transactions",
            "n_customers": n_customers,
            "features": feature_cols,
        },
        "metrics": metrics,
        "personas": persona_map,
        "caveats": [
            "Clustering is unsupervised — cluster IDs are not stable across "
            "retrains; personas must be re-validated on each retrain.",
            "DBSCAN noise points (-1) are excluded from persona assignment.",
            "Model should be retrained if RFM feature PSI exceeds 0.20.",
        ],
    }
    return card


def save_model_card(card: dict, path: str):
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved -> %s", path)
