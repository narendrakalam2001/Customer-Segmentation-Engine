# ============================================================
# CONFIGURATION — Customer Segmentation Engine
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS       = -1

# ── Paths ────────────────────────────────────────────────────
RAW_DATA_PATH = r"D:\Data Science Datasets\Customer Segmentation Engine\data.csv"
MODEL_DIR     = "segmentation_models"
METRICS_LOG   = "segmentation_models/metrics_log.csv"
TESTS_DIR     = "tests"
SERVING_DIR   = "serving"

# ── RFM snapshot ──────────────────────────────────────────────
# snapshot_date = max(InvoiceDate) + SNAPSHOT_BUFFER_DAYS
SNAPSHOT_BUFFER_DAYS = 1

# ── Outlier clipping (IQR-based, same standard as other projects) ─
CLIP_FOLD = 1.5

# ── Clustering search space ────────────────────────────────────
KMEANS_K_RANGE      = list(range(2, 11))   # k = 2..10
GMM_K_RANGE         = list(range(2, 9))
DBSCAN_EPS_GRID     = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
DBSCAN_MIN_SAMPLES  = [5, 10, 15]

# Hierarchical (Agglomerative) Clustering
HIERARCHICAL_K_RANGE          = list(range(2, 9))
HIERARCHICAL_LINKAGE_METHODS  = ["ward", "average", "complete"]
DENDROGRAM_TRUNCATE_LAST_P    = 15   # readability: show only last-p merges on the plot

# ── Champion-Challenger promotion gates (3-gate system) ─────────
# Adapted for UNSUPERVISED clustering (no ground-truth labels):
#   Gate 1: Silhouette score must improve by >= MIN_SILHOUETTE_IMPROVEMENT
#   Gate 2: Davies-Bouldin index (lower=better) must be <= MAX_DB_INDEX
#   Gate 3: No cluster (excluding DBSCAN noise, label=-1) may hold
#           less than MIN_CLUSTER_SHARE of customers (avoids
#           degenerate / unusable micro-segments in production)
MIN_SILHOUETTE_IMPROVEMENT = 0.01
MAX_DB_INDEX                = 1.20
MIN_CLUSTER_SHARE           = 0.03      # each segment >= 3% of customers

# ── PSI drift thresholds (same standard as other projects) ──────
PSI_MODERATE = 0.10   # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH     = 0.20   # PSI >= 0.20 → critical drift, retrain

# ── RFM persona thresholds (quantile-based, business standard) ──
RFM_QUANTILES = [0.2, 0.4, 0.6, 0.8]

# ── Business impact assumptions (for cost-sensitive evaluation) ─
AVG_CAMPAIGN_COST_PER_CUSTOMER = 0.75     # currency units (₹/$) per targeted email/SMS
CHAMPIONS_UPLIFT_RATE          = 0.22     # expected response uplift from targeting Champions
AT_RISK_RETENTION_RATE         = 0.15     # expected retention uplift from win-back campaign

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
