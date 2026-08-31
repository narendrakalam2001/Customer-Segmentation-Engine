# ============================================================
# DATA LOADER — Customer Segmentation Engine
# ============================================================
# Source dataset : UCI / Kaggle "Online Retail" (carrie1/ecommerce-data)
#                   541,909 transactions | UK-based online retailer
# Columns expected: InvoiceNo, StockCode, Description, Quantity,
#                    InvoiceDate, UnitPrice, CustomerID, Country
# ============================================================

import os
import logging
import pandas as pd

from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw Online Retail transaction log.

    Handles the common encoding issue with this dataset
    (UnicodeDecodeError on utf-8) by falling back to 'ISO-8859-1'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Download 'Online Retail' from Kaggle "
            "(https://www.kaggle.com/datasets/carrie1/ecommerce-data) "
            f"and place the CSV at {path}."
        )

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("utf-8 decode failed — retrying with ISO-8859-1")
        df = pd.read_csv(path, encoding="ISO-8859-1")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    logger.info("Raw data loaded: %d rows, %d columns", len(df), df.shape[1])
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard cleaning for the Online Retail dataset:
      - drop rows with missing CustomerID (can't attribute to a customer)
      - drop cancelled invoices (InvoiceNo starting with 'C')
      - drop non-positive Quantity / UnitPrice (returns, data errors)
      - parse InvoiceDate to datetime
      - de-duplicate exact duplicate rows
    """
    n0 = len(df)
    out = df.copy()

    out = out.dropna(subset=["CustomerID"])
    out["CustomerID"] = out["CustomerID"].astype(int).astype(str)

    out["InvoiceNo"] = out["InvoiceNo"].astype(str)
    out = out[~out["InvoiceNo"].str.startswith("C")]

    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]

    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])

    out = out.drop_duplicates()

    logger.info(
        "Cleaning: %d -> %d rows retained (%.1f%%), %d unique customers",
        n0, len(out), 100 * len(out) / max(n0, 1), out["CustomerID"].nunique()
    )
    return out.reset_index(drop=True)


def load_and_clean(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load raw -> clean -> return."""
    return clean_transactions(load_raw_data(path))
