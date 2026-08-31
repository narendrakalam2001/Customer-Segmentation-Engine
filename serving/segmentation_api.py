# ============================================================
# SERVING — Customer Segmentation Engine (FastAPI)
# ============================================================

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.prediction_service import predict_segment, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Segmentation Engine API",
    description="RFM + KMeans/DBSCAN customer segmentation for targeted marketing campaigns.",
    version="1.0.0",
)


class CustomerRFM(BaseModel):
    Recency:        float = Field(..., ge=0, description="Days since last purchase")
    Frequency:       float = Field(..., ge=0, description="Number of distinct invoices")
    Monetary:        float = Field(..., ge=0, description="Total historical spend")
    AvgBasketValue:  float = Field(..., ge=0, description="Average order value")
    UniqueProducts:  float = Field(..., ge=0, description="Distinct products purchased")
    Tenure:          float = Field(..., ge=0, description="Days between first and last purchase")

    class Config:
        json_schema_extra = {
            "example": {
                "Recency": 12, "Frequency": 8, "Monetary": 2450.50,
                "AvgBasketValue": 306.31, "UniqueProducts": 34, "Tenure": 210,
            }
        }


@app.get("/health")
def health():
    return {"status": "ok", "service": "customer-segmentation-engine"}


@app.get("/model_info")
def model_info():
    try:
        return get_model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict")
def predict(customer: CustomerRFM):
    try:
        result = predict_segment(customer.dict())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
