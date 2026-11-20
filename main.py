"""FastAPI service for the Retail Demand Forecasting model (Random Forest).

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "outputs/demand_model.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
LOW_CUT, HIGH_CUT = bundle["low_cut"], bundle["high_cut"]
LEAD_TIME = bundle["lead_time_days"]

app = FastAPI(title="Retail Demand Forecasting API", version="1.0")


class DemandRequest(BaseModel):
    # past demand (units sold on earlier days)
    Lag1: float = Field(..., ge=0, description="units sold 1 day ago")
    Lag2: float = Field(..., ge=0, description="units sold 2 days ago")
    Lag7: float = Field(..., ge=0, description="units sold 7 days ago")
    Lag14: float = Field(..., ge=0, description="units sold 14 days ago")
    Avg7: float = Field(..., ge=0, description="average units of previous 7 days")
    Avg28: float = Field(..., ge=0, description="average units of previous 28 days")
    Std7: float = Field(..., ge=0, description="std deviation of previous 7 days")
    # calendar
    DayOfWeek: int = Field(..., ge=0, le=6, description="0=Monday ... 6=Sunday")
    IsWeekend: int = Field(..., ge=0, le=1)
    Month: int = Field(..., ge=1, le=12)
    WeekOfYear: int = Field(..., ge=1, le=53)
    # planned events / price
    Holiday: int = Field(..., ge=0, le=1)
    Promotion: int = Field(..., ge=0, le=1)
    UnitPrice: float = Field(..., gt=0)
    PriceRatio: float = Field(..., gt=0, description="UnitPrice / usual (28-day max) price")
    # identifiers
    StoreID: int
    ProductID: int
    CategoryCode: int = Field(..., ge=0, le=4)
    # optional, only for stockout risk
    StockNextDay: Optional[float] = Field(None, ge=0, description="stock available tomorrow")
    NextDeliveryInDays: Optional[int] = Field(None, ge=1, description="days until next delivery, if an order is open")


class DemandResponse(BaseModel):
    model: str
    predicted_units: int
    demand_level: str
    stockout_risk: Optional[str] = None
    days_of_cover: Optional[float] = None
    suggested_order_qty: Optional[int] = None


def demand_level(units: float) -> str:
    return "Low" if units <= LOW_CUT else ("Medium" if units <= HIGH_CUT else "High")


@app.get("/")
def home():
    return {"message": "Retail Demand Forecasting API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"],
            "test_r2": round(bundle["test_r2"], 3),
            "forecast_accuracy_pct": round(bundle["forecast_accuracy_pct"], 1)}


@app.post("/predict", response_model=DemandResponse)
def predict(req: DemandRequest):
    try:
        row = pd.DataFrame([req.model_dump()])[FEATURES]      # exact training column order
        units = float(max(0.0, model.predict(row)[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    forecast = round(units)
    out = {"model": bundle["model_name"], "predicted_units": forecast,
           "demand_level": demand_level(forecast)}

    # stockout risk (same rules as the notebook) when stock is supplied
    if req.StockNextDay is not None:
        days_needed = req.NextDeliveryInDays - 1 if (req.NextDeliveryInDays and req.NextDeliveryInDays >= 2) else LEAD_TIME
        cover = req.StockNextDay / max(forecast, 1)
        if req.StockNextDay < forecast:
            risk = "High"
        elif cover < days_needed:
            risk = "Medium"
        else:
            risk = "Low"
        out.update({"stockout_risk": risk, "days_of_cover": round(cover, 2),
                    "suggested_order_qty": 0 if risk == "Low"
                    else int(max(0, np.ceil(forecast * (days_needed + 1) - req.StockNextDay)))})
    return out
