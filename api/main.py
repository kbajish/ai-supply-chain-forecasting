import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.forecasting.predictor import forecast
from src.risk.scorer import score_supplier, get_high_risk_suppliers
from src.llm.insight_chain import generate_insight

load_dotenv()

app = FastAPI(
    title       = "AI Supply Chain Forecasting API",
    description = "LSTM demand forecasting and XGBoost supply risk scoring",
    version     = "1.0.0"
)


class ForecastRequest(BaseModel):
    product_id: str

class RiskRequest(BaseModel):
    supplier_id: str

class InsightRequest(BaseModel):
    product_id:   str
    supplier_id:  str


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        return forecast(req.product_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk")
def get_risk(req: RiskRequest):
    try:
        return score_supplier(req.supplier_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/alerts")
def get_alerts():
    try:
        return get_high_risk_suppliers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insights")
def get_insights(req: InsightRequest):
    try:
        fc        = forecast(req.product_id)
        risk      = score_supplier(req.supplier_id)
        high_risk = get_high_risk_suppliers()

        inventory_status = (
            f"Current stock: {fc['avg_demand'] * 7} units. "
            f"Safety stock: {fc['avg_demand'] * 3} units. "
            f"Reorder point: {fc['avg_demand'] * 5} units."
        )

        narrative = generate_insight(
            product_name     = req.product_id,
            avg_demand       = fc["avg_demand"],
            total_28d        = fc["total_28d"],
            high_risk        = high_risk[:3],
            inventory_status = inventory_status
        )

        return {
            "product_id":  req.product_id,
            "supplier_id": req.supplier_id,
            "forecast":    fc,
            "risk":        risk,
            "narrative":   narrative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))