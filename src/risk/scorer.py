import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

MODELS   = Path("models")
RISK_DIR = Path("data/risk")


def load_artifacts():
    model      = joblib.load(MODELS / "xgboost_risk.pkl")
    le         = joblib.load(MODELS / "risk_label_encoder.pkl")
    feat_cols  = joblib.load(MODELS / "risk_feature_cols.pkl")
    explainer  = joblib.load(MODELS / "risk_shap_explainer.pkl")
    return model, le, feat_cols, explainer


def score_supplier(supplier_id: str) -> dict:
    df             = pd.read_csv(RISK_DIR / "supplier_risk.csv")
    supplier       = df[df["supplier_id"] == supplier_id]

    if supplier.empty:
        raise ValueError(f"Supplier {supplier_id} not found")

    supplier = supplier.iloc[0]
    model, le, feat_cols, explainer = load_artifacts()

    X          = pd.DataFrame([supplier[feat_cols]])
    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    risk_level = le.inverse_transform([pred_enc])[0]
    confidence = round(float(pred_proba.max()), 4)

    # SHAP explanation
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        values = shap_values[pred_enc][0]
    else:
        values = shap_values[0, :, pred_enc]

    top_features = [
        {"feature": feat_cols[i], "shap_value": round(float(values[i]), 4)}
        for i in np.argsort(np.abs(values))[::-1][:5]
    ]

    # Alert flag
    alert = risk_level == "High" or (
        risk_level == "Medium" and confidence > 0.85
    )

    return {
        "supplier_id":   supplier_id,
        "supplier_name": supplier["supplier_name"],
        "risk_level":    risk_level,
        "confidence":    confidence,
        "top_features":  top_features,
        "alert":         alert,
        "raw_score":     round(float(supplier["risk_score"]), 3)
    }


def get_high_risk_suppliers() -> list:
    df             = pd.read_csv(RISK_DIR / "supplier_risk.csv")
    model, le, feat_cols, _ = load_artifacts()

    X      = df[feat_cols]
    preds  = le.inverse_transform(model.predict(X))
    df["predicted_risk"] = preds

    high_risk = df[df["predicted_risk"] == "High"][[
        "supplier_id", "supplier_name", "risk_score", "predicted_risk"
    ]].to_dict("records")

    log.info(f"High risk suppliers: {len(high_risk)}")
    return high_risk


if __name__ == "__main__":
    # Score a single supplier
    result = score_supplier("SUP-0001")
    print(f"Supplier:   {result['supplier_name']}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Alert:      {result['alert']}")
    print(f"Top features:")
    for f in result["top_features"]:
        print(f"  {f['feature']}: {f['shap_value']}")

    print("\nHigh risk suppliers:")
    for s in get_high_risk_suppliers():
        print(f"  {s['supplier_id']} — {s['supplier_name']} (score: {s['risk_score']})")