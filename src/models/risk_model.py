import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

MODELS = Path("models")


def train():
    log.info("Loading risk data...")
    X        = pd.read_parquet(MODELS / "X_risk.parquet")
    y        = np.load(MODELS / "y_risk.npy")
    le       = joblib.load(MODELS / "risk_label_encoder.pkl")
    feat_cols = joblib.load(MODELS / "risk_feature_cols.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("supply-chain-risk")

    with mlflow.start_run(run_name="xgboost_risk"):
        params = {
            "n_estimators":  200,
            "max_depth":     6,
            "learning_rate": 0.1,
            "subsample":     0.8,
            "random_state":  42,
            "n_jobs":        -1,
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(model, "xgboost_risk")

        log.info(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        log.info("\n" + classification_report(
            y_test, preds, target_names=le.classes_
        ))

        # SHAP explainer
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        joblib.dump(explainer, MODELS / "risk_shap_explainer.pkl")

        # Save model and feature info
        joblib.dump(model,     MODELS / "xgboost_risk.pkl")
        log.info("Risk model and SHAP explainer saved.")


if __name__ == "__main__":
    train()