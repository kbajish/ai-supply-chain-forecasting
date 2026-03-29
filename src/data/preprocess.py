import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

DEMAND_DIR = Path("data/demand")
RISK_DIR   = Path("data/risk")
MODELS     = Path("models")

SEQ_LEN    = 30   # lookback window
HORIZON    = 28   # forecast horizon


def load_demand() -> pd.DataFrame:
    df = pd.read_csv(DEMAND_DIR / "demand_data.csv", parse_dates=["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding time series features...")
    df = df.copy()

    for sku, group in df.groupby("product_id"):
        idx = group.index
        df.loc[idx, "lag_7"]       = group["demand"].shift(7)
        df.loc[idx, "lag_14"]      = group["demand"].shift(14)
        df.loc[idx, "lag_30"]      = group["demand"].shift(30)
        df.loc[idx, "rolling_7"]   = group["demand"].shift(1).rolling(7).mean()
        df.loc[idx, "rolling_30"]  = group["demand"].shift(1).rolling(30).mean()

    df = df.dropna().reset_index(drop=True)
    log.info(f"Features added — shape: {df.shape}")
    return df


def build_sequences(df: pd.DataFrame, feature_cols: list):
    """Build LSTM sequences (X, y) per SKU."""
    log.info("Building LSTM sequences...")
    X_list, y_list = [], []

    for sku, group in df.groupby("product_id"):
        values = group[feature_cols].values
        demand = group["demand"].values

        for i in range(SEQ_LEN, len(values) - HORIZON + 1):
            X_list.append(values[i - SEQ_LEN:i])
            y_list.append(demand[i:i + HORIZON])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info(f"Sequences — X: {X.shape} | y: {y.shape}")
    return X, y


def load_risk() -> pd.DataFrame:
    df = pd.read_csv(RISK_DIR / "supplier_risk.csv")
    return df


def prepare_risk_features(df: pd.DataFrame):
    feature_cols = [
        "lead_time_days", "lead_time_variance", "single_source",
        "geo_risk_score", "financial_risk_score", "quality_score",
        "on_time_delivery_rate", "num_alternatives"
    ]
    le = LabelEncoder()
    y  = le.fit_transform(df["risk_level"])

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, MODELS / "risk_label_encoder.pkl")
    joblib.dump(feature_cols, MODELS / "risk_feature_cols.pkl")

    log.info(f"Risk classes: {list(le.classes_)}")
    return df[feature_cols], y, le


if __name__ == "__main__":
    df      = load_demand()
    df      = add_features(df)

    feature_cols = ["demand", "lag_7", "lag_14", "lag_30",
                    "rolling_7", "rolling_30", "day_of_week", "month", "is_weekend"]

    joblib.dump(feature_cols, MODELS / "lstm_feature_cols.pkl") if MODELS.exists() else None
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_cols, MODELS / "lstm_feature_cols.pkl")

    X, y = build_sequences(df, feature_cols)
    np.save(MODELS / "X_sequences.npy", X)
    np.save(MODELS / "y_sequences.npy", y)

    risk_df                   = load_risk()
    X_risk, y_risk, le_risk   = prepare_risk_features(risk_df)
    X_risk.to_parquet(MODELS  / "X_risk.parquet",  index=False)
    np.save(MODELS / "y_risk.npy", y_risk)

    log.info("Preprocessing complete.")