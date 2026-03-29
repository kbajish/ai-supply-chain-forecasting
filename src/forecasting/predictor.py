import numpy as np
import torch
import pandas as pd
import joblib
from pathlib import Path
from src.models.lstm_model import DemandLSTM
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

MODELS     = Path("models")
DEMAND_DIR = Path("data/demand")
SEQ_LEN    = 30
HORIZON    = 28


def load_model() -> DemandLSTM:
    input_size = joblib.load(MODELS / "lstm_input_size.pkl")
    model      = DemandLSTM(input_size=input_size)
    model.load_state_dict(torch.load(MODELS / "lstm_demand.pt", map_location="cpu"))
    model.eval()
    return model


def get_last_sequence(product_id: str) -> np.ndarray:
    df           = pd.read_csv(DEMAND_DIR / "demand_data.csv", parse_dates=["date"])
    df           = df[df["product_id"] == product_id].sort_values("date")
    feature_cols = joblib.load(MODELS / "lstm_feature_cols.pkl")

    # Rebuild lag features for last sequence
    df["lag_7"]      = df["demand"].shift(7)
    df["lag_14"]     = df["demand"].shift(14)
    df["lag_30"]     = df["demand"].shift(30)
    df["rolling_7"]  = df["demand"].shift(1).rolling(7).mean()
    df["rolling_30"] = df["demand"].shift(1).rolling(30).mean()
    df               = df.dropna()

    if len(df) < SEQ_LEN:
        raise ValueError(f"Not enough data for {product_id}")

    seq = df[feature_cols].values[-SEQ_LEN:]
    return seq.astype(np.float32)


def forecast(product_id: str, n_samples: int = 50) -> dict:
    """
    Forecast demand for next HORIZON days with prediction intervals
    using Monte Carlo dropout for uncertainty estimation.
    """
    model = load_model()
    seq   = get_last_sequence(product_id)
    X     = torch.tensor(seq).unsqueeze(0)

    # Enable dropout for MC sampling
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.apply(enable_dropout)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X).squeeze().numpy()
            preds.append(pred)

    preds  = np.array(preds)
    mean   = preds.mean(axis=0)
    lower  = np.percentile(preds, 10, axis=0)
    upper  = np.percentile(preds, 90, axis=0)
    mean   = np.clip(mean, 0, None).round().astype(int)
    lower  = np.clip(lower, 0, None).round().astype(int)
    upper  = np.clip(upper, 0, None).round().astype(int)

    dates  = pd.date_range(
        start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        periods = HORIZON
    ).strftime("%Y-%m-%d").tolist()

    log.info(f"Forecast for {product_id} — avg demand: {mean.mean():.0f} units/day")

    return {
        "product_id": product_id,
        "horizon":    HORIZON,
        "dates":      dates,
        "forecast":   mean.tolist(),
        "lower":      lower.tolist(),
        "upper":      upper.tolist(),
        "avg_demand": int(mean.mean()),
        "total_28d":  int(mean.sum())
    }


if __name__ == "__main__":
    result = forecast("SKU-0001")
    print(f"Product:      {result['product_id']}")
    print(f"Avg demand:   {result['avg_demand']} units/day")
    print(f"Total 28-day: {result['total_28d']} units")
    print(f"Day 1 forecast: {result['forecast'][0]} [{result['lower'][0]}-{result['upper'][0]}]")