import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow
import mlflow.pytorch
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

MODELS  = Path("models")
HORIZON = 28
SEQ_LEN = 30


class DemandLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = HORIZON, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train():
    log.info("Loading sequences...")
    X = np.load(MODELS / "X_sequences.npy")
    y = np.load(MODELS / "y_sequences.npy")

    # Train/val split
    split     = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val)

    train_ds  = TensorDataset(X_train_t, y_train_t)
    train_dl  = DataLoader(train_ds, batch_size=128, shuffle=True)

    input_size = X.shape[2]
    model      = DemandLSTM(input_size=input_size)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion  = nn.MSELoss()

    mlflow.set_experiment("supply-chain-lstm")

    with mlflow.start_run(run_name="lstm_demand"):
        mlflow.log_params({
            "input_size":  input_size,
            "hidden_size": 64,
            "num_layers":  2,
            "batch_size":  128,
            "epochs":      20,
            "horizon":     HORIZON,
            "seq_len":     SEQ_LEN,
        })

        log.info("Training LSTM...")
        for epoch in range(20):
            model.train()
            train_loss = 0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dl)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                mae      = torch.mean(torch.abs(val_pred - y_val_t)).item()
                rmse     = torch.sqrt(torch.mean((val_pred - y_val_t) ** 2)).item()

            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "val_loss":   round(val_loss, 4),
                "mae":        round(mae, 4),
                "rmse":       round(rmse, 4)
            }, step=epoch)

            if (epoch + 1) % 5 == 0:
                log.info(f"Epoch {epoch+1:02d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        # Save model
        torch.save(model.state_dict(), MODELS / "lstm_demand.pt")
        joblib.dump(input_size, MODELS / "lstm_input_size.pkl")
        mlflow.pytorch.log_model(model, "lstm_model")
        log.info("LSTM model saved.")
        log.info(f"Final — MAE: {mae:.2f} | RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train()