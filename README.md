# 📦 AI Supply Chain Forecasting

![CI](https://github.com/kbajish/ai-supply-chain-forecasting/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-compose-blue)

An AI-driven supply chain intelligence system that combines LSTM-based demand forecasting with XGBoost supply risk scoring, built on synthetic industrial data modelled around automotive and manufacturing scenarios relevant to the German market. The system forecasts 28-day product demand per SKU, scores supplier risk with SHAP explanations, generates inventory reorder recommendations, and produces planner-ready LLM narratives via LangChain + Ollama.

---

## 🚀 Key Features

- 📈 LSTM demand forecasting — 28-day horizon with prediction intervals per SKU
- ⚠️ XGBoost supply risk scorer — Low / Medium / High risk classification per supplier
- 🔍 SHAP explanations — top risk drivers per supplier decision
- 📦 Inventory reorder recommendations — forecast vs safety stock alerts
- 🤖 LLM-powered planner insights using LangChain + Ollama (local, no API key)
- 📉 MLflow experiment tracking for both LSTM and XGBoost model tracks
- ⚡ FastAPI backend (`/forecast`, `/risk`, `/insights`, `/health`)
- 📊 Streamlit dashboard — forecast chart, risk heatmap, LLM insights panel
- 🐳 Docker Compose for full-stack deployment
- 🔄 GitHub Actions CI/CD with model and import tests

---

## 🧠 System Architecture

```
Synthetic Supply Chain Data
        ↓
src/data/generator.py         — 50 SKUs demand series + 100 supplier risk records
src/data/preprocess.py        — lag features, rolling stats, calendar features
        ↓
src/models/lstm_model.py      — LSTM demand forecasting (PyTorch)
src/models/risk_model.py      — XGBoost supply risk scorer
        ↓
MLflow                        — experiment tracking (loss, MAE, RMSE)
        ↓
src/forecasting/predictor.py  — 28-day forecast + prediction intervals
src/risk/scorer.py            — risk score + SHAP explanations + alerts
        ↓
src/llm/insight_chain.py      — LangChain + Ollama planner narratives
        ↓
api/main.py                   — FastAPI (/forecast, /risk, /insights, /health)
        ↓
dashboard/app.py              — Streamlit dashboard
```

---

## ⚙️ How It Works

Synthetic industrial supply chain data is generated for 50 automotive SKUs (brake systems, sensors, control units) with 2 years of daily demand incorporating trend, weekly and monthly seasonality, and random supply disruption events. A separate supplier risk dataset covers 100 suppliers with features including lead time variance, single-source dependency, geopolitical risk score, and financial stability score.

An LSTM network trained on the demand time series forecasts the next 28 days per SKU with upper and lower prediction intervals. A separate XGBoost classifier scores each supplier as Low, Medium, or High risk, with SHAP values explaining the top contributing risk factors. Inventory reorder alerts are triggered when the forecasted demand exceeds current stock minus a safety buffer.

LangChain + Ollama synthesises forecast data, risk alerts, and reorder recommendations into plain-English planner narratives. All results are served via FastAPI and visualised in a Streamlit dashboard with an interactive forecast chart, supplier risk heatmap, and LLM insights panel.

---

## 📊 Dashboard Overview

The Streamlit dashboard provides:

- 📈 Interactive demand forecast chart with prediction intervals
- ⚠️ Supplier risk heatmap — colour-coded Low / Medium / High
- 📦 Reorder alert table — SKUs requiring immediate action
- 🤖 LLM-generated planner insight narrative
- 📉 MLflow experiment metrics summary

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Demand forecasting | LSTM (PyTorch) |
| Risk scoring | XGBoost + SHAP |
| Data | Synthetic industrial supply chain |
| LLM insights | LangChain + Ollama (llama3.2, local) |
| Experiment tracking | MLflow |
| Backend | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📂 Project Structure

```
ai-supply-chain-forecasting/
│
├── data/                          # Generated data (not committed)
│   ├── demand/
│   │   └── demand_data.csv        # 50 SKUs × 730 days
│   └── risk/
│       └── supplier_risk.csv      # 100 supplier risk records
│
├── src/
│   ├── data/
│   │   ├── generator.py           # Synthetic data generator
│   │   └── preprocess.py          # Feature engineering pipeline
│   ├── models/
│   │   ├── lstm_model.py          # LSTM model definition + training
│   │   └── risk_model.py          # XGBoost risk scorer training
│   ├── forecasting/
│   │   └── predictor.py           # Demand forecast inference
│   ├── risk/
│   │   └── scorer.py              # Risk scoring + SHAP + alerts
│   └── llm/
│       └── insight_chain.py       # LangChain + Ollama insights
│
├── models/                        # Saved model artifacts (not committed)
│
├── api/
│   └── main.py                    # FastAPI app
│
├── dashboard/
│   └── app.py                     # Streamlit dashboard
│
├── tests/
│   ├── test_generator.py          # Data generator tests
│   └── test_imports.py            # CI-safe import tests
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions pipeline
│
├── mlruns/                        # MLflow tracking (not committed)
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── .env.example
├── requirements.api.txt
├── requirements.dashboard.txt
├── requirements.dev.txt
├── AI_ACT_COMPLIANCE.md
└── README.md
```

---

## ▶️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kbajish/ai-supply-chain-forecasting.git
cd ai-supply-chain-forecasting
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

### 3. Generate synthetic data and train models
```bash
python -m src.data.generator
python -m src.models.lstm_model
python -m src.models.risk_model
```

### 4. Start all services
```bash
docker compose up --build
```

### 5. Access services

| Service   | URL                        |
|-----------|----------------------------|
| API       | http://localhost:8000      |
| API docs  | http://localhost:8000/docs |
| Dashboard | http://localhost:8501      |
| MLflow    | http://localhost:5000      |

---

## 🧪 API Endpoints

| Method | Endpoint     | Description                                          |
|--------|--------------|------------------------------------------------------|
| `POST` | `/forecast`  | 28-day demand forecast for a given SKU               |
| `POST` | `/risk`      | Supplier risk score + SHAP explanations              |
| `POST` | `/insights`  | LLM-generated planner narrative                      |
| `GET`  | `/health`    | Health check                                         |

---

## 📈 Future Improvements

- M5 Forecasting dataset integration (Walmart retail benchmark)
- Kafka real-time streaming for live demand signals
- Transformer-based forecasting (Temporal Fusion Transformer)
- Multi-echelon inventory optimisation
- Cloud deployment (AWS/GCP) with managed MLflow

---

## 👤 Author

Experienced IT professional with a background in development, cybersecurity, and ERP systems, with expertise in Industrial AI. Focused on building production-ready AI systems with explainability, LLM integration, and MLOps best practices.
