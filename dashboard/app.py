import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "AI Supply Chain Forecasting",
    page_icon  = "📦",
    layout     = "wide"
)

st.title("📦 AI Supply Chain Forecasting — Dashboard")
st.caption("LSTM demand forecasting + XGBoost supply risk scoring")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    product_ids = [f"SKU-{i:04d}" for i in range(1, 51)]
    selected_sku = st.selectbox("Select SKU", product_ids)

    supplier_ids = [f"SUP-{i:04d}" for i in range(1, 101)]
    selected_sup = st.selectbox("Select Supplier", supplier_ids)

    st.markdown("---")
    if st.button("Check API health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            st.success(f"API online — {r.json()}")
        except Exception:
            st.error("API not reachable")

# ── Key metrics ───────────────────────────────────────────────────
try:
    risk_df = pd.read_csv("data/risk/supplier_risk.csv")
    demand_df = pd.read_csv("data/demand/demand_data.csv")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs",             demand_df["product_id"].nunique())
    col2.metric("Total suppliers",         len(risk_df))
    col3.metric("High risk suppliers",     int((risk_df["risk_level"] == "High").sum()))
    col4.metric("Avg daily demand",        f"{demand_df['demand'].mean():.0f} units")
except Exception:
    st.warning("Could not load local data files.")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Supplier Risk", "Planner Insights"])

# ── Tab 1: Demand Forecast ────────────────────────────────────────
with tab1:
    st.subheader(f"28-day demand forecast — {selected_sku}")

    if st.button("Run forecast", key="forecast_btn"):
        with st.spinner("Generating forecast..."):
            try:
                resp   = requests.post(
                    f"{API_URL}/forecast",
                    json    = {"product_id": selected_sku},
                    timeout = 30
                )
                result = resp.json()

                col1, col2 = st.columns(2)
                col1.metric("Avg daily demand", f"{result['avg_demand']} units")
                col2.metric("Total 28-day",     f"{result['total_28d']} units")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x    = result["dates"],
                    y    = result["forecast"],
                    mode = "lines",
                    name = "Forecast",
                    line = dict(color="royalblue", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x         = result["dates"] + result["dates"][::-1],
                    y         = result["upper"] + result["lower"][::-1],
                    fill      = "toself",
                    fillcolor = "rgba(65,105,225,0.15)",
                    line      = dict(color="rgba(255,255,255,0)"),
                    name      = "Prediction interval"
                ))
                fig.update_layout(
                    xaxis_title = "Date",
                    yaxis_title = "Units",
                    height      = 400,
                    hovermode   = "x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ── Tab 2: Supplier Risk ──────────────────────────────────────────
with tab2:
    st.subheader(f"Supplier risk — {selected_sup}")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Score supplier", key="risk_btn"):
            with st.spinner("Scoring..."):
                try:
                    resp   = requests.post(
                        f"{API_URL}/risk",
                        json    = {"supplier_id": selected_sup},
                        timeout = 15
                    )
                    result = resp.json()
                    st.session_state["risk_result"] = result
                except Exception as e:
                    st.error(f"Error: {e}")

        if "risk_result" in st.session_state:
            r = st.session_state["risk_result"]
            level = r["risk_level"]
            if level == "High":
                st.error(f"Risk level: {level}")
            elif level == "Medium":
                st.warning(f"Risk level: {level}")
            else:
                st.success(f"Risk level: {level}")

            st.metric("Confidence", f"{r['confidence']:.2%}")
            st.metric("Raw risk score", r["raw_score"])
            if r["alert"]:
                st.error("ALERT — Immediate action required")

    with col2:
        if "risk_result" in st.session_state:
            r     = st.session_state["risk_result"]
            feats = pd.DataFrame(r["top_features"])
            fig   = go.Figure(go.Bar(
                x           = feats["shap_value"],
                y           = feats["feature"],
                orientation = "h",
                marker_color = ["crimson" if v > 0 else "steelblue"
                                for v in feats["shap_value"]]
            ))
            fig.update_layout(
                title  = "SHAP feature contributions",
                height = 300,
                xaxis_title = "SHAP value"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("High risk supplier alerts")
    if st.button("Load alerts", key="alerts_btn"):
        with st.spinner("Loading..."):
            try:
                resp    = requests.get(f"{API_URL}/risk/alerts", timeout=10)
                alerts  = resp.json()
                alert_df = pd.DataFrame(alerts)
                st.dataframe(alert_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ── Tab 3: Planner Insights ───────────────────────────────────────
with tab3:
    st.subheader("LLM planner insight")
    st.caption(f"SKU: {selected_sku} | Supplier: {selected_sup}")

    if st.button("Generate insight", key="insight_btn"):
        with st.spinner("Generating planner narrative (this may take 20-30 seconds)..."):
            try:
                resp   = requests.post(
                    f"{API_URL}/insights",
                    json    = {
                        "product_id":  selected_sku,
                        "supplier_id": selected_sup
                    },
                    timeout = 120
                )
                result = resp.json()

                st.info(result["narrative"])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg demand",   f"{result['forecast']['avg_demand']} units/day")
                    st.metric("Total 28-day", f"{result['forecast']['total_28d']} units")
                with col2:
                    st.metric("Supplier risk", result["risk"]["risk_level"])
                    st.metric("Risk score",    result["risk"]["raw_score"])

            except Exception as e:
                st.error(f"Error: {e}")