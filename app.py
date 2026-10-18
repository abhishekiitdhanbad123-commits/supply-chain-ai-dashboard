
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

st.set_page_config(
    page_title="Supply Chain AI Dashboard",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Supply Chain AI/ML Dashboard")
st.caption("Demand Forecasting & Inventory Optimization")

forecast_path = OUTPUT_DIR / "forecast_7_days.csv"
inventory_path = OUTPUT_DIR / "inventory_recommendations.csv"

if not forecast_path.exists() or not inventory_path.exists():
    st.error("Run the forecasting and inventory scripts first.")
    st.stop()

forecast = pd.read_csv(forecast_path)
inventory = pd.read_csv(inventory_path)

# KPI cards
total_forecast = forecast["forecast_qty"].sum()
total_stock = inventory["available_stock"].sum()
total_replenishment = inventory["suggested_replenishment"].sum()

c1, c2, c3 = st.columns(3)

c1.metric("7-Day Forecast", f"{total_forecast:,.0f} units")
c2.metric("Illustrative Available Stock", f"{total_stock:,.0f} units")
c3.metric("Suggested Replenishment", f"{total_replenishment:,.0f} units")
# Forecast model performance
st.subheader("Model Performance")

predictions_path = OUTPUT_DIR / "test_predictions.csv"

if predictions_path.exists():
    predictions = pd.read_csv(predictions_path)

    actual = predictions["sales_qty"]
    predicted = predictions["prediction"]

    mae = (actual - predicted).abs().mean()
    rmse = ((actual - predicted) ** 2).mean() ** 0.5

    nonzero_actual = actual != 0
    mape = (
        (actual[nonzero_actual] - predicted[nonzero_actual]).abs()
        / actual[nonzero_actual]
    ).mean() * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f} units")
    m2.metric("RMSE", f"{rmse:.2f} units")
    m3.metric("MAPE", f"{mape:.2f}%")
st.subheader("7-Day Demand Forecast")


forecast["date"] = pd.to_datetime(forecast["date"])

chart_data = forecast.pivot(
    index="date",
    columns="product",
    values="forecast_qty",
)

st.line_chart(chart_data)

st.subheader("Inventory Recommendations")
st.dataframe(inventory, use_container_width=True)

st.subheader("Daily Forecast Details")
st.dataframe(
    forecast.sort_values(["date", "product"]),
    use_container_width=True,
)

st.info(
    "Demonstration only: stock values and demand uncertainty "
    "are illustrative. Validate with real ERP data before use."
)
