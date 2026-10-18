
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

forecast = pd.read_csv(
    BASE_DIR / "output" / "forecast_7_days.csv"
)

# Illustrative assumptions for this exercise
lead_time_days = 3
review_period_days = 7
z_score = 1.65

# Simulated available stock; replace with ERP stock later
stock_days = {
    "Croissant": 500,
    "Brownie": 350,
    "Chocolate_Cake": 180,
}

# Calculate average daily demand
summary = (
    forecast.groupby("product")["forecast_qty"]
    .agg(["sum", "mean"])
    .reset_index()
)

summary = summary.rename(columns={
    "sum": "forecast_7_days",
    "mean": "avg_daily_demand",
})

# Illustrative uncertainty assumption for safety stock.
# Replace with validated forecast-error estimates later.
summary["daily_demand_std"] = (
    summary["avg_daily_demand"] * 0.15
)

summary["lead_time_days"] = lead_time_days

summary["safety_stock"] = (
    z_score
    * summary["daily_demand_std"]
    * np.sqrt(lead_time_days)
)

summary["reorder_point"] = (
    summary["avg_daily_demand"] * lead_time_days
    + summary["safety_stock"]
)

summary["target_stock"] = (
    summary["avg_daily_demand"]
    * (lead_time_days + review_period_days)
    + summary["safety_stock"]
)

summary["available_stock"] = (
    summary["product"].map(stock_days)
)

summary["suggested_replenishment"] = np.ceil(
    (
        summary["target_stock"]
        - summary["available_stock"]
    ).clip(lower=0)
)

summary["stock_status"] = np.where(
    summary["available_stock"] <= summary["reorder_point"],
    "REPLENISHMENT REVIEW",
    "ABOVE REORDER POINT",
)

summary = summary.round(2)

output_path = (
    BASE_DIR / "output" / "inventory_recommendations.csv"
)

summary.to_csv(output_path, index=False)

print("\n--- INVENTORY OPTIMIZATION RESULTS ---")
print(summary.to_string(index=False))
print("\nSaved to:", output_path)
