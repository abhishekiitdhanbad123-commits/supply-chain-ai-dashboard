from pathlib import Path
import numpy as np
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PLOT_DIR = BASE_DIR / "plots"

for folder in [DATA_DIR, OUTPUT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Generate sample bakery sales data
np.random.seed(42)

dates = pd.date_range(
    start="2025-10-01",
    periods=365,
    freq="D"
)

products = {
    "Croissant": 120,
    "Brownie": 80,
    "Chocolate_Cake": 45
}

records = []

for product, base_demand in products.items():
    for date in dates:

        weekend_factor = (
            1.25 if date.dayofweek >= 5 else 1.0
        )

        sales = int(max(
            0,
            np.random.normal(
                base_demand * weekend_factor,
                base_demand * 0.12
            )
        ))

        records.append({
            "date": date,
            "product": product,
            "city": "Hyderabad",
            "sales_qty": sales
        })

df = pd.DataFrame(records)

# Save dataset
file_path = DATA_DIR / "sales_history.csv"
df.to_csv(file_path, index=False)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print(df.head())
print("Saved at:", file_path)

# ==========================================
# STEP 2: DATA ANALYSIS
# ==========================================

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Check duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# Product-wise sales analysis
product_summary = (
    df.groupby("product")["sales_qty"]
      .agg(["sum", "mean", "min", "max"])
      .round(2)
)

print("\n--- PRODUCT-WISE SALES ---")
print(product_summary)

# Weekday analysis
df["date"] = pd.to_datetime(df["date"])
df["day_name"] = df["date"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_summary = (
    df.groupby("day_name")["sales_qty"]
      .mean()
      .reindex(weekday_order)
      .round(2)
)

print("\n--- AVERAGE DEMAND BY WEEKDAY ---")
print(weekday_summary)

# Save reports
product_summary.to_csv(
    OUTPUT_DIR / "product_summary.csv"
)

weekday_summary.to_csv(
    OUTPUT_DIR / "weekday_summary.csv"
)

print("\nAnalysis completed successfully!")

weekday_summary = (
    df.groupby(["product", "day_name"])["sales_qty"]
      .mean()
      .unstack()
      .reindex(columns=weekday_order)
      .round(2)
)

print("\n--- PRODUCT-WISE AVERAGE DEMAND BY WEEKDAY ---")
print(weekday_summary)
