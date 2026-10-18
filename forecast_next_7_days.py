
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_history.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load historical sales
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["product", "date"]).reset_index(drop=True)

# 2. Create historical features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["lag_1"] = df.groupby("product")["sales_qty"].shift(1)
df["lag_7"] = df.groupby("product")["sales_qty"].shift(7)

df["rolling_mean_7"] = (
    df.groupby("product")["sales_qty"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

features = [
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
]

model_df = df.dropna(subset=features).copy()

# 3. Train on all available historical data
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(model_df[features], model_df["sales_qty"])

# 4. Forecast the next 7 days recursively
last_date = df["date"].max()
forecast_records = []

for product in df["product"].unique():

    history = (
        df.loc[df["product"] == product, "sales_qty"]
          .astype(float)
          .tolist()
    )

    for step in range(1, 8):

        future_date = last_date + pd.Timedelta(days=step)

        X_future = pd.DataFrame([{
            "day_of_week": future_date.dayofweek,
            "month": future_date.month,
            "is_weekend": int(future_date.dayofweek >= 5),
            "lag_1": history[-1],
            "lag_7": history[-7],
            "rolling_mean_7": np.mean(history[-7:]),
        }])

        prediction = max(
            0,
            float(model.predict(X_future)[0])
        )

        forecast_records.append({
            "date": future_date,
            "product": product,
            "forecast_qty": round(prediction, 2),
        })

        # Use this prediction to generate the next day
        history.append(prediction)

# 5. Save forecast
forecast_df = pd.DataFrame(forecast_records)

forecast_df.to_csv(
    OUTPUT_DIR / "forecast_7_days.csv",
    index=False,
)

print("\n--- NEXT 7 DAYS DEMAND FORECAST ---")
print(forecast_df.to_string(index=False))

print("\n--- TOTAL FORECAST BY PRODUCT ---")
print(
    forecast_df.groupby("product")["forecast_qty"]
               .sum()
               .round(2)
)

print("\nSaved to:", OUTPUT_DIR / "forecast_7_days.csv")
