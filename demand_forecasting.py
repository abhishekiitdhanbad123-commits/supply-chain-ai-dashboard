
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "sales_history.csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["product", "date"]).reset_index(drop=True)

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["lag_1"] = df.groupby("product")["sales_qty"].shift(1)
df["lag_7"] = df.groupby("product")["sales_qty"].shift(7)

df["rolling_mean_7"] = (
    df.groupby("product")["sales_qty"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

model_df = df.dropna(
    subset=["lag_1", "lag_7", "rolling_mean_7"]
).copy()

# --------------------------------------------------
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# --------------------------------------------------

cutoff_date = df["date"].min() + pd.Timedelta(days=292)

train = model_df[model_df["date"] < cutoff_date].copy()
test = model_df[model_df["date"] >= cutoff_date].copy()

features = [
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
]

X_train = train[features]
X_test = test[features]

y_train = train["sales_qty"]
y_test = test["sales_qty"]

print("Training rows:", len(train))
print("Testing rows:", len(test))

# --------------------------------------------------
# 4. TRAIN MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

test["prediction"] = np.maximum(
    0, model.predict(X_test)
)

# --------------------------------------------------
# 5. EVALUATE FORECAST ACCURACY
# --------------------------------------------------

actual = test["sales_qty"]
predicted = test["prediction"]

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

mape = (
    np.abs((actual - predicted) / actual).mean() * 100
)

print("\n--- RANDOM FOREST MODEL PERFORMANCE ---")
print(f"MAE:  {mae:.2f} units")
print(f"RMSE: {rmse:.2f} units")
print(f"MAPE: {mape:.2f}%")

# Save test predictions
test.to_csv(
    BASE_DIR / "output" / "test_predictions.csv",
    index=False,
)

print("\nTest predictions saved successfully!")
