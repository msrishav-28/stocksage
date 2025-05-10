import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# for Model
ticker = "TSLA"  # Example ticker, replace with your desired ticker
n_lags = 5
model_output_path = "best_model.pkl"
scaler_output_path = "scaler.pkl"

#features
def create_lag_features(df, n_lags=5):
    df_copy = df.copy()
    for i in range(1, n_lags + 1):
        df_copy[f'Close_Lag_{i}'] = df_copy['Close'].shift(i)
    df_copy['Volatility'] = df_copy['Close'].rolling(window=5).std()
    df_copy['Momentum'] = df_copy['Close'].pct_change(periods=5)
    df_copy['MA10_Ratio'] = df_copy['Close'] / df_copy['Close'].rolling(window=10).mean()
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    df_copy = df_copy.dropna()
    return df_copy

# fetch
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 3)  # Data for 3 years

# Fetch data from Yahoo Finance using yfinance
df = yf.download(ticker, start=start_date, end=end_date)

# Process the data with lag features
df_processed = create_lag_features(df, n_lags=n_lags)
features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
            'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
X = df_processed[features]
y = df_processed['Close']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

best_model = None
best_mse = float('inf')

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.2f}")
    if mse < best_mse:
        best_mse = mse
        best_model = model

# save using joblib
joblib.dump(best_model, model_output_path)
joblib.dump(scaler, scaler_output_path)

print(f"\nSaved model : {model_output_path}")
print(f"Saved scaler :{scaler_output_path}")
