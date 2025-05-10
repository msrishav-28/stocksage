from flask import Flask, render_template, request, jsonify
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the best model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to create lag features
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

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user inputs
        ticker = request.form['ticker']
        source = request.form['source']
        
        print(f"Received ticker: {ticker} from {source}")

        # Fetch stock data from Yahoo Finance only
        if source != 'yfinance':
            return jsonify({'predicted_price': "Currently, only Yahoo Finance (yfinance) is supported."})
        
        # Fetch data from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            return jsonify({"error": "No data found for the given ticker"})

        print(f"Fetched data for {ticker} from {source}")

        # Process the data with lag features
        df_processed = create_lag_features(df, n_lags=5)
        features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                    'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
        X = df_processed[features]
        
        # Scale the data
        X_scaled = scaler.transform(X.tail(1))  # Use only the latest data (last row)

        # Make prediction
        predicted_price = model.predict(X_scaled)

        # Convert the prediction from ndarray to float and round to 2 decimal places
        predicted_price = round(float(predicted_price[0]), 2)

        print(f"Predicted Price: {predicted_price}")

        return jsonify({'predicted_price': predicted_price})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while making the prediction."})

if __name__ == '__main__':
    app.run(debug=True)
