# StockMind Pro: Stock Price Prediction & Analysis

This application predicts the next day's closing price of stocks using a machine learning model and provides in-depth AI-powered analysis. It features a Flask API backend and a modern React frontend.

## Features

- **AI-Powered Analysis**: Get insights on market sentiment, risk level, and an actionable AI recommendation.
- **Price Prediction**: Predicts the next day's closing price using an ML model.
- **Company Information**: Fetches company summaries from Wikipedia.
- **Competitor Analysis**: Identifies and compares key industry competitors.
- **Interactive Charts**: Visualizes historical price data, volume, and competitor trends using Chart.js.
- **Modern UI**: A responsive and beautiful interface built with React and Material-UI.

## Technologies Used

- **Backend**: Flask, YFinance, Scikit-learn, Pandas, Joblib
- **Frontend**: React, Axios, Material-UI (MUI), Chart.js
- **ML Model**: The prediction model uses historical price lags, volatility, momentum, and volume data.

## Installation

Follow these steps to get your local development environment set up.

### 1. Backend Setup

```bash
# Clone the repository
git clone [https://github.com/yourusername/stockmind-pro.git](https://github.com/yourusername/stockmind-pro.git)
cd stockmind-pro

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install backend dependencies
pip install -r requirements.txt

# Make sure you have the trained model (best_model.pkl) and scaler (scaler.pkl).
# If not, you can train a new model by running:
python model.py
