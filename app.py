from flask import Flask, render_template, request, jsonify
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import wikipedia
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the best model and scaler - your existing Linear Regression model
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Successfully loaded the prediction model")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Function to create lag features - keeping your original implementation
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

# Function to get Wikipedia summary
def fetch_wikipedia_summary(company_name):
    try:
        search_results = wikipedia.search(company_name)
        if search_results:
            page_title = search_results[0]
            summary = wikipedia.summary(page_title, sentences=3)
            return summary
    except Exception as e:
        print(f"Error fetching Wikipedia summary: {e}")
        return f"No description available for {company_name}."
    
    return f"No description available for {company_name}."

# Function to get ticker symbol from company name
def get_ticker_from_company_name(company_name):
    try:
        # Simple mapping for common companies
        common_tickers = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'meta': 'META',
            'facebook': 'META',
            'tesla': 'TSLA',
            'netflix': 'NFLX',
            'walmart': 'WMT',
        }
        
        # Check our simple mapping first
        for name, ticker in common_tickers.items():
            if name in company_name.lower():
                return ticker
                
        # If not found, try yfinance search
        ticker = yf.Ticker(company_name)
        if hasattr(ticker, 'info') and 'symbol' in ticker.info:
            return ticker.info['symbol']
            
        return None
    except:
        return None

# Function to identify competitors (simplified version)
def get_competitors(company_name, description):
    # Simplified industry mapping
    tech_companies = ['microsoft', 'apple', 'google', 'meta', 'amazon', 'netflix']
    auto_companies = ['ford', 'gm', 'tesla', 'toyota', 'honda']
    retail_companies = ['walmart', 'target', 'costco', 'amazon']
    
    # Determine industry based on company name
    company_lower = company_name.lower()
    competitors = []
    
    # Check which industry the company belongs to
    if any(company in company_lower for company in tech_companies):
        competitors = [c for c in tech_companies if c not in company_lower]
    elif any(company in company_lower for company in auto_companies):
        competitors = [c for c in auto_companies if c not in company_lower]
    elif any(company in company_lower for company in retail_companies):
        competitors = [c for c in retail_companies if c not in company_lower]
    else:
        # Default to tech companies if we can't identify the industry
        competitors = tech_companies
    
    # Get the top 3 competitors
    top_competitors = competitors[:3]
    
    # Format the competitors
    formatted_competitors = []
    for comp in top_competitors:
        ticker = get_ticker_from_company_name(comp)
        if ticker:
            # Get competitor price data
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period="3mo")
                current_price = round(history['Close'].iloc[-1], 2) if not history.empty else 0
                
                formatted_competitors.append({
                    "name": comp.title(),
                    "ticker": ticker,
                    "stock_price": current_price,
                    "stock_prices": history['Close'].tolist() if not history.empty else [],
                    "time_labels": history.index.strftime('%Y-%m-%d').tolist() if not history.empty else []
                })
            except Exception as e:
                print(f"Error getting data for {comp}: {e}")
    
    return formatted_competitors

# Route to render the enhanced HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Combined analyze endpoint - handles both company analysis and price prediction
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    company_name = data.get('company_name', '')
    ticker_input = data.get('ticker', '')
    source = data.get('source', 'yfinance')
    
    # If ticker is not provided, try to get it from company name
    if not ticker_input and company_name:
        ticker = get_ticker_from_company_name(company_name)
    else:
        ticker = ticker_input
    
    if not ticker:
        return jsonify({
            'success': False,
            'error': 'Could not determine ticker symbol. Please provide a valid ticker.'
        })
    
    # Get company description
    if company_name:
        description = fetch_wikipedia_summary(company_name)
    else:
        try:
            # Try to get company name from ticker
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            description = fetch_wikipedia_summary(company_name)
        except:
            description = f"Information for {ticker}"
            company_name = ticker
    
    try:
        # Fetch stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': f"No data found for ticker {ticker}"
            })
        
        # Get current price and historical data
        current_price = round(df['Close'].iloc[-1], 2)
        stock_prices = df['Close'].tolist()
        time_labels = df.index.strftime('%Y-%m-%d').tolist()
        
        # Use your Linear Regression model to predict next day's price
        if model is not None and scaler is not None:
            try:
                # Process data with lag features
                df_processed = create_lag_features(df, n_lags=5)
                features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                           'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
                
                # Check if we have enough data
                if len(df_processed) > 0:
                    X = df_processed[features]
                    X_scaled = scaler.transform(X.tail(1))
                    predicted_price = round(float(model.predict(X_scaled)[0]), 2)
                else:
                    # Fallback prediction if not enough data
                    predicted_price = round(current_price * 1.01, 2)  # Simple 1% increase
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_price = round(current_price * 1.01, 2)  # Simple 1% increase
        else:
            # Fallback if model not loaded
            predicted_price = round(current_price * 1.01, 2)  # Simple 1% increase
        
        # Get competitors
        competitors = get_competitors(company_name, description)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'description': description,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'stock_prices': stock_prices,
            'time_labels': time_labels,
            'top_competitors': competitors
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        })

# Original predict endpoint for backward compatibility
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
