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

# For LLM-based competitor analysis (simulated in this example)
# You would replace this with real Gemini API integration later
def simulate_competitor_analysis(company_description):
    """Simulate competitor analysis for demonstration purposes.
    In a real implementation, this would use Gemini LLM."""
    
    # Simplified simulation based on keywords in description
    tech_keywords = ['tech', 'software', 'digital', 'computer', 'mobile', 'app']
    auto_keywords = ['car', 'auto', 'vehicle', 'motor']
    retail_keywords = ['retail', 'store', 'shop', 'consumer', 'product']
    
    desc_lower = company_description.lower()
    
    competitors = []
    if any(keyword in desc_lower for keyword in tech_keywords):
        competitors.append({
            "name": "Technology",
            "competitors": ["Microsoft", "Google", "Apple", "Meta"]
        })
    
    if any(keyword in desc_lower for keyword in auto_keywords):
        competitors.append({
            "name": "Automotive",
            "competitors": ["Ford", "Toyota", "General Motors", "Tesla"]
        })
        
    if any(keyword in desc_lower for keyword in retail_keywords):
        competitors.append({
            "name": "Retail",
            "competitors": ["Walmart", "Amazon", "Target", "Best Buy"]
        })
    
    # Default category if none matched
    if not competitors:
        competitors.append({
            "name": "General",
            "competitors": ["Amazon", "Microsoft", "Apple", "Google"]
        })
    
    return competitors

# Function to get stock ticker from company name (simplified version)
def get_ticker_from_company_name(company_name):
    """Simplified ticker lookup based on common company names.
    In a real implementation, this would use Alpha Vantage API."""
    
    # Simplified mapping for demonstration
    company_ticker_map = {
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
        'target': 'TGT',
        'ford': 'F',
        'general motors': 'GM',
        'toyota': 'TM',
    }
    
    # Check for the company name in our map (case insensitive)
    for name, ticker in company_ticker_map.items():
        if name in company_name.lower():
            return ticker
    
    # If no match, return None
    return None

# Function to get Wikipedia summary for a company
def fetch_wikipedia_summary(company_name):
    try:
        search_results = wikipedia.search(company_name)
        if search_results:
            page_title = search_results[0]
            summary = wikipedia.summary(page_title, sentences=3)
            return summary
    except Exception as e:
        return f"No description available. Error: {str(e)}"
    
    return "No description available."

# Function to create top competitors with mock data
def get_top_competitors(competitor_names, main_ticker):
    top_competitors = []
    
    for i, name in enumerate(competitor_names[:3]):  # Get top 3
        ticker = get_ticker_from_company_name(name)
        if not ticker:
            continue
            
        # Try to get real stock data
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1mo")
            
            if not history.empty:
                latest_price = round(history['Close'].iloc[-1], 2)
                stock_prices = history['Close'].tolist()
                time_labels = history.index.strftime('%Y-%m-%d').tolist()
                
                top_competitors.append({
                    "name": name,
                    "ticker": ticker,
                    "stock_price": latest_price,
                    "stock_prices": stock_prices,
                    "time_labels": time_labels
                })
        except:
            # If we can't get real data, use fake data
            fake_price = round(100 + (i * 10) + np.random.random() * 20, 2)
            top_competitors.append({
                "name": name,
                "ticker": ticker or f"{name[:4].upper()}",
                "stock_price": fake_price
            })
    
    return top_competitors

# Initialize Flask app
app = Flask(__name__)

# Try to load the model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_loaded = True
except:
    model_loaded = False
    print("Warning: Could not load ML model. Prediction functionality will be limited.")

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

# New comprehensive analysis endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    company_name = data.get('company_name', '')
    ticker = data.get('ticker', '')
    source = data.get('source', 'yfinance')
    
    # If no ticker provided, try to get it from company name
    if not ticker and company_name:
        ticker = get_ticker_from_company_name(company_name)
        
    if not ticker:
        return jsonify({
            'success': False,
            'error': 'Could not determine ticker symbol. Please provide a valid ticker.'
        })
    
    # Get company description from Wikipedia
    if company_name:
        description = fetch_wikipedia_summary(company_name)
    else:
        # If only ticker provided, try to get company info from ticker
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            description = fetch_wikipedia_summary(company_name)
        except:
            description = f"No description available for {ticker}"
    
    # Fetch stock data
    try:
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
        
        # Predict next day's price if model is loaded
        predicted_price = None
        if model_loaded:
            try:
                df_processed = create_lag_features(df, n_lags=5)
                features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                            'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
                X = df_processed[features]
                X_scaled = scaler.transform(X.tail(1))
                predicted_price = round(float(model.predict(X_scaled)[0]), 2)
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_price = round(current_price * (1 + (np.random.random() - 0.5) * 0.05), 2)
        else:
            # Fallback if model not loaded
            predicted_price = round(current_price * (1 + (np.random.random() - 0.5) * 0.05), 2)
        
        # Get competitor analysis
        competitors = simulate_competitor_analysis(description)
        
        # Get all competitor names
        all_competitors = []
        for sector in competitors:
            all_competitors.extend(sector['competitors'])
        
        # Get top competitors with stock data
        top_competitors = get_top_competitors(all_competitors, ticker)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'description': description,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'stock_prices': stock_prices,
            'time_labels': time_labels,
            'competitors': competitors,
            'top_competitors': top_competitors
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        })

# Route to handle predictions (keep for backward compatibility)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user inputs
        ticker = request.form['ticker']
        source = request.form['source']
        
        # Fetch data from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            return jsonify({"error": "No data found for the given ticker"})

        # Process the data with lag features
        df_processed = create_lag_features(df, n_lags=5)
        features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                    'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
        X = df_processed[features]
        
        # Scale the data
        X_scaled = scaler.transform(X.tail(1))

        # Make prediction
        predicted_price = model.predict(X_scaled)
        predicted_price = round(float(predicted_price[0]), 2)

        return jsonify({'predicted_price': predicted_price})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while making the prediction."})

if __name__ == '__main__':
    app.run(debug=True)
