# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import wikipedia
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os
import json

app = Flask(__name__)
# Enable CORS to allow requests from your React frontend
CORS(app) 

# Load the best model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Successfully loaded the prediction model")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# ============= ALL YOUR EXISTING FUNCTIONS (create_lag_features, fetch_wikipedia_summary, etc.) GO HERE =============
# (No changes are needed in your analysis functions)
def create_lag_features(df, n_lags=5):
    """Create lag features for the model - EXISTING FUNCTION"""
    df_copy = df.copy()
    for i in range(1, n_lags + 1):
        df_copy[f'Close_Lag_{i}'] = df_copy['Close'].shift(i)
    df_copy['Volatility'] = df_copy['Close'].rolling(window=5).std()
    df_copy['Momentum'] = df_copy['Close'].pct_change(periods=5)
    df_copy['MA10_Ratio'] = df_copy['Close'] / df_copy['Close'].rolling(window=10).mean()
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    df_copy = df_copy.dropna()
    return df_copy

def fetch_wikipedia_summary(company_name):
    """Get Wikipedia summary - EXISTING FUNCTION"""
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

def get_ticker_from_company_name(company_name):
    """Get ticker symbol from company name - EXISTING FUNCTION"""
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
            'nvidia': 'NVDA',
            'amd': 'AMD',
            'intel': 'INTC',
            'berkshire': 'BRK-B',
            'jpmorgan': 'JPM',
            'johnson': 'JNJ',
            'visa': 'V',
            'mastercard': 'MA',
            'disney': 'DIS',
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'nike': 'NKE',
            'adidas': 'ADDYY',
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

def get_competitors(company_name, description):
    """Identify competitors - EXISTING FUNCTION WITH ENHANCEMENTS"""
    # Enhanced industry mapping
    industry_map = {
        'tech': {
            'companies': ['microsoft', 'apple', 'google', 'meta', 'amazon', 'netflix', 'nvidia', 'intel', 'amd'],
            'keywords': ['technology', 'software', 'hardware', 'computer', 'internet', 'cloud']
        },
        'auto': {
            'companies': ['tesla', 'ford', 'gm', 'toyota', 'honda', 'volkswagen', 'bmw', 'mercedes'],
            'keywords': ['automotive', 'vehicle', 'car', 'electric', 'automobile']
        },
        'retail': {
            'companies': ['walmart', 'target', 'costco', 'amazon', 'home depot', 'lowes'],
            'keywords': ['retail', 'store', 'shopping', 'commerce', 'wholesale']
        },
        'finance': {
            'companies': ['jpmorgan', 'bank of america', 'wells fargo', 'goldman sachs', 'morgan stanley'],
            'keywords': ['bank', 'financial', 'investment', 'insurance', 'capital']
        },
        'pharma': {
            'companies': ['johnson & johnson', 'pfizer', 'merck', 'abbvie', 'eli lilly'],
            'keywords': ['pharmaceutical', 'drug', 'medicine', 'healthcare', 'biotech']
        }
    }
    
    # Determine industry
    company_lower = company_name.lower()
    desc_lower = description.lower() if description else ""
    detected_industry = None
    
    for industry, data in industry_map.items():
        if any(company in company_lower for company in data['companies']):
            detected_industry = industry
            break
        elif any(keyword in desc_lower for keyword in data['keywords']):
            detected_industry = industry
            break
    
    # Get competitors based on industry
    if detected_industry:
        all_companies = industry_map[detected_industry]['companies']
        competitors = [c for c in all_companies if c not in company_lower][:3]
    else:
        # Default to tech if industry not detected
        competitors = ['apple', 'microsoft', 'google'][:3]
    
    # Format competitors with price data
    formatted_competitors = []
    for comp in competitors:
        ticker = get_ticker_from_company_name(comp)
        if ticker:
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

# ============= NEW AI ANALYSIS FUNCTIONS =============

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    try:
        # Ensure we have enough data
        if len(df) < 50:
            return {
                'rsi': 50,
                'ma_20': df['Close'].mean() if not df.empty else 0,
                'ma_50': df['Close'].mean() if not df.empty else 0,
                'position_52w': 50,
                'volume_avg': df['Volume'].mean() if not df.empty else 0,
                'current_volume': df['Volume'].iloc[-1] if not df.empty else 0,
                'week_52_high': df['High'].max() if not df.empty else 0,
                'week_52_low': df['Low'].min() if not df.empty else 0,
                'bollinger_upper': df['Close'].mean() + 2 * df['Close'].std() if not df.empty else 0,
                'bollinger_lower': df['Close'].mean() - 2 * df['Close'].std() if not df.empty else 0
            }
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        ma_20 = df['Close'].rolling(window=20).mean()
        ma_50 = df['Close'].rolling(window=50).mean()
        
        # 52-week calculations
        week_52_high = df['High'].rolling(window=252, min_periods=1).max()
        week_52_low = df['Low'].rolling(window=252, min_periods=1).min()
        position_52w = ((df['Close'] - week_52_low) / (week_52_high - week_52_low) * 100)
        
        # Bollinger Bands
        bb_sma = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bollinger_upper = bb_sma + (bb_std * 2)
        bollinger_lower = bb_sma - (bb_std * 2)
        
        # Volume analysis
        volume_avg = df['Volume'].rolling(window=20).mean()
        
        return {
            'rsi': round(rsi.iloc[-1], 2) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50,
            'ma_20': round(ma_20.iloc[-1], 2) if not ma_20.empty and not pd.isna(ma_20.iloc[-1]) else 0,
            'ma_50': round(ma_50.iloc[-1], 2) if not ma_50.empty and not pd.isna(ma_50.iloc[-1]) else 0,
            'position_52w': round(position_52w.iloc[-1], 2) if not position_52w.empty and not pd.isna(position_52w.iloc[-1]) else 50,
            'volume_avg': round(volume_avg.iloc[-1], 0) if not volume_avg.empty and not pd.isna(volume_avg.iloc[-1]) else 0,
            'current_volume': round(df['Volume'].iloc[-1], 0) if not df.empty else 0,
            'week_52_high': round(week_52_high.iloc[-1], 2) if not week_52_high.empty else 0,
            'week_52_low': round(week_52_low.iloc[-1], 2) if not week_52_low.empty else 0,
            'bollinger_upper': round(bollinger_upper.iloc[-1], 2) if not bollinger_upper.empty and not pd.isna(bollinger_upper.iloc[-1]) else 0,
            'bollinger_lower': round(bollinger_lower.iloc[-1], 2) if not bollinger_lower.empty and not pd.isna(bollinger_lower.iloc[-1]) else 0
        }
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return {
            'rsi': 50,
            'ma_20': 0,
            'ma_50': 0,
            'position_52w': 50,
            'volume_avg': 0,
            'current_volume': 0,
            'week_52_high': 0,
            'week_52_low': 0,
            'bollinger_upper': 0,
            'bollinger_lower': 0
        }

def analyze_sentiment(ticker, current_price, technical_indicators, price_change_percent):
    """Analyze market sentiment with multiple factors"""
    sentiment_score = 0
    factors = []
    
    # Price change analysis
    if abs(price_change_percent) > 3:
        if price_change_percent > 0:
            sentiment_score += 2
            factors.append(f"Strong upward movement (+{price_change_percent:.1f}%)")
        else:
            sentiment_score -= 2
            factors.append(f"Strong downward movement ({price_change_percent:.1f}%)")
    
    # RSI analysis
    rsi = technical_indicators['rsi']
    if rsi < 30:
        sentiment_score -= 2
        factors.append("RSI indicates oversold conditions (potential bounce)")
    elif rsi > 70:
        sentiment_score += 2
        factors.append("RSI indicates overbought conditions (potential pullback)")
    elif 40 <= rsi <= 60:
        factors.append("RSI in neutral zone")
    
    # Moving average analysis
    if current_price > technical_indicators['ma_20'] > 0:
        sentiment_score += 1
        factors.append("Price above 20-day moving average (bullish)")
    elif technical_indicators['ma_20'] > 0:
        sentiment_score -= 1
        factors.append("Price below 20-day moving average (bearish)")
    
    if current_price > technical_indicators['ma_50'] > 0:
        sentiment_score += 1
        factors.append("Price above 50-day moving average (bullish)")
    elif technical_indicators['ma_50'] > 0:
        sentiment_score -= 1
        factors.append("Price below 50-day moving average (bearish)")
    
    # 52-week position
    position = technical_indicators['position_52w']
    if position < 20:
        sentiment_score -= 2
        factors.append("Near 52-week lows (oversold)")
    elif position > 80:
        sentiment_score += 2
        factors.append("Near 52-week highs (strong momentum)")
    elif 40 <= position <= 60:
        factors.append("Mid-range of 52-week prices")
    
    # Volume analysis
    if technical_indicators['current_volume'] > technical_indicators['volume_avg'] * 1.5:
        if price_change_percent > 0:
            sentiment_score += 1
            factors.append("High volume on upward move (bullish)")
        else:
            sentiment_score -= 1
            factors.append("High volume on downward move (bearish)")
    
    # Bollinger Bands analysis
    if current_price > technical_indicators['bollinger_upper'] and technical_indicators['bollinger_upper'] > 0:
        sentiment_score += 1
        factors.append("Breaking above Bollinger Band (strong momentum)")
    elif current_price < technical_indicators['bollinger_lower'] and technical_indicators['bollinger_lower'] > 0:
        sentiment_score -= 1
        factors.append("Breaking below Bollinger Band (weakness)")
    
    # Determine overall sentiment
    if sentiment_score >= 3:
        sentiment = "BULLISH"
        confidence = min(75 + sentiment_score * 3, 95)
    elif sentiment_score <= -3:
        sentiment = "BEARISH"
        confidence = min(75 + abs(sentiment_score) * 3, 95)
    else:
        sentiment = "NEUTRAL"
        confidence = 60 + abs(sentiment_score) * 5
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'factors': factors,
        'score': sentiment_score
    }

def assess_risk_level(volatility, position_52w, rsi, volume_ratio, price_change_percent):
    """Comprehensive risk assessment"""
    risk_score = 0
    risk_factors = []
    
    # Volatility risk
    daily_volatility = volatility * 100  # Convert to percentage
    if daily_volatility > 3:
        risk_score += 3
        risk_factors.append(f"High volatility ({daily_volatility:.1f}% daily)")
    elif daily_volatility > 2:
        risk_score += 2
        risk_factors.append(f"Moderate volatility ({daily_volatility:.1f}% daily)")
    elif daily_volatility > 1:
        risk_score += 1
        risk_factors.append(f"Normal volatility ({daily_volatility:.1f}% daily)")
    else:
        risk_factors.append(f"Low volatility ({daily_volatility:.1f}% daily)")
    
    # Position risk
    if position_52w < 15:
        risk_score += 2
        risk_factors.append("Very close to 52-week lows")
    elif position_52w > 85:
        risk_score += 2
        risk_factors.append("Very close to 52-week highs")
    elif position_52w < 30 or position_52w > 70:
        risk_score += 1
        risk_factors.append("Near yearly extremes")
    
    # RSI extremes
    if rsi < 25:
        risk_score += 2
        risk_factors.append("Extremely oversold (RSI < 25)")
    elif rsi > 75:
        risk_score += 2
        risk_factors.append("Extremely overbought (RSI > 75)")
    elif rsi < 35 or rsi > 65:
        risk_score += 1
        risk_factors.append("RSI showing extreme conditions")
    
    # Volume spike risk
    if volume_ratio > 3:
        risk_score += 2
        risk_factors.append("Unusual volume spike (3x average)")
    elif volume_ratio > 2:
        risk_score += 1
        risk_factors.append("High volume activity")
    
    # Price movement risk
    if abs(price_change_percent) > 5:
        risk_score += 2
        risk_factors.append(f"Large price movement ({price_change_percent:.1f}%)")
    elif abs(price_change_percent) > 3:
        risk_score += 1
        risk_factors.append(f"Significant price movement ({price_change_percent:.1f}%)")
    
    # Determine risk level
    if risk_score >= 6:
        level = "HIGH"
    elif risk_score >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return {
        'level': level,
        'score': risk_score,
        'factors': risk_factors
    }

def generate_eli5_explanation(ticker, current_price, change_percent, sentiment, risk_level, technical_indicators):
    """Generate simple, engaging explanation for beginners"""
    
    # Price movement explanation
    if abs(change_percent) < 0.5:
        movement = "stayed pretty much the same"
        reason = "buyers and sellers are balanced"
    elif change_percent > 3:
        movement = f"jumped up {abs(change_percent):.1f}%"
        reason = "lots of people are excited and buying"
    elif change_percent > 0:
        movement = f"went up {abs(change_percent):.1f}%"
        reason = "more people wanted to buy than sell"
    elif change_percent < -3:
        movement = f"dropped {abs(change_percent):.1f}%"
        reason = "lots of people are worried and selling"
    else:
        movement = f"went down {abs(change_percent):.1f}%"
        reason = "more people wanted to sell than buy"
    
    # Position context
    position = technical_indicators['position_52w']
    if position < 30:
        position_context = "It's currently on sale compared to this year's prices - like finding your favorite game at a discount!"
    elif position > 70:
        position_context = "It's near its highest price this year - like a hot toy everyone wants!"
    else:
        position_context = "It's in the middle of its yearly price range - not too high, not too low."
    
    # Sentiment explanation
    sentiment_explain = {
        "BULLISH": "Most investors think it will go up 📈 - like betting your favorite team will win!",
        "BEARISH": "Most investors think it will go down 📉 - like thinking it might rain tomorrow.",
        "NEUTRAL": "Investors aren't sure which way it'll go ↔️ - like a coin flip!"
    }
    
    # Risk explanation
    risk_explain = {
        "HIGH": "This stock is like a roller coaster 🎢 - big ups and downs! You could make or lose money quickly.",
        "MEDIUM": "This stock has normal ups and downs - like waves at the beach 🌊.",
        "LOW": "This stock is pretty calm - like a gentle boat ride 🚤."
    }
    
    explanation = f"""
🎯 **{ticker} Simple Explanation**

📊 **What happened today?**
The stock price {movement} to ${current_price:.2f} because {reason}.

📍 **Where is it now?**
{position_context}

🤔 **What do investors think?**
{sentiment_explain.get(sentiment['sentiment'], "Investors are watching carefully.")}

⚡ **How risky is it?**
{risk_explain.get(risk_level['level'], "Check with a financial advisor.")}

💡 **Think of it this way:**
Imagine {ticker} is like a store in the mall. {f"Today, the store got {abs(change_percent):.1f}% more popular!" if change_percent > 0 else f"Today, the store got {abs(change_percent):.1f}% less popular."} The stock market is just people voting with their money on which companies they think will do well!

Remember: Stock prices change based on how people feel about the company's future - just like how toy prices go up before Christmas when everyone wants them! 🎄
    """
    
    return explanation.strip()

def generate_recommendation(sentiment, risk_level, technical_indicators, predicted_price, current_price):
    """Generate actionable investment recommendation"""
    
    # Calculate expected return
    expected_return = ((predicted_price - current_price) / current_price) * 100
    
    # Base recommendation on multiple factors
    if risk_level['level'] == "HIGH":
        if sentiment['sentiment'] == "BULLISH" and technical_indicators['rsi'] < 30:
            return {
                'action': 'WATCH',
                'summary': 'High volatility but potentially oversold. Wait for stability.',
                'details': 'The stock shows high risk but RSI suggests it may be oversold. Consider waiting for confirmation of reversal before entry.'
            }
        else:
            return {
                'action': 'AVOID',
                'summary': 'High risk environment. Not suitable for conservative investors.',
                'details': 'Extreme volatility makes this unsuitable for risk-averse investors. Consider more stable alternatives.'
            }
    
    if sentiment['sentiment'] == "BULLISH":
        if technical_indicators['rsi'] < 65 and expected_return > 2:
            return {
                'action': 'BUY',
                'summary': f'Positive outlook with {expected_return:.1f}% upside potential.',
                'details': f'Technical indicators and AI analysis suggest bullish momentum. RSI at {technical_indicators["rsi"]:.0f} leaves room for growth.'
            }
        elif technical_indicators['rsi'] > 70:
            return {
                'action': 'TAKE PROFITS',
                'summary': 'Overbought conditions suggest limited upside.',
                'details': 'While sentiment remains positive, RSI above 70 indicates the stock may be due for a pullback.'
            }
        else:
            return {
                'action': 'HOLD',
                'summary': 'Positive sentiment but wait for better entry.',
                'details': 'The outlook is positive but current levels may not offer the best risk/reward ratio.'
            }
    
    elif sentiment['sentiment'] == "BEARISH":
        if technical_indicators['position_52w'] < 30 and technical_indicators['rsi'] < 35:
            return {
                'action': 'POTENTIAL BUY',
                'summary': 'Oversold conditions may present opportunity.',
                'details': 'While sentiment is negative, technical indicators suggest the selling may be overdone. Consider small position for contrarian play.'
            }
        else:
            return {
                'action': 'SELL/AVOID',
                'summary': 'Negative sentiment suggests further downside.',
                'details': f'Multiple indicators point to continued weakness. Consider protecting capital by avoiding or reducing exposure.'
            }
    
    else:  # NEUTRAL
        if abs(expected_return) < 1:
            return {
                'action': 'HOLD',
                'summary': 'No clear direction. Wait for trend to develop.',
                'details': 'Market is undecided. Best to wait for clearer signals before making investment decisions.'
            }
        elif expected_return > 0:
            return {
                'action': 'WATCH',
                'summary': 'Slight positive bias but needs confirmation.',
                'details': 'Small predicted upside but neutral sentiment suggests caution. Add to watchlist and wait for bullish confirmation.'
            }
        else:
            return {
                'action': 'WATCH',
                'summary': 'Slight negative bias. Monitor closely.',
                'details': 'Small predicted downside with neutral sentiment. Avoid new positions until clearer trend emerges.'
            }

# ============= ROUTES =============

# REMOVED the home() route that served index.html

@app.route('/analyze', methods=['POST'])
def analyze():
    """Enhanced analyze endpoint with AI features"""
    data = request.json
    company_name = data.get('company_name', '')
    ticker_input = data.get('ticker', '')
    
    # Get ticker
    if not ticker_input and company_name:
        ticker = get_ticker_from_company_name(company_name)
    else:
        ticker = ticker_input.upper() if ticker_input else None
    
    if not ticker:
        return jsonify({
            'success': False,
            'error': 'Could not determine ticker symbol. Please provide a valid ticker.'
        })
    
    try:
        # Fetch comprehensive data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': f"No data found for ticker {ticker}"
            })
        
        # Get ticker info
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        # Get company name if not provided
        if not company_name:
            company_name = stock_info.get('longName', ticker)
        
        # Calculate current metrics
        current_price = round(df['Close'].iloc[-1], 2)
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - previous_close
        change_percent = (price_change / previous_close) * 100 if previous_close != 0 else 0
        
        # Calculate technical indicators
        print("Calculating technical indicators...")
        technical_indicators = calculate_technical_indicators(df)
        
        # Calculate volatility
        volatility = df['Close'].pct_change().std()
        
        # Volume ratio
        volume_ratio = technical_indicators['current_volume'] / technical_indicators['volume_avg'] if technical_indicators['volume_avg'] > 0 else 1
        
        # Analyze sentiment
        print("Analyzing market sentiment...")
        sentiment_analysis = analyze_sentiment(ticker, current_price, technical_indicators, change_percent)
        
        # Assess risk
        print("Assessing risk level...")
        risk_assessment = assess_risk_level(volatility, technical_indicators['position_52w'], 
                                          technical_indicators['rsi'], volume_ratio, change_percent)
        
        # Generate ELI5 explanation
        print("Generating explanations...")
        eli5_explanation = generate_eli5_explanation(
            ticker, current_price, change_percent, 
            sentiment_analysis, risk_assessment, technical_indicators
        )
        
        # Price prediction using your model
        predicted_price = current_price  # Default
        confidence_score = 50  # Default confidence
        
        if model is not None and scaler is not None:
            try:
                # Use last 60 days for prediction
                df_recent = df.tail(60).copy()
                df_processed = create_lag_features(df_recent, n_lags=5)
                features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                           'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
                
                if len(df_processed) > 0:
                    X = df_processed[features]
                    X_scaled = scaler.transform(X.tail(1))
                    predicted_price = round(float(model.predict(X_scaled)[0]), 2)
                    
                    # Calculate prediction confidence based on model features
                    prediction_change = abs((predicted_price - current_price) / current_price)
                    if prediction_change < 0.02:  # Less than 2% change
                        confidence_score = 75
                    elif prediction_change < 0.05:  # Less than 5% change
                        confidence_score = 65
                    else:
                        confidence_score = 50
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_price = round(current_price * 1.01, 2)
        
        # Generate recommendation
        recommendation = generate_recommendation(
            sentiment_analysis, 
            risk_assessment, 
            technical_indicators, 
            predicted_price, 
            current_price
        )
        
        # Get company description
        description = fetch_wikipedia_summary(company_name)
        
        # Get competitors
        print("Finding competitors...")
        competitors = get_competitors(company_name, description)
        
        # Get recent data for charts
        recent_df = df.tail(60)
        
        # Prepare comprehensive response
        response_data = {
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'description': description,
            
            # Price data
            'current_price': current_price,
            'price_change': round(price_change, 2),
            'change_percent': round(change_percent, 2),
            'predicted_price': predicted_price,
            'prediction_confidence': confidence_score,
            
            # Chart data
            'stock_prices': recent_df['Close'].tolist(),
            'time_labels': recent_df.index.strftime('%Y-%m-%d').tolist(),
            'volumes': recent_df['Volume'].tolist(),
            
            # Competitors
            'top_competitors': competitors,
            
            # AI Analysis
            'ai_analysis': {
                'sentiment': sentiment_analysis['sentiment'],
                'sentiment_confidence': sentiment_analysis['confidence'],
                'sentiment_factors': sentiment_analysis['factors'],
                'sentiment_score': sentiment_analysis['score'],
                
                'risk_level': risk_assessment['level'],
                'risk_score': risk_assessment['score'],
                'risk_factors': risk_assessment['factors'],
                
                'technical_indicators': technical_indicators,
                
                'eli5_explanation': eli5_explanation,
                
                'recommendation': recommendation,
                
                # Additional metrics
                'market_cap': stock_info.get('marketCap', 0),
                'pe_ratio': stock_info.get('trailingPE', 0),
                'dividend_yield': stock_info.get('dividendYield', 0) * 100 if stock_info.get('dividendYield') else 0,
                'week_52_high': technical_indicators['week_52_high'],
                'week_52_low': technical_indicators['week_52_low']
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        })

# This route can be removed if you no longer need it.
@app.route('/predict', methods=['POST'])
def predict():
    """Original predict endpoint for backward compatibility"""
    try:
        # Get the user inputs
        ticker = request.form['ticker']
        
        # Fetch data from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

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

        # Convert the prediction from ndarray to float and round to 2 decimal places
        predicted_price = round(float(predicted_price[0]), 2)

        return jsonify({'predicted_price': predicted_price})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while making the prediction."})


if __name__ == '__main__':
    app.run(debug=True, port=5000)