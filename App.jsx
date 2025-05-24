import React, { useState } from 'react';
import './App.css';
import StockForm from './components/StockForm';
import PriceDisplay from './components/PriceDisplay';
import AIAnalysis from './components/AIAnalysis';
import TechnicalAnalysis from './components/TechnicalAnalysis';
import CompetitorAnalysis from './components/CompetitorAnalysis';
import { analyzeStock } from './services/api';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('prediction');

  const handleAnalyze = async (companyName, ticker) => {
    setLoading(true);
    try {
      const data = await analyzeStock(companyName, ticker);
      if (data.success) {
        setResults(data);
      } else {
        alert(data.error || 'Error analyzing stock');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>StockMind Pro</h1>
        <p>AI-Powered Stock Analysis & Prediction</p>
      </header>
      
      <div className="container">
        <StockForm onSubmit={handleAnalyze} loading={loading} />
        
        {results && (
          <>
            <div className="company-overview">
              <div className="overview-section">
                <h3>{results.company_name} ({results.ticker})</h3>
                <p className="company-description">{results.description}</p>
              </div>
              <PriceDisplay 
                currentPrice={results.current_price}
                priceChange={results.price_change}
                changePercent={results.change_percent}
                predictedPrice={results.predicted_price}
                predictionConfidence={results.prediction_confidence}
              />
            </div>

            <div className="tabs">
              <button 
                className={`tab ${activeTab === 'prediction' ? 'active' : ''}`}
                onClick={() => setActiveTab('prediction')}
              >
                📈 Prediction
              </button>
              <button 
                className={`tab ${activeTab === 'ai-analysis' ? 'active' : ''}`}
                onClick={() => setActiveTab('ai-analysis')}
              >
                🤖 AI Analysis
              </button>
              <button 
                className={`tab ${activeTab === 'technical' ? 'active' : ''}`}
                onClick={() => setActiveTab('technical')}
              >
                📊 Technical
              </button>
              <button 
                className={`tab ${activeTab === 'competitors' ? 'active' : ''}`}
                onClick={() => setActiveTab('competitors')}
              >
                🏢 Competitors
              </button>
            </div>

            <div className="tab-content">
              {activeTab === 'prediction' && (
                <div className="prediction-tab">
                  <div className="prediction-card">
                    <h3>Next Day Price Prediction</h3>
                    <div className="prediction-value">${results.predicted_price}</div>
                    <div className="confidence-meter">
                      <div 
                        className="confidence-fill" 
                        style={{ width: `${results.prediction_confidence}%` }}
                      />
                    </div>
                    <p>Confidence: {results.prediction_confidence}%</p>
                  </div>
                  <div className="chart-card">
                    <h3>Historical Trend</h3>
                    <PriceChart 
                      labels={results.time_labels} 
                      prices={results.stock_prices}
                    />
                  </div>
                </div>
              )}

              {activeTab === 'ai-analysis' && (
                <AIAnalysis analysis={results.ai_analysis} />
              )}

              {activeTab === 'technical' && (
                <TechnicalAnalysis 
                  indicators={results.ai_analysis.technical_indicators}
                  volumes={results.volumes}
                  labels={results.time_labels}
                />
              )}

              {activeTab === 'competitors' && (
                <CompetitorAnalysis competitors={results.top_competitors} />
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// Simple price chart component
function PriceChart({ labels, prices }) {
  // In a real app, you'd use a charting library like Chart.js or Recharts
  // This is a simplified representation
  return (
    <div className="price-chart">
      <div className="chart-placeholder">
        <p>Price trend for last {labels.length} days</p>
        <p>Current: ${prices[prices.length - 1]}</p>
        <p>Min: ${Math.min(...prices).toFixed(2)}</p>
        <p>Max: ${Math.max(...prices).toFixed(2)}</p>
      </div>
    </div>
  );
}

export default App;