import React from 'react';
import './AIAnalysis.css';

const AIAnalysis = ({ analysis }) => {
  if (!analysis) return null;

  const getSentimentClass = (sentiment) => {
    return `sentiment-${sentiment.toLowerCase()}`;
  };

  const getRiskClass = (risk) => {
    return `risk-${risk.toLowerCase()}`;
  };

  const getRecommendationClass = (action) => {
    const actionMap = {
      'BUY': 'buy',
      'SELL': 'sell',
      'HOLD': 'hold',
      'WATCH': 'watch',
      'AVOID': 'sell',
      'SELL/AVOID': 'sell',
      'TAKE PROFITS': 'sell',
      'POTENTIAL BUY': 'watch'
    };
    return actionMap[action] || 'hold';
  };

  return (
    <div className="ai-analysis">
      <h2>🤖 AI-Powered Analysis</h2>
      
      <div className="analysis-grid">
        {/* Sentiment Analysis Card */}
        <div className="analysis-card">
          <h3>📊 Market Sentiment</h3>
          <div className={`sentiment-badge ${getSentimentClass(analysis.sentiment)}`}>
            {analysis.sentiment}
          </div>
          <div className="confidence">
            Confidence: {analysis.sentiment_confidence}%
          </div>
          <div className="sentiment-score">
            Score: {analysis.sentiment_score > 0 ? '+' : ''}{analysis.sentiment_score}
          </div>
          <h4>Key Factors:</h4>
          <ul className="factors">
            {analysis.sentiment_factors.map((factor, i) => (
              <li key={i}>{factor}</li>
            ))}
          </ul>
        </div>

        {/* Risk Assessment Card */}
        <div className="analysis-card">
          <h3>⚠️ Risk Assessment</h3>
          <div className={`risk-badge ${getRiskClass(analysis.risk_level)}`}>
            {analysis.risk_level} RISK
          </div>
          <div className="risk-score">
            Risk Score: {analysis.risk_score}/10
          </div>
          <h4>Risk Factors:</h4>
          <ul className="factors">
            {analysis.risk_factors.map((factor, i) => (
              <li key={i}>{factor}</li>
            ))}
          </ul>
        </div>

        {/* Recommendation Card */}
        <div className="analysis-card recommendation-card">
          <h3>💡 AI Recommendation</h3>
          <div className={`recommendation-action recommendation-${getRecommendationClass(analysis.recommendation.action)}`}>
            {analysis.recommendation.action}
          </div>
          <p className="recommendation-summary">
            {analysis.recommendation.summary}
          </p>
          <p className="recommendation-details">
            {analysis.recommendation.details}
          </p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-section">
        <h3>📈 Key Technical Metrics</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-label">RSI (14)</span>
            <span className="metric-value">{analysis.technical_indicators.rsi}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">52-Week Position</span>
            <span className="metric-value">{analysis.technical_indicators.position_52w}%</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">52-Week High</span>
            <span className="metric-value">${analysis.week_52_high}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">52-Week Low</span>
            <span className="metric-value">${analysis.week_52_low}</span>
          </div>
          {analysis.pe_ratio > 0 && (
            <div className="metric-card">
              <span className="metric-label">P/E Ratio</span>
              <span className="metric-value">{analysis.pe_ratio.toFixed(2)}</span>
            </div>
          )}
          {analysis.dividend_yield > 0 && (
            <div className="metric-card">
              <span className="metric-label">Dividend Yield</span>
              <span className="metric-value">{analysis.dividend_yield.toFixed(2)}%</span>
            </div>
          )}
        </div>
      </div>

      {/* ELI5 Explanation */}
      <div className="eli5-section">
        <h3>🎓 Simple Explanation (ELI5)</h3>
        <div className="eli5-card">
          <pre>{analysis.eli5_explanation}</pre>
        </div>
      </div>
    </div>
  );
};

export default AIAnalysis;