import React from 'react';
import { Grid, Card, CardContent, Typography, Box, List, ListItem, ListItemIcon, ListItemText, Paper } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import DangerousOutlinedIcon from '@mui/icons-material/DangerousOutlined';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import PauseIcon from '@mui/icons-material/Pause';

const getSentimentStyle = (sentiment) => {
  switch (sentiment) {
    case 'BULLISH':
      return { color: '#4caf50', icon: <ShowChartIcon /> };
    case 'BEARISH':
      return { color: '#f44336', icon: <TrendingDownIcon /> };
    default:
      return { color: '#ffc107', icon: <PauseIcon /> };
  }
};

const getRiskStyle = (risk) => {
    switch (risk) {
      case 'HIGH':
        return { color: '#f44336', icon: <DangerousOutlinedIcon /> };
      case 'MEDIUM':
        return { color: '#ffc107', icon: <HelpOutlineIcon /> };
      default:
        return { color: '#4caf50', icon: <CheckCircleOutlineIcon /> };
    }
  };

const getRecommendationAction = (action) => {
    const actionMap = {
      'BUY': { color: 'success.main', text: 'BUY' },
      'SELL': { color: 'error.main', text: 'SELL' },
      'HOLD': { color: 'warning.main', text: 'HOLD' },
      'WATCH': { color: 'info.main', text: 'WATCH' },
      'AVOID': { color: 'error.main', text: 'AVOID' },
      'SELL/AVOID': { color: 'error.main', text: 'SELL / AVOID' },
      'TAKE PROFITS': { color: 'success.light', text: 'TAKE PROFITS' },
      'POTENTIAL BUY': { color: 'info.light', text: 'POTENTIAL BUY' },
    };
    return actionMap[action] || { color: 'text.secondary', text: 'HOLD' };
};

const AIAnalysis = ({ analysis }) => {
  if (!analysis) return null;

  const sentimentStyle = getSentimentStyle(analysis.sentiment);
  const riskStyle = getRiskStyle(analysis.risk_level);
  const recommendationStyle = getRecommendationAction(analysis.recommendation.action);

  return (
    <Grid container spacing={3}>
      {/* Recommendation Card */}
      <Grid item xs={12}>
        <Card sx={{ borderLeft: `5px solid ${recommendationStyle.color}` }}>
          <CardContent>
            <Typography variant="h5" component="div" gutterBottom>
              AI Recommendation
            </Typography>
            <Typography variant="h3" sx={{ color: recommendationStyle.color, fontWeight: 'bold' }}>
              {recommendationStyle.text}
            </Typography>
            <Typography variant="subtitle1" sx={{ mt: 1 }}>
              {analysis.recommendation.summary}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {analysis.recommendation.details}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      {/* Sentiment Analysis Card */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Market Sentiment</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                {sentimentStyle.icon}
                <Typography variant="h4" sx={{ ml: 1, color: sentimentStyle.color }}>
                    {analysis.sentiment}
                </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">Confidence: {analysis.sentiment_confidence}%</Typography>
            <Typography variant="body2" color="text.secondary">Score: {analysis.sentiment_score > 0 ? '+' : ''}{analysis.sentiment_score}</Typography>
            <Typography variant="subtitle2" sx={{ mt: 2 }}>Key Factors:</Typography>
            <List dense>
              {analysis.sentiment_factors.map((factor, i) => (
                <ListItem key={i}>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <CheckCircleOutlineIcon fontSize="small" color="success" />
                  </ListItemIcon>
                  <ListItemText primary={factor} />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Grid>

      {/* Risk Assessment Card */}
      <Grid item xs={12} md={6}>
        <Card>
            <CardContent>
                <Typography variant="h6" gutterBottom>Risk Assessment</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {riskStyle.icon}
                    <Typography variant="h4" sx={{ ml: 1, color: riskStyle.color }}>
                        {analysis.risk_level} RISK
                    </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">Risk Score: {analysis.risk_score}/10</Typography>
                <Typography variant="subtitle2" sx={{ mt: 2 }}>Risk Factors:</Typography>
                <List dense>
                {analysis.risk_factors.map((factor, i) => (
                    <ListItem key={i}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                            <DangerousOutlinedIcon fontSize="small" color="error" />
                        </ListItemIcon>
                        <ListItemText primary={factor} />
                    </ListItem>
                ))}
                </List>
            </CardContent>
        </Card>
      </Grid>

      {/* ELI5 Explanation */}
      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 3, backgroundColor: 'background.default' }}>
          <Typography variant="h6" gutterBottom>Simple Explanation (ELI5)</Typography>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
            {analysis.eli5_explanation}
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default AIAnalysis;