import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Paper, LinearProgress } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PriceDisplay = ({
  companyName,
  ticker,
  description,
  currentPrice,
  priceChange,
  changePercent,
  predictedPrice,
  predictionConfidence,
  stockPrices,
  timeLabels
}) => {
  const priceChangeClass = changePercent >= 0 ? 'price-positive' : 'price-negative';
  const priceChangeSymbol = changePercent >= 0 ? '+' : '';

  const chartData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'Stock Price',
        data: stockPrices,
        borderColor: '#90caf9',
        backgroundColor: 'rgba(144, 202, 249, 0.2)',
        tension: 0.3,
        fill: true,
        pointRadius: 0,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 10,
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#c9d1d9', maxTicksLimit: 8 },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#c9d1d9' },
      },
    },
  };

  return (
    <Card>
      <CardContent>
        <Grid container spacing={3}>
          {/* Company Info */}
          <Grid item xs={12} md={8}>
            <Typography variant="h4" component="h2">{companyName} ({ticker})</Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
              {description}
            </Typography>
          </Grid>

          {/* Price Info */}
          <Grid item xs={12} md={4}>
            <Paper elevation={3} sx={{ p: 2, textAlign: 'right' }}>
              <Typography variant="h3">${currentPrice?.toFixed(2)}</Typography>
              <Typography variant="h6" className={priceChangeClass}>
                {priceChangeSymbol}{priceChange?.toFixed(2)} ({priceChangeSymbol}{changePercent?.toFixed(2)}%)
              </Typography>
            </Paper>
          </Grid>
          
          {/* Prediction and Chart */}
          <Grid item xs={12} md={4}>
            <Typography variant="h6">Prediction</Typography>
            <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
              ${predictedPrice?.toFixed(2)}
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              Next Day Close
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Prediction Confidence: {predictionConfidence}%
              </Typography>
              <LinearProgress variant="determinate" value={predictionConfidence} />
            </Box>
          </Grid>

          <Grid item xs={12} md={8}>
            <Box sx={{ height: '300px' }}>
              <Line data={chartData} options={chartOptions} />
            </Box>
          </Grid>

        </Grid>
      </CardContent>
    </Card>
  );
};

export default PriceDisplay;