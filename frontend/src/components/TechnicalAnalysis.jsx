import React from 'react';
import { Grid, Card, CardContent, Typography, List, ListItem, ListItemText, Divider, Box } from '@mui/material';
import { Bar } from 'react-chartjs-2';

const TechnicalAnalysis = ({ indicators, volumes, labels }) => {
  const chartData = {
    labels: labels,
    datasets: [
      {
        label: 'Volume',
        data: volumes,
        backgroundColor: 'rgba(144, 202, 249, 0.5)',
        borderColor: '#90caf9',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#c9d1d9', maxTicksLimit: 8 },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: {
          color: '#c9d1d9',
          callback: (value) => value.toLocaleString(),
        },
      },
    },
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={5}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Key Technical Indicators</Typography>
            <List>
              <ListItem>
                <ListItemText primary="RSI (14)" />
                <Typography variant="body1">{indicators.rsi}</Typography>
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText primary="20-Day MA" />
                <Typography variant="body1">${indicators.ma_20}</Typography>
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText primary="50-Day MA" />
                <Typography variant="body1">${indicators.ma_50}</Typography>
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText primary="52-Week Position" />
                <Typography variant="body1">{indicators.position_52w}%</Typography>
              </ListItem>
            </List>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={7}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Volume Analysis</Typography>
            <Box sx={{ height: '300px' }}>
                <Bar data={chartData} options={chartOptions} />
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default TechnicalAnalysis;