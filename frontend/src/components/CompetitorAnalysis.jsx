import React from 'react';
import { Card, CardContent, Typography, Box, Grid, List, ListItem, ListItemText } from '@mui/material';
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
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const CompetitorAnalysis = ({ competitors }) => {
    // Define a set of colors for the competitor chart lines
    const chartColors = ['#90caf9', '#f48fb1', '#a5d6a7', '#ffcc80', '#b39ddb'];

    const chartData = {
        // Use the first competitor's time labels as the base for the X-axis
        labels: competitors[0]?.time_labels || [],
        datasets: competitors.map((comp, index) => ({
            label: comp.name,
            data: comp.stock_prices,
            borderColor: chartColors[index % chartColors.length],
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 5,
        })),
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: 'index',
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#c9d1d9' // Style for legend text
                }
            },
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
        <Grid container spacing={3}>
            {/* List of Competitors */}
            <Grid item xs={12} md={4}>
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>Top Industry Competitors</Typography>
                        <List>
                            {competitors.map((comp) => (
                                <ListItem key={comp.ticker} divider>
                                    <ListItemText
                                        primary={comp.name}
                                        secondary={comp.ticker}
                                    />
                                    <Typography variant="h6">${comp.stock_price}</Typography>
                                </ListItem>
                            ))}
                        </List>
                    </CardContent>
                </Card>
            </Grid>
            {/* Competitor Price Chart */}
            <Grid item xs={12} md={8}>
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>Competitor Price Trends (3 Months)</Typography>
                        <Box sx={{ height: '350px' }}>
                            <Line data={chartData} options={chartOptions} />
                        </Box>
                    </CardContent>
                </Card>
            </Grid>
        </Grid>
    );
};

export default CompetitorAnalysis;