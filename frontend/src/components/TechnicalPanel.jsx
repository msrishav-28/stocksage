import React, { useEffect, useState } from 'react';
import {
  Card, CardContent, Box, Typography, Grid, Chip, Stack, CircularProgress, Alert,
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import { getTechnical, errorMessage } from '../api';
import { directionColor } from '../theme';

const KEY_INDICATORS = [
  ['rsi_14', 'RSI (14)'],
  ['adx_14', 'ADX (14)'],
  ['macd', 'MACD'],
  ['ema_50', 'EMA 50'],
  ['ema_200', 'EMA 200'],
  ['atr_14', 'ATR (14)'],
  ['bb_pct', 'Bollinger %B'],
  ['mfi_14', 'MFI (14)'],
  ['cmf_20', 'CMF (20)'],
];

const compact = (n) => {
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return n.toFixed(2);
};

const TechnicalPanel = ({ ticker }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    getTechnical(ticker)
      .then((res) => { if (!cancelled) setData(res.data); })
      .catch((err) => { if (!cancelled) setError(errorMessage(err, 'Could not load technical data')); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ticker]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!data) return null;

  const { confluence = {}, indicators = {}, price = {}, history = [] } = data;
  const dColor = directionColor(confluence.direction);

  const chartData = {
    labels: history.map((h) => h.date),
    datasets: [
      {
        label: `${ticker} close`,
        data: history.map((h) => h.close),
        borderColor: '#5b8def',
        backgroundColor: 'rgba(91,141,239,0.12)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (c) => ` $${c.parsed.y}` } },
    },
    scales: {
      x: { ticks: { maxTicksLimit: 8, color: '#8b97a6' }, grid: { color: '#1d2530' } },
      y: { ticks: { color: '#8b97a6' }, grid: { color: '#1d2530' } },
    },
  };

  return (
    <Stack spacing={2.5}>
      <Card>
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="h6">{ticker} price — last {history.length} sessions</Typography>
            <Typography variant="h6">${price.close}</Typography>
          </Stack>
          <Box className="chart-container chart-container--tall">
            {history.length > 0
              ? <Line data={chartData} options={chartOptions} />
              : <Typography color="text.secondary">No price history available.</Typography>}
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="overline" color="text.secondary">Indicator confluence</Typography>
              <Typography variant="h4" sx={{ color: dColor, textTransform: 'capitalize', my: 0.5 }}>
                {confluence.direction || 'neutral'}
              </Typography>
              <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap', gap: 1 }}>
                <Chip size="small" label={`${confluence.bullish_signals ?? 0} bullish`}
                      sx={{ bgcolor: '#26a69a22', color: '#26a69a' }} />
                <Chip size="small" label={`${confluence.bearish_signals ?? 0} bearish`}
                      sx={{ bgcolor: '#ef535022', color: '#ef5350' }} />
                <Chip size="small" label={`${confluence.neutral_signals ?? 0} neutral`} />
              </Stack>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
                Score {confluence.raw_score >= 0 ? '+' : ''}{confluence.raw_score} across {confluence.signal_count} indicator groups.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="overline" color="text.secondary">Key indicators</Typography>
              <Grid container spacing={1.5} sx={{ mt: 0.5 }}>
                {KEY_INDICATORS.filter(([k]) => indicators[k] !== undefined).map(([k, label]) => (
                  <Grid item xs={6} sm={4} key={k}>
                    <Box sx={{ p: 1.5, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)' }}>
                      <Typography variant="caption" color="text.secondary">{label}</Typography>
                      <Typography variant="h6">{compact(Number(indicators[k]))}</Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Stack>
  );
};

export default TechnicalPanel;
