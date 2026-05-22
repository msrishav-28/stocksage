import React, { useEffect, useState } from 'react';
import {
  Card, CardContent, Box, Typography, Grid, Stack, CircularProgress, Alert,
  Table, TableBody, TableCell, TableHead, TableRow,
} from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { getCompetitors, errorMessage } from '../api';

const money = (n) => {
  if (n === null || n === undefined) return '—';
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  return `$${n}`;
};
const num = (n, d = 2) => (n === null || n === undefined ? '—' : Number(n).toFixed(d));

const Fact = ({ label, value }) => (
  <Box sx={{ p: 1.5, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)' }}>
    <Typography variant="caption" color="text.secondary">{label}</Typography>
    <Typography variant="h6">{value}</Typography>
  </Box>
);

const CompetitorPanel = ({ ticker }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    getCompetitors(ticker)
      .then((res) => { if (!cancelled) setData(res.data); })
      .catch((err) => { if (!cancelled) setError(errorMessage(err, 'Could not load competitor data')); })
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

  const { sector, company = {}, peers = [] } = data;

  const chartData = {
    labels: peers.map((p) => p.ticker),
    datasets: [
      {
        label: '3-month return %',
        data: peers.map((p) => p.change_3mo_pct),
        backgroundColor: peers.map((p) => (p.change_3mo_pct >= 0 ? '#26a69a' : '#ef5350')),
        borderRadius: 4,
      },
    ],
  };
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8b97a6' }, grid: { display: false } },
      y: { ticks: { color: '#8b97a6', callback: (v) => `${v}%` }, grid: { color: '#1d2530' } },
    },
  };

  return (
    <Stack spacing={2.5}>
      <Card>
        <CardContent>
          <Typography variant="overline" color="text.secondary">
            {company.name || ticker} · {sector}
          </Typography>
          <Grid container spacing={1.5} sx={{ mt: 0.5 }}>
            <Grid item xs={6} sm={3}><Fact label="Market cap" value={money(company.market_cap)} /></Grid>
            <Grid item xs={6} sm={3}><Fact label="P/E ratio" value={num(company.pe_ratio, 1)} /></Grid>
            <Grid item xs={6} sm={3}><Fact label="Beta" value={num(company.beta)} /></Grid>
            <Grid item xs={6} sm={3}>
              <Fact label="52-week range" value={`${num(company['52w_low'], 0)}–${num(company['52w_high'], 0)}`} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {peers.length === 0 ? (
        <Alert severity="info">No peer companies found for this sector.</Alert>
      ) : (
        <>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>Peer 3-month performance</Typography>
              <Box className="chart-container">
                <Bar data={chartData} options={chartOptions} />
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ overflowX: 'auto' }}>
              <Typography variant="h6" sx={{ mb: 1 }}>Peer comparison</Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Ticker</TableCell>
                    <TableCell>Company</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell align="right">3M %</TableCell>
                    <TableCell align="right">Market cap</TableCell>
                    <TableCell align="right">P/E</TableCell>
                    <TableCell align="right">Beta</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {peers.map((p) => (
                    <TableRow key={p.ticker}>
                      <TableCell sx={{ fontWeight: 600 }}>{p.ticker}</TableCell>
                      <TableCell sx={{ color: 'text.secondary' }}>{p.name}</TableCell>
                      <TableCell align="right">${p.price}</TableCell>
                      <TableCell align="right" sx={{ color: p.change_3mo_pct >= 0 ? '#26a69a' : '#ef5350' }}>
                        {p.change_3mo_pct >= 0 ? '+' : ''}{p.change_3mo_pct}%
                      </TableCell>
                      <TableCell align="right">{money(p.market_cap)}</TableCell>
                      <TableCell align="right">{num(p.pe_ratio, 1)}</TableCell>
                      <TableCell align="right">{num(p.beta)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </Stack>
  );
};

export default CompetitorPanel;
