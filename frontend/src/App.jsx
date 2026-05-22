import React, { useState } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import {
  Container, Box, Typography, Tabs, Tab, CircularProgress, Alert, Stack, Chip,
} from '@mui/material';
import InsightsIcon from '@mui/icons-material/Insights';

import StockForm from './components/StockForm';
import SignalHero from './components/SignalHero';
import ThesisCard from './components/ThesisCard';
import AgentBreakdown from './components/AgentBreakdown';
import TechnicalPanel from './components/TechnicalPanel';
import CompetitorPanel from './components/CompetitorPanel';
import { getPrediction, errorMessage } from './api';

function EmptyState() {
  return (
    <Box sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
      <InsightsIcon sx={{ fontSize: 56, color: 'primary.main', opacity: 0.8 }} />
      <Typography variant="h6" sx={{ mt: 2, color: 'text.primary' }}>
        Analyze any stock with a multi-agent AI ensemble
      </Typography>
      <Typography variant="body2" sx={{ mt: 1, maxWidth: 520, mx: 'auto' }}>
        Enter a ticker above. StockSage runs technical, sentiment, and macro
        agents, then synthesizes an explainable BUY / HOLD / SELL signal with a
        confidence score and written thesis.
      </Typography>
    </Box>
  );
}

function App() {
  const [prediction, setPrediction] = useState(null);
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState(0);

  const handleSearch = async (symbol) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setTab(0);
    const toastId = toast.loading(`Analyzing ${symbol}…`);
    try {
      const { data } = await getPrediction(symbol);
      setPrediction(data);
      setTicker(symbol);
      toast.success(`${symbol}: ${data.final_signal}`, { id: toastId });
    } catch (err) {
      const msg = errorMessage(err, 'Failed to analyze that ticker');
      setError(msg);
      toast.error(msg, { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box className="app-root">
      <Toaster position="bottom-right" toastOptions={{ style: { background: '#161b22', color: '#e6edf3' } }} />

      <Box className="app-header">
        <Container maxWidth="lg" sx={{ py: 3 }}>
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <InsightsIcon sx={{ color: 'primary.main', fontSize: 32 }} />
            <Box>
              <Typography variant="h5" sx={{ lineHeight: 1.1 }}>StockSage</Typography>
              <Typography variant="caption" color="text.secondary">
                AI-powered stock analysis &amp; prediction
              </Typography>
            </Box>
            <Box sx={{ flex: 1 }} />
            <Chip label="v2.0" size="small" variant="outlined" />
          </Stack>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <StockForm onSearch={handleSearch} loading={loading} />

        {loading && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <CircularProgress />
            <Typography color="text.secondary" sx={{ mt: 2 }}>
              Running the agent ensemble…
            </Typography>
          </Box>
        )}

        {!loading && error && (
          <Alert severity="error" sx={{ mt: 4 }}>{error}</Alert>
        )}

        {!loading && !error && !prediction && <EmptyState />}

        {!loading && prediction && (
          <Box sx={{ mt: 4 }}>
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Overview" />
              <Tab label="Technical" />
              <Tab label="Competitors" />
            </Tabs>

            {tab === 0 && (
              <Stack spacing={2.5}>
                <SignalHero prediction={prediction} />
                <ThesisCard prediction={prediction} />
                <Box>
                  <Typography variant="overline" color="text.secondary">
                    Agent breakdown
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    <AgentBreakdown agentSignals={prediction.agent_signals || {}} />
                  </Box>
                </Box>
              </Stack>
            )}

            {tab === 1 && <TechnicalPanel ticker={ticker} />}
            {tab === 2 && <CompetitorPanel ticker={ticker} />}
          </Box>
        )}
      </Container>

      <Box sx={{ borderTop: 1, borderColor: 'divider', py: 3, mt: 4 }}>
        <Container maxWidth="lg">
          <Typography variant="caption" color="text.secondary">
            StockSage is an analytical tool, not financial advice. Markets carry risk.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
