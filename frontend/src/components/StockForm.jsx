import React, { useState } from 'react';
import { Paper, InputBase, Button, Box, Chip, Stack } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

const SUGGESTIONS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL'];

/**
 * Ticker search bar. Calls onSearch(ticker) with an upper-cased, trimmed symbol.
 */
const StockForm = ({ onSearch, loading }) => {
  const [ticker, setTicker] = useState('');

  const submit = (value) => {
    const symbol = (value || '').toUpperCase().trim();
    if (symbol && !loading) onSearch(symbol);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    submit(ticker);
  };

  return (
    <Box>
      <Paper
        component="form"
        onSubmit={handleSubmit}
        elevation={0}
        sx={{
          display: 'flex',
          alignItems: 'center',
          p: '6px 6px 6px 18px',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 3,
        }}
      >
        <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
        <InputBase
          sx={{ flex: 1, fontSize: '1.05rem', letterSpacing: 0.5 }}
          placeholder="Enter a ticker — e.g. AAPL"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          inputProps={{ 'aria-label': 'stock ticker', maxLength: 6 }}
          disabled={loading}
        />
        <Button
          type="submit"
          variant="contained"
          size="large"
          disabled={loading || !ticker.trim()}
          sx={{ borderRadius: 2, px: 3 }}
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </Button>
      </Paper>

      <Stack direction="row" spacing={1} sx={{ mt: 1.5, flexWrap: 'wrap', gap: 1 }}>
        {SUGGESTIONS.map((s) => (
          <Chip
            key={s}
            label={s}
            size="small"
            variant="outlined"
            onClick={() => {
              setTicker(s);
              submit(s);
            }}
            disabled={loading}
          />
        ))}
      </Stack>
    </Box>
  );
};

export default StockForm;
