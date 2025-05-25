import Grid from '@mui/material/Grid';
import React, { useState } from 'react';
import { Card, CardContent, TextField, Button, Box, CircularProgress, Typography } from '@mui/material';

const StockForm = ({ onSubmit, loading }) => {
  const [companyName, setCompanyName] = useState('');
  const [ticker, setTicker] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!companyName && !ticker) {
      alert('Please enter a company name or ticker symbol.');
      return;
    }
    onSubmit(companyName, ticker);
  };

  return (
    <Card sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h5" component="h2" gutterBottom>
          Analyze & Predict Stock Performance
        </Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={5}>
              <TextField
                fullWidth
                label="Company Name (e.g., Apple)"
                variant="outlined"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                disabled={loading}
              />
            </Grid>
             <Grid item xs={12} sm={2} sx={{textAlign: 'center'}}>
                <Typography color="text.secondary">OR</Typography>
            </Grid>
            <Grid item xs={12} sm={5}>
              <TextField
                fullWidth
                label="Stock Ticker (e.g., AAPL)"
                variant="outlined"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                disabled={loading}
              />
            </Grid>
          </Grid>
          <Box sx={{ mt: 2, position: 'relative' }}>
            <Button
              type="submit"
              variant="contained"
              size="large"
              fullWidth
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze & Predict'}
            </Button>
            {loading && (
              <CircularProgress
                size={24}
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  marginTop: '-12px',
                  marginLeft: '-12px',
                }}
              />
            )}
          </Box>
        </form>
      </CardContent>
    </Card>
  );
};

export default StockForm;