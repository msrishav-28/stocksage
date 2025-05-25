import React, { useState } from 'react';
import axios from 'axios';
import { Toaster, toast } from 'react-hot-toast';
import { createTheme, ThemeProvider, Container, Box, Typography, CircularProgress, Tabs, Tab } from '@mui/material';

import StockForm from './components/StockForm';
import PriceDisplay from './components/PriceDisplay';
import AIAnalysis from './components/AIAnalysis';
import TechnicalAnalysis from './components/TechnicalAnalysis';
import CompetitorAnalysis from './components/CompetitorAnalysis';
import './App.css';

// Create a dark theme using Material-UI
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: 'Poppins, sans-serif',
  },
});

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  // Handles the API call to the Flask backend
  const handleAnalyze = async (companyName, ticker) => {
    setLoading(true);
    setResults(null);
    const toastId = toast.loading('Fetching data and running analysis...');
    try {
      const response = await axios.post('/analyze', { company_name: companyName, ticker });
      
      if (response.data.success) {
        setResults(response.data);
        toast.success('Analysis complete!', { id: toastId });
      } else {
        toast.error(response.data.error || 'Error analyzing stock', { id: toastId });
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('An error occurred. Please check the console.', { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Renders the content for the selected tab
  const renderTabContent = () => {
    if (!results) return null;
    
    switch(activeTab) {
      case 0:
        return <AIAnalysis analysis={results.ai_analysis} />;
      case 1:
        return <TechnicalAnalysis 
                  indicators={results.ai_analysis.technical_indicators}
                  volumes={results.volumes}
                  labels={results.time_labels}
                />;
      case 2:
        return <CompetitorAnalysis competitors={results.top_competitors} />;
      default:
        return null;
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <Toaster position="bottom-right" />
      <Box className="App" sx={{ bgcolor: 'background.default', color: 'text.primary' }}>
        <header className="app-header">
          <Container>
            <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
              StockMind Pro
            </Typography>
            <Typography variant="h6" component="p" sx={{ color: 'text.secondary' }}>
              AI-Powered Stock Analysis & Prediction
            </Typography>
          </Container>
        </header>
        
        <Container sx={{ py: 4 }}>
          <StockForm onSubmit={handleAnalyze} loading={loading} />
          
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          )}

          {results && (
            <Box sx={{ mt: 4 }}>
              <PriceDisplay 
                companyName={results.company_name}
                ticker={results.ticker}
                description={results.description}
                currentPrice={results.current_price}
                priceChange={results.price_change}
                changePercent={results.change_percent}
                predictedPrice={results.predicted_price}
                predictionConfidence={results.prediction_confidence}
                stockPrices={results.stock_prices}
                timeLabels={results.time_labels}
              />
              
              <Box sx={{ borderBottom: 1, borderColor: 'divider', my: 3 }}>
                <Tabs value={activeTab} onChange={handleTabChange} centered>
                  <Tab label="AI Analysis" />
                  <Tab label="Technical" />
                  <Tab label="Competitors" />
                </Tabs>
              </Box>

              {renderTabContent()}
            </Box>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;