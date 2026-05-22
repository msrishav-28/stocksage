/**
 * StockSage API client.
 *
 * In development, REACT_APP_API_URL is left blank and the CRA dev-server proxy
 * (package.json "proxy") forwards requests to the FastAPI backend on :8000.
 * In production, set REACT_APP_API_URL to the deployed API origin.
 */
import axios from 'axios';

const baseURL = process.env.REACT_APP_API_URL || '';

const client = axios.create({
  baseURL,
  timeout: 60000, // predictions fetch live market data — allow generous headroom
  headers: { 'Content-Type': 'application/json' },
});

/** Extract a human-readable message from an axios error. */
export function errorMessage(err, fallback = 'Something went wrong') {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail[0]?.msg || fallback;
  }
  if (err?.code === 'ECONNABORTED') return 'The request timed out. Please try again.';
  if (err?.message === 'Network Error') return 'Cannot reach the StockSage API.';
  return fallback;
}

export const getPrediction = (ticker, period = '2y') =>
  client.post('/api/predict/', { ticker, period });

export const getTechnical = (ticker, period = '1y') =>
  client.get(`/api/technical/${encodeURIComponent(ticker)}`, { params: { period } });

export const getCompetitors = (ticker) =>
  client.get(`/api/competitor/${encodeURIComponent(ticker)}`);

export const getSentiment = (ticker, hours = 48) =>
  client.get(`/api/sentiment/${encodeURIComponent(ticker)}`, { params: { hours } });

export const getHealth = () => client.get('/health');

export default client;
