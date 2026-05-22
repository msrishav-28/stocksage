import { createTheme } from '@mui/material/styles';

/** Signal → colour mapping, shared across components. */
export const SIGNAL_COLORS = {
  BUY: '#26a69a',
  SELL: '#ef5350',
  HOLD: '#ffa726',
};

/** Agent direction → colour mapping. */
export const DIRECTION_COLORS = {
  bullish: '#26a69a',
  bearish: '#ef5350',
  neutral: '#90a4ae',
};

export const directionColor = (d) => DIRECTION_COLORS[d] || DIRECTION_COLORS.neutral;
export const signalColor = (s) => SIGNAL_COLORS[s] || SIGNAL_COLORS.HOLD;

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#5b8def' },
    secondary: { main: '#26a69a' },
    background: { default: '#0e1116', paper: '#161b22' },
    text: { primary: '#e6edf3', secondary: '#8b97a6' },
    divider: '#232a35',
    success: { main: '#26a69a' },
    error: { main: '#ef5350' },
    warning: { main: '#ffa726' },
  },
  typography: {
    fontFamily: "'Poppins', sans-serif",
    h3: { fontWeight: 700 },
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    button: { textTransform: 'none', fontWeight: 600 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid #232a35',
        },
      },
    },
    MuiButton: { defaultProps: { disableElevation: true } },
  },
});

export default theme;
