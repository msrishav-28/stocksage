import React from 'react';
import { Card, CardContent, Box, Typography, LinearProgress, Stack, Chip } from '@mui/material';
import { signalColor } from '../theme';

/** Map a 0–10 risk score to a label + colour. */
function riskBand(score) {
  if (score >= 7) return { label: 'High risk', color: '#ef5350' };
  if (score >= 4) return { label: 'Moderate risk', color: '#ffa726' };
  return { label: 'Low risk', color: '#26a69a' };
}

const Metric = ({ label, children }) => (
  <Box>
    <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: 1 }}>
      {label}
    </Typography>
    <Box sx={{ mt: 0.5 }}>{children}</Box>
  </Box>
);

/**
 * Headline prediction card: signal, confidence, risk, and the one-line
 * explanation. `prediction` is the /api/predict response.
 */
const SignalHero = ({ prediction }) => {
  const { ticker, final_signal, confidence, weighted_score, risk_score, explanation } = prediction;
  const color = signalColor(final_signal);
  const risk = riskBand(risk_score);
  const overridden = prediction.guardrail_applied;

  return (
    <Card>
      <CardContent sx={{ p: { xs: 2.5, md: 3.5 } }}>
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          spacing={3}
          alignItems={{ xs: 'flex-start', md: 'center' }}
        >
          {/* Signal block */}
          <Box
            sx={{
              minWidth: 200,
              px: 3,
              py: 2.5,
              borderRadius: 3,
              border: `1px solid ${color}55`,
              background: `${color}14`,
              textAlign: 'center',
            }}
          >
            <Typography variant="overline" color="text.secondary">
              {ticker}
            </Typography>
            <Typography variant="h3" sx={{ color, lineHeight: 1.1 }}>
              {final_signal}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              weighted score {weighted_score >= 0 ? '+' : ''}
              {weighted_score?.toFixed(3)}
            </Typography>
          </Box>

          {/* Metrics block */}
          <Box sx={{ flex: 1, width: '100%' }}>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={4} sx={{ mb: 2 }}>
              <Metric label="Confidence">
                <Typography variant="h5">{Math.round(confidence)}%</Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, Math.max(0, confidence))}
                  sx={{ mt: 0.5, width: 160, height: 6, borderRadius: 3 }}
                />
              </Metric>
              <Metric label="Risk">
                <Typography variant="h5" sx={{ color: risk.color }}>
                  {risk_score?.toFixed(1)}
                  <Typography component="span" variant="body2" color="text.secondary">
                    {' '}/ 10
                  </Typography>
                </Typography>
                <Chip
                  label={risk.label}
                  size="small"
                  sx={{ mt: 0.5, bgcolor: `${risk.color}22`, color: risk.color }}
                />
              </Metric>
            </Stack>

            <Typography variant="body2" color="text.secondary">
              {explanation}
            </Typography>

            {overridden && (
              <Chip
                label="Adjusted by risk guardrail"
                size="small"
                color="warning"
                variant="outlined"
                sx={{ mt: 1.5 }}
              />
            )}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default SignalHero;
