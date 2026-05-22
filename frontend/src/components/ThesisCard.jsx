import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Stack, Divider } from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

/**
 * The written investment thesis plus any guardrail flags and a small
 * orchestration-trace footer.
 */
const ThesisCard = ({ prediction }) => {
  const { thesis, guardrail_flags = [], trace } = prediction;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
          <AutoAwesomeIcon fontSize="small" sx={{ color: 'primary.main' }} />
          <Typography variant="h6">Investment Thesis</Typography>
        </Stack>

        <Typography variant="body1" sx={{ color: 'text.primary', lineHeight: 1.75 }}>
          {thesis}
        </Typography>

        {guardrail_flags.length > 0 && (
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: 'wrap', gap: 1 }}>
            {guardrail_flags.map((flag) => (
              <Chip
                key={flag}
                label={flag.replace(/_/g, ' ').toLowerCase()}
                size="small"
                color="warning"
                variant="outlined"
              />
            ))}
          </Stack>
        )}

        {trace && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              <Typography variant="caption" color="text.secondary">
                {trace.span_count} agents · {Math.round(trace.total_duration_ms)} ms
              </Typography>
              {trace.failed_agents?.length > 0 && (
                <Typography variant="caption" color="warning.main">
                  degraded: {trace.failed_agents.join(', ')}
                </Typography>
              )}
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ThesisCard;
