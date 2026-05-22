import React from 'react';
import {
  Grid, Card, CardContent, Box, Typography, Chip, Stack, Divider, Tooltip,
} from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import PublicIcon from '@mui/icons-material/Public';
import { directionColor } from '../theme';

const AGENT_META = {
  technical: { label: 'Technical', icon: ShowChartIcon, blurb: 'Indicator confluence' },
  sentiment: { label: 'Sentiment', icon: NewspaperIcon, blurb: 'News & FinBERT' },
  macro: { label: 'Macro', icon: PublicIcon, blurb: 'FRED environment' },
};

/** Bipolar score meter — fills from the centre, left for negative scores. */
const ScoreMeter = ({ score = 0 }) => {
  const pct = Math.min(100, Math.abs(score) * 100);
  const positive = score >= 0;
  const color = positive ? '#26a69a' : '#ef5350';
  return (
    <Box sx={{ position: 'relative', height: 8, borderRadius: 4, bgcolor: 'rgba(255,255,255,0.06)' }}>
      <Box sx={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '2px', bgcolor: 'rgba(255,255,255,0.18)' }} />
      <Box
        sx={{
          position: 'absolute', top: 0, bottom: 0, borderRadius: 4, bgcolor: color,
          width: `${pct / 2}%`,
          left: positive ? '50%' : `${50 - pct / 2}%`,
        }}
      />
    </Box>
  );
};

const StatRow = ({ label, value, color }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 0.4 }}>
    <Typography variant="body2" color="text.secondary">{label}</Typography>
    <Typography variant="body2" sx={{ color: color || 'text.primary', fontWeight: 500 }}>{value}</Typography>
  </Box>
);

const fmt = (v, digits = 2) =>
  (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(digits);

/* ── Per-agent bodies ─────────────────────────────────────────────────────── */

function TechnicalBody({ data }) {
  const ki = data.key_indicators || {};
  return (
    <>
      <StatRow label="Bullish signals" value={data.bullish_signals ?? 0} color="#26a69a" />
      <StatRow label="Bearish signals" value={data.bearish_signals ?? 0} color="#ef5350" />
      <StatRow label="Neutral signals" value={data.neutral_signals ?? 0} />
      <Divider sx={{ my: 1 }} />
      <StatRow label="RSI (14)" value={fmt(ki.rsi_14, 1)} />
      <StatRow label="ADX (14)" value={fmt(ki.adx_14, 1)} />
      <StatRow label="MACD hist" value={fmt(ki.macd_hist, 3)} />
    </>
  );
}

function SentimentBody({ data }) {
  const headlines = (data.headlines || []).slice(0, 3);
  return (
    <>
      <StatRow label="Articles analysed" value={data.total_articles ?? 0} />
      <StatRow label="Positive" value={data.bullish_count ?? 0} color="#26a69a" />
      <StatRow label="Negative" value={data.bearish_count ?? 0} color="#ef5350" />
      {headlines.length > 0 && (
        <>
          <Divider sx={{ my: 1 }} />
          <Stack spacing={0.75}>
            {headlines.map((h, i) => (
              <Box key={i} sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
                <Box sx={{
                  mt: '6px', width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                  bgcolor: directionColor(
                    h.label === 'positive' ? 'bullish' : h.label === 'negative' ? 'bearish' : 'neutral',
                  ),
                }} />
                <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.4 }}>
                  {h.headline}
                </Typography>
              </Box>
            ))}
          </Stack>
        </>
      )}
    </>
  );
}

function MacroBody({ data }) {
  const reasons = data.reasons || [];
  const snap = data.snapshot || {};
  return (
    <>
      {reasons.length > 0 ? (
        <Stack spacing={0.5} sx={{ mb: 1 }}>
          {reasons.map((r, i) => (
            <Typography key={i} variant="caption" color="text.secondary">• {r}</Typography>
          ))}
        </Stack>
      ) : (
        <Typography variant="caption" color="text.secondary">
          No strong macro signals detected.
        </Typography>
      )}
      {(snap.vix != null || snap.fed_funds_rate != null) && (
        <>
          <Divider sx={{ my: 1 }} />
          <StatRow label="VIX" value={fmt(snap.vix, 1)} />
          <StatRow label="Fed funds rate" value={snap.fed_funds_rate != null ? `${fmt(snap.fed_funds_rate, 2)}%` : '—'} />
          <StatRow label="Yield curve (10y–2y)" value={fmt(snap.yield_curve, 2)} />
        </>
      )}
    </>
  );
}

const BODIES = { technical: TechnicalBody, sentiment: SentimentBody, macro: MacroBody };

/* ── Agent card ───────────────────────────────────────────────────────────── */

function AgentCard({ name, data }) {
  const meta = AGENT_META[name];
  const Icon = meta.icon;
  const Body = BODIES[name];
  const direction = data.direction || 'neutral';
  const score = Number(data.raw_score || 0);
  const dColor = directionColor(direction);

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1 }}>
          <Box sx={{
            width: 38, height: 38, borderRadius: 2, display: 'flex',
            alignItems: 'center', justifyContent: 'center', bgcolor: `${dColor}1f`,
          }}>
            <Icon sx={{ color: dColor }} fontSize="small" />
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, lineHeight: 1.1 }}>
              {meta.label}
            </Typography>
            <Typography variant="caption" color="text.secondary">{meta.blurb}</Typography>
          </Box>
          <Chip
            label={direction}
            size="small"
            sx={{ bgcolor: `${dColor}22`, color: dColor, textTransform: 'capitalize', fontWeight: 600 }}
          />
        </Stack>

        <Box sx={{ mb: 1.5 }}>
          <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
            <Typography variant="caption" color="text.secondary">Signal score</Typography>
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              {score >= 0 ? '+' : ''}{score.toFixed(2)}
            </Typography>
          </Stack>
          <ScoreMeter score={score} />
        </Box>

        {data.error ? (
          <Tooltip title={String(data.error)}>
            <Typography variant="caption" color="warning.main">
              Agent degraded — using a neutral fallback.
            </Typography>
          </Tooltip>
        ) : (
          <Body data={data} />
        )}
      </CardContent>
    </Card>
  );
}

/** Renders the three ensemble agents side by side. */
const AgentBreakdown = ({ agentSignals }) => {
  const order = ['technical', 'sentiment', 'macro'];
  return (
    <Grid container spacing={2.5}>
      {order.map((name) =>
        agentSignals[name] ? (
          <Grid item xs={12} md={4} key={name}>
            <AgentCard name={name} data={agentSignals[name]} />
          </Grid>
        ) : null,
      )}
    </Grid>
  );
};

export default AgentBreakdown;
