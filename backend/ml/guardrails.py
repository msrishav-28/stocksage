"""
Input and output guardrails for StockSage agent predictions.

InputGuardrail:  validates ticker format, market context, and data availability.
OutputGuardrail: enforces confidence thresholds, schema compliance, and
                 emits a HOLD override when the system is under-confident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from loguru import logger


# Tickers explicitly flagged as unsupported (delisted, bankrupt, etc.)
BANNED_TICKERS: frozenset = frozenset({
    "ENRNQ", "LEHMAN", "WAMU",  # examples of delisted/defunct
})


@dataclass
class GuardrailResult:
    passed: bool
    message: str = ""
    overrides: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


class InputGuardrail:
    """
    Validates all inputs before the orchestrator runs.
    Prevents bad tickers, malformed periods, and unsupported instruments.
    """

    VALID_TICKER_PATTERN = re.compile(r"^[A-Z\.\-]{1,6}$")
    VALID_PERIODS = {"1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}

    def validate(self, ticker: str, period: str = "1y", context: dict = None) -> GuardrailResult:
        flags = []

        # 1. Ticker format
        if not self.VALID_TICKER_PATTERN.match(ticker.upper()):
            return GuardrailResult(
                passed=False,
                message=f"Invalid ticker format: '{ticker}'. Must be 1-6 alpha chars.",
                flags=["INVALID_TICKER_FORMAT"],
            )

        # 2. Banned tickers
        if ticker.upper() in BANNED_TICKERS:
            return GuardrailResult(
                passed=False,
                message=f"Ticker '{ticker}' is on the exclusion list.",
                flags=["BANNED_TICKER"],
            )

        # 3. Period validation
        if period not in self.VALID_PERIODS:
            return GuardrailResult(
                passed=False,
                message=f"Invalid period '{period}'. Must be one of: {sorted(self.VALID_PERIODS)}",
                flags=["INVALID_PERIOD"],
            )

        # 4. Soft warnings (don't block, but flag)
        if ticker.upper().endswith(".OTC") or "-" in ticker:
            flags.append("OTC_OR_ADR_TICKER")
            logger.warning(f"Ticker {ticker} may be OTC/ADR — analysis quality may be reduced.")

        return GuardrailResult(passed=True, flags=flags)


class OutputGuardrail:
    """
    Validates and potentially overrides orchestrator output.

    Rules enforced:
      1. If confidence < MIN_CONFIDENCE → force HOLD regardless of signal
      2. If synthesis_result is missing required fields → reject and return error
      3. If risk_score > 9.0 AND signal is not SELL → downgrade to HOLD
    """

    def __init__(self, min_confidence: float = 0.30):
        self.min_confidence = min_confidence

    def validate(self, result: dict) -> GuardrailResult:
        flags = []
        overrides = {}

        required_keys = {"ticker", "final_signal", "confidence", "risk_score",
                         "agent_signals", "explanation", "thesis"}
        missing = required_keys - result.keys()
        if missing:
            return GuardrailResult(
                passed=False,
                message=f"Output schema violation — missing fields: {missing}",
                flags=["SCHEMA_VIOLATION"],
            )

        confidence_pct = result.get("confidence", 0)
        confidence_frac = confidence_pct / 100 if confidence_pct > 1 else confidence_pct

        # Rule 1: Low confidence → force HOLD
        if confidence_frac < self.min_confidence:
            old_signal = result["final_signal"]
            if old_signal != "HOLD":
                overrides["final_signal"] = "HOLD"
                overrides["override_reason"] = (
                    f"Confidence {confidence_pct:.1f}% < minimum "
                    f"{self.min_confidence * 100:.0f}% — signal overridden to HOLD"
                )
                flags.append("LOW_CONFIDENCE_HOLD_OVERRIDE")
                logger.warning(
                    f"OutputGuardrail: {result['ticker']} {old_signal}→HOLD "
                    f"(confidence {confidence_pct:.1f}% < {self.min_confidence * 100:.0f}%)"
                )

        # Rule 2: Extreme risk + non-SELL → downgrade to HOLD
        risk_score = result.get("risk_score", 0)
        if risk_score >= 9.0 and result.get("final_signal") == "BUY":
            overrides["final_signal"] = "HOLD"
            overrides["override_reason"] = (
                f"Risk score {risk_score}/10 too high for BUY — downgraded to HOLD"
            )
            flags.append("HIGH_RISK_HOLD_OVERRIDE")
            logger.warning(f"OutputGuardrail: {result['ticker']} BUY→HOLD (risk={risk_score})")

        # Rule 3: Final signal must be valid enum
        final_signal = overrides.get("final_signal", result.get("final_signal"))
        if final_signal not in ("BUY", "HOLD", "SELL"):
            return GuardrailResult(
                passed=False,
                message=f"Invalid final_signal value: '{final_signal}'",
                flags=["INVALID_SIGNAL_VALUE"],
            )

        return GuardrailResult(
            passed=True,
            flags=flags,
            overrides=overrides,
            message="Output passed guardrails" + (f" with overrides: {flags}" if flags else ""),
        )

    def apply(self, result: dict, gr: GuardrailResult) -> dict:
        """Apply overrides from GuardrailResult to the prediction dict in-place."""
        result = dict(result)
        if gr.overrides:
            result.update(gr.overrides)
        result["guardrail_flags"]   = gr.flags
        result["guardrail_applied"] = bool(gr.flags)
        return result
