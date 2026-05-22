"""
Temporal Fusion Transformer — training, dataset building, and inference.

Replaces the old sklearn .pkl model with a multi-horizon, attention-based
architecture that provides quantile forecasts and feature importance.
"""

import pandas as pd
from loguru import logger
from typing import Optional

try:
    # pytorch-forecasting transitively requires torch; torch itself is imported
    # lazily inside the functions that need it.
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    logger.warning("pytorch-forecasting not installed. TFT model disabled.")


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ENCODER_LENGTH = 60      # 60 trading days lookback
MAX_PREDICTION_LENGTH = 10   # 10-day forecast horizon
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3

TIME_VARYING_UNKNOWN = [
    "close", "open", "high", "low", "volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_pct",
    "atr_14", "obv", "vwap", "adx_14",
    "stoch_k", "stoch_d", "cci_20", "mfi_14",
    "sentiment_score", "sentiment_volume",
]

TIME_VARYING_KNOWN = [
    "day_of_week", "day_of_month", "month", "quarter",
    "is_earnings_week", "is_holiday_proximity",
]

STATIC_CATEGORICALS = ["ticker", "sector", "market_cap_tier"]
STATIC_REALS = ["avg_daily_volume_30d", "beta"]


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_timeseries_dataset(
    df: pd.DataFrame,
    training: bool = True,
    training_cutoff: Optional[int] = None,
) -> "TimeSeriesDataSet":
    """
    df must have columns: time_idx, ticker, sector, market_cap_tier,
    all TIME_VARYING_UNKNOWN, TIME_VARYING_KNOWN, STATIC_REALS columns,
    and target column 'close_return'.
    """
    if not TFT_AVAILABLE:
        raise RuntimeError("pytorch-forecasting is not installed.")

    if training_cutoff is None:
        training_cutoff = int(df["time_idx"].max() * 0.8)

    # Filter available columns
    available_unknown = [c for c in TIME_VARYING_UNKNOWN if c in df.columns]
    available_known = [c for c in TIME_VARYING_KNOWN if c in df.columns]
    available_static_cats = [c for c in STATIC_CATEGORICALS if c in df.columns]
    available_static_reals = [c for c in STATIC_REALS if c in df.columns]

    dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff] if training else df,
        time_idx="time_idx",
        target="close_return",           # predict % return, not raw price
        group_ids=["ticker"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=available_static_cats,
        static_reals=available_static_reals,
        time_varying_known_reals=available_known,
        time_varying_unknown_reals=available_unknown,
        target_normalizer=GroupNormalizer(
            groups=["ticker"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_tft_model(training_dataset: "TimeSeriesDataSet") -> "TemporalFusionTransformer":
    if not TFT_AVAILABLE:
        raise RuntimeError("pytorch-forecasting is not installed.")

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=64,
        output_size=7,              # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )
    logger.info(f"TFT model has {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ── Trainer ───────────────────────────────────────────────────────────────────

def train_tft(
    training_dataset: "TimeSeriesDataSet",
    validation_dataset: "TimeSeriesDataSet",
    checkpoint_dir: str = "models/",
) -> "TemporalFusionTransformer":
    if not TFT_AVAILABLE:
        raise RuntimeError("pytorch-forecasting is not installed.")


    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    val_loader = validation_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )

    model = build_tft_model(training_dataset)

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, mode="min"
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tft_checkpoint",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.success(f"Training complete. Best val_loss: {early_stop.best_score:.4f}")

    # Persist the training dataset parameters next to the checkpoint. Inference
    # MUST rebuild its dataset from these (via TimeSeriesDataSet.from_parameters)
    # so categorical encoders and target normalizers match training exactly.
    import os
    import joblib
    params_path = os.path.join(checkpoint_dir, "tft_dataset_params.pkl")
    joblib.dump(training_dataset.get_parameters(), params_path)
    logger.info(f"Saved TFT dataset parameters to {params_path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

class TFTPredictor:
    """
    Inference wrapper for a trained TFT checkpoint.

    NOTE: inference rebuilds its TimeSeriesDataSet from the *training* dataset
    parameters (saved by ``train_tft`` as ``tft_dataset_params.pkl`` beside the
    checkpoint). Building a fresh dataset instead would fit new categorical
    encoders / normalizers and silently produce wrong forecasts.
    """

    _instance = None

    def __init__(self, checkpoint_path: str):
        if not TFT_AVAILABLE:
            raise RuntimeError("pytorch-forecasting is not installed.")

        import os
        import joblib

        self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        self.model.eval()

        params_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "tft_dataset_params.pkl")
        if not os.path.exists(params_path):
            raise RuntimeError(
                f"TFT dataset parameters not found at {params_path}. "
                f"Retrain with scripts/train_tft.py so encoders match the checkpoint."
            )
        self.dataset_params = joblib.load(params_path)
        logger.info(f"TFT loaded from {checkpoint_path} (+ dataset params)")

    @classmethod
    def get_instance(cls, checkpoint_path: str) -> "TFTPredictor":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def predict(self, df: pd.DataFrame, ticker: str) -> dict:
        """
        Returns a dict with:
          - point_forecasts: median predicted returns over the horizon
          - quantile_bands:  {q02, q10, q25, q50, q75, q90, q98}
          - attention_weights: feature importance (best-effort; may be omitted)
        """
        import torch

        # Rebuild the dataset from TRAINING parameters so encoders align.
        dataset = TimeSeriesDataSet.from_parameters(
            self.dataset_params, df, predict=True, stop_randomization=True
        )
        loader = dataset.to_dataloader(train=False, batch_size=1)

        with torch.no_grad():
            raw = self.model.predict(loader, mode="quantiles")

        # `predict` returns a tensor (or an object exposing `.output`).
        output = getattr(raw, "output", raw)
        predictions = output.squeeze().cpu().numpy()
        if predictions.ndim == 1:  # single-horizon edge case
            predictions = predictions.reshape(1, -1)

        quantile_labels = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
        median_idx = 3  # q50

        result = {
            "ticker": ticker,
            "horizon_days": predictions.shape[0],
            "point_forecasts": predictions[:, median_idx].tolist(),
            "quantile_bands": {
                f"q{int(q * 100):02d}": predictions[:, i].tolist()
                for i, q in enumerate(quantile_labels)
                if i < predictions.shape[1]
            },
        }

        # Attention/interpretation is best-effort — never fail a forecast for it.
        try:
            with torch.no_grad():
                raw_full = self.model.predict(loader, mode="raw")
            interpretation = self.model.interpret_output(
                getattr(raw_full, "output", raw_full), reduction="sum"
            )
            result["attention_weights"] = {
                "encoder_variables": interpretation["encoder_variables"].cpu().numpy().tolist(),
                "decoder_variables": interpretation["decoder_variables"].cpu().numpy().tolist(),
            }
        except Exception as e:
            logger.warning(f"TFT interpretation unavailable: {e}")

        return result
