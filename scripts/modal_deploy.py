"""
Modal.com deployment for GPU-accelerated TFT inference and FinBERT scoring.

Usage:
    modal deploy scripts/modal_deploy.py
"""

try:
    import modal

    # Define Modal image with all ML dependencies
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install([
            "torch==2.4.0",
            "pytorch-lightning==2.4.0",
            "pytorch-forecasting==1.1.1",
            "transformers==4.44.0",
            "huggingface-hub==0.24.5",
            "pandas==2.2.2",
            "numpy==1.26.4",
            "PyWavelets==1.7.0",
            "yfinance==0.2.43",
            "fredapi==0.5.2",
        ])
    )

    # Mount model files from local
    model_volume = modal.Volume.from_name("stocksage-models", create_if_missing=True)

    app = modal.App("stocksage-ml", image=image)


    @app.function(
        gpu="T4",
        volumes={"/models": model_volume},
        timeout=120,
        retries=2,
    )
    def predict_tft(ticker: str, df_json: str) -> dict:
        """
        GPU-accelerated TFT inference on Modal.
        Called remotely from FastAPI backend via modal.Function.lookup().
        """
        import pandas as pd
        from backend.ml.tft_model import TFTPredictor

        df = pd.read_json(df_json)
        predictor = TFTPredictor.get_instance("/models/tft_checkpoint.ckpt")
        return predictor.predict(df, ticker)


    @app.function(
        gpu=None,                   # FinBERT runs fine on CPU
        volumes={"/models": model_volume},
        timeout=60,
    )
    def score_sentiment_batch(headlines: list[str]) -> list[dict]:
        """
        Batch FinBERT sentiment scoring on Modal.
        """
        from backend.ml.finbert_sentiment import score_batch
        return score_batch(headlines)


    @app.local_entrypoint()
    def main():
        # Test call
        result = predict_tft.remote("AAPL", "{}")
        print(result)

except ImportError:
    print("modal not installed. Run: pip install modal")
