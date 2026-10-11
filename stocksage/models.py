"""Fixed predeclared comparators; no automatic random validation or tuning."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import FEATURES

MODEL_NAMES = ("zero", "train_mean", "ridge", "hist_gradient_boosting")
TREE_PARAMS = dict(max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
                   learning_rate=.05, l2_regularization=1.0, early_stopping=False)


def predict(train, test, model: str, seed: int = 28) -> np.ndarray:
    """Return columns point, q10, q90. Never inspect test labels."""
    x, y, z = train[FEATURES].to_numpy(), train["target"].to_numpy(), test[FEATURES].to_numpy()
    if not len(train) or not np.isfinite(y).all():
        raise ValueError("Training labels must be observed and finite")
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    with threadpool_limits(limits=1):
        if model in ("zero", "train_mean"):
            point = np.full(len(test), 0.0 if model == "zero" else np.mean(y))
            low, high = np.full(len(test), np.quantile(y, .1)), np.full(len(test), np.quantile(y, .9))
        elif model == "ridge":
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            estimator.fit(x, y)
            point = estimator.predict(z)
            residual = y - estimator.predict(x)
            low, high = point + np.quantile(residual, .1), point + np.quantile(residual, .9)
        else:
            point = HistGradientBoostingRegressor(**TREE_PARAMS, random_state=seed).fit(x, y).predict(z)
            low, high = [HistGradientBoostingRegressor(
                **TREE_PARAMS, random_state=seed, loss="quantile", quantile=q
            ).fit(x, y).predict(z) for q in (.1, .9)]
    low, high = np.minimum.reduce([low, high, point]), np.maximum.reduce([low, high, point])
    result = np.column_stack([point, low, high])
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model output")
    return result
