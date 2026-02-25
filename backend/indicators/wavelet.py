"""Wavelet decomposition for noise filtering."""

import numpy as np
from loguru import logger

try:
    import pywt
except ImportError:
    pywt = None
    logger.warning("PyWavelets not installed. Wavelet features disabled.")


def wavelet_smooth(prices: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    Applies discrete wavelet transform to remove high-frequency noise.
    Returns the low-frequency approximation coefficients reconstructed to original length.
    """
    if pywt is None:
        return prices

    if len(prices) < 2 ** (level + 1):
        logger.warning(f"Not enough data for wavelet level {level}. Returning original.")
        return prices

    coeffs = pywt.wavedec(prices, wavelet, level=level)
    # Zero out detail coefficients (noise), keep approximation
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    smoothed = pywt.waverec(coeffs, wavelet)
    # Align length
    return smoothed[:len(prices)]


def wavelet_denoise(prices: np.ndarray, wavelet: str = "db4", level: int = 3,
                    threshold_mode: str = "soft") -> np.ndarray:
    """
    Applies wavelet denoising with thresholding.
    Uses universal threshold (VisuShrink).
    """
    if pywt is None:
        return prices

    coeffs = pywt.wavedec(prices, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(prices)))

    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(prices)]
