"""
extract_features.py

Converts raw signals (time domain) into a vector of physically
interpretable features, ready to feed a scikit-learn classifier.

Extracted features:
    - dominant_freq : frequency of the highest spectral peak (Hz)
    - dominant_amp   : magnitude (FFT) of that peak
    - variance      : variance of the signal in the time domain
    - total_energy   : total energy of the spectrum (sum of |FFT|^2)
    - estimated_snr    : ratio between the dominant peak's energy and
                         the rest of the spectrum's energy (SNR proxy)
    - bandwidth     : energy spread around the dominant peak
"""

import numpy as np
from scipy.fft import rfft, rfftfreq

from generate_signals import FS, N_SAMPLES


def _spectrum(signal):
    """Returns (frequencies, magnitudes) of a real signal's spectrum."""
    magnitudes = np.abs(rfft(signal))
    frequencies = rfftfreq(len(signal), d=1.0 / FS)
    return frequencies, magnitudes


def extract_features_single_signal(signal):
    """Computes the feature vector for a single 1D signal."""
    frequencies, magnitudes = _spectrum(signal)

    # We ignore the DC component (frequency 0) so it is not mistaken
    # for a spurious "dominant frequency".
    magnitudes_no_dc = magnitudes.copy()
    magnitudes_no_dc[0] = 0.0

    peak_idx = np.argmax(magnitudes_no_dc)
    dominant_freq = frequencies[peak_idx]
    dominant_amp = magnitudes_no_dc[peak_idx]

    total_energy = np.sum(magnitudes_no_dc ** 2)
    peak_energy = dominant_amp ** 2
    rest_energy = max(total_energy - peak_energy, 1e-12)
    estimated_snr = 10 * np.log10(peak_energy / rest_energy)

    # Bandwidth: how many Hz around the peak concentrate its energy
    window = 3  # bins on each side of the peak
    lo, hi = max(peak_idx - window, 0), min(peak_idx + window + 1, len(magnitudes_no_dc))
    bandwidth = frequencies[hi - 1] - frequencies[lo] if hi > lo else 0.0

    variance = np.var(signal)

    return {
        "dominant_freq": dominant_freq,
        "dominant_amp": dominant_amp,
        "variance": variance,
        "total_energy": total_energy,
        "estimated_snr": estimated_snr,
        "bandwidth": bandwidth,
    }


FEATURE_NAMES = [
    "dominant_freq",
    "dominant_amp",
    "variance",
    "total_energy",
    "estimated_snr",
    "bandwidth",
]


def extract_features(X_time):
    """
    Applies extract_features_single_signal to a full batch of signals.

    Parameters
    ----------
    X_time : np.ndarray, shape (n_samples, N_SAMPLES)

    Returns
    -------
    X_features : np.ndarray, shape (n_samples, n_features)
        Feature matrix in the same order as FEATURE_NAMES.
    """
    rows = []
    for signal in X_time:
        feats = extract_features_single_signal(signal)
        rows.append([feats[name] for name in FEATURE_NAMES])
    return np.array(rows)


if __name__ == "__main__":
    from generate_signals import generate_dataset

    X, y, meta = generate_dataset(n_per_class=5)
    X_feat = extract_features(X)
    print("Features shape:", X_feat.shape)
    print("Names:", FEATURE_NAMES)
    print("First 3 rows:\n", X_feat[:3])