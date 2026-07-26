"""
generate_signals.py

Generates synthetic signals for the classification problem:

1. "clean"  -> pure sinusoid (variable frequency and amplitude)
2. "noisy"  -> sinusoid + gaussian noise (variable SNR)
3. "noise"  -> pure noise (no dominant periodic component)

Each signal is generated in the time domain, sampled at a fixed rate
over a fixed duration, and returned together with its label.

Typical usage:
    from generate_signals import generate_dataset
    X_time, y, meta = generate_dataset(n_per_class=200)
"""

import numpy as np

# ---------------------------------------------------------------
# Global sampling parameters (adjustable as needed)
# ---------------------------------------------------------------
FS = 500.0          # sampling frequency (Hz)
DURATION = 1.0       # duration of each signal (seconds)
N_SAMPLES = int(FS * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

# Range of physical parameters to vary across samples
FREQ_MIN, FREQ_MAX = 5.0, 80.0     # Hz, must respect Nyquist (< FS/2)
AMP_MIN, AMP_MAX = 0.5, 2.0
SNR_MIN_DB, SNR_MAX_DB = -3.0, 15.0  # SNR range for the "noisy" class


def _gaussian_noise(n, sigma, rng):
    return rng.normal(loc=0.0, scale=sigma, size=n)


def generate_clean_signal(rng):
    """Pure sinusoid: random frequency and amplitude, negligible noise."""
    freq = rng.uniform(FREQ_MIN, FREQ_MAX)
    amp = rng.uniform(AMP_MIN, AMP_MAX)
    phase = rng.uniform(0, 2 * np.pi)
    signal = amp * np.sin(2 * np.pi * freq * T + phase)
    # minimal noise floor (every real instrument has some noise)
    signal += _gaussian_noise(N_SAMPLES, sigma=0.01, rng=rng)
    return signal, {"frequency": freq, "amplitude": amp, "snr_db": None}


def generate_noisy_signal(rng):
    """Sinusoid + gaussian noise with controlled SNR."""
    freq = rng.uniform(FREQ_MIN, FREQ_MAX)
    amp = rng.uniform(AMP_MIN, AMP_MAX)
    phase = rng.uniform(0, 2 * np.pi)
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB)

    pure_signal = amp * np.sin(2 * np.pi * freq * T + phase)
    signal_power = np.mean(pure_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_sigma = np.sqrt(noise_power)

    signal = pure_signal + _gaussian_noise(N_SAMPLES, sigma=noise_sigma, rng=rng)
    return signal, {"frequency": freq, "amplitude": amp, "snr_db": snr_db}


def generate_pure_noise(rng):
    """Pure gaussian noise, with no dominant periodic component."""
    sigma = rng.uniform(0.5, 1.5)
    signal = _gaussian_noise(N_SAMPLES, sigma=sigma, rng=rng)
    return signal, {"frequency": None, "amplitude": None, "snr_db": None}


GENERATORS = {
    "clean": generate_clean_signal,
    "noisy": generate_noisy_signal,
    "noise": generate_pure_noise,
}


def generate_dataset(n_per_class=200, seed=42, classes=None):
    """
    Generates a balanced dataset of synthetic signals.

    Parameters
    ----------
    n_per_class : int
        Number of samples to generate per class.
    seed : int
        Random seed for reproducibility.
    classes : list[str] or None
        Subset of classes to generate. Defaults to all three:
        ["clean", "noisy", "noise"].

    Returns
    -------
    X_time : np.ndarray, shape (n_total, N_SAMPLES)
        Signals in the time domain.
    y : np.ndarray, shape (n_total,)
        Class labels (string).
    metadata : list[dict]
        Physical parameters used to generate each signal (useful for
        debugging and understanding how hard each sample is).
    """
    rng = np.random.default_rng(seed)
    classes = classes or list(GENERATORS.keys())

    X_time, y, metadata = [], [], []
    for cls in classes:
        generator = GENERATORS[cls]
        for _ in range(n_per_class):
            signal, meta = generator(rng)
            X_time.append(signal)
            y.append(cls)
            metadata.append(meta)

    return np.array(X_time), np.array(y), metadata


if __name__ == "__main__":
    X, y, meta = generate_dataset(n_per_class=5)
    print(f"Generated dataset: {X.shape[0]} signals of {X.shape[1]} samples each")
    print("Classes:", np.unique(y, return_counts=True))