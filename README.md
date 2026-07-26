# Simple Signal / Spectrum Classifier

A data science project applied to experimental physics: generation of
synthetic signals (clean sinusoids, noisy sinusoids, and pure noise),
extraction of physically meaningful features via FFT, and training of
simple classifiers (`scikit-learn`) to distinguish between the three
types.

## Motivation

In a real experiment, distinguishing a genuine periodic signal from
instrumental noise is a recurring problem (spectral peak detection,
data filtering, sensor quality control). This project reproduces that
problem at a small, fully reproducible scale, using Fourier analysis
as a direct bridge into feature engineering for machine learning.

## Problem classes

| Class   | Description                                                   |
|---------|-----------------------------------------------------------------|
| `clean` | Pure sinusoid, random frequency and amplitude, negligible noise |
| `noisy` | Sinusoid + gaussian noise, with variable SNR (-3 to 15 dB)       |
| `noise` | Pure gaussian noise, no dominant periodic component              |

## Repository structure

```
signal-classifier/
├── README.md
├── requirements.txt
├── notebooks/
│ └── 01_signal_classifier.ipynb # Exploration and results
├── src/
│ ├── generate_signals.py # Synthetic signal generation
│ ├── extract_features.py # FFT-based feature extraction
│ └── model.py # Training and evaluation
├── data/ # (optional) exported signals
└── figures/ # Plots for the README/notebook
```
## Extracted features

From each signal's Fourier spectrum (`scipy.fft`), we compute:

- **dominant_freq**: frequency of the highest spectral peak
- **dominant_amp**: magnitude of that peak
- **variance**: variance of the signal in the time domain
- **total_energy**: total energy of the spectrum
- **estimated_snr**: ratio between the dominant peak's energy and the rest of the spectrum
- **bandwidth**: energy spread around the dominant peak

## Models

- Logistic regression (interpretable baseline)
- Decision tree (allows inspecting feature importance)

## How to run the project

```bash
pip install -r requirements.txt
python src/model.py          # trains and prints metrics to console
jupyter notebook notebooks/01_signal_classifier.ipynb
```

## Preliminary results

With 200 samples per class, both models perfectly distinguish pure
`noise` from the other two classes. The main confusion happens
between `clean` and `noisy` at intermediate SNR (~5-10 dB), which is
physically expected: the higher the SNR, a "noisy" signal increasingly
resembles a "clean" one. The most relevant feature for the decision
tree is `estimated_snr`.

## Next steps

- [ ] Add cross-validation instead of a single train/test split
- [ ] Explore a fourth signal type (e.g. chirp or dual-peak spectrum)
- [ ] Add unit tests for the feature extraction functions
- [ ] ROC curve / error analysis by SNR range
