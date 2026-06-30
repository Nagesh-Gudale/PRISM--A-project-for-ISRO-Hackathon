# AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

An AI-powered pipeline for robust exoplanet detection from noisy TESS light curves using **noise fingerprinting**, **physics-aware feature engineering**, **multimodal deep learning**, and **explainable AI**.

> 🚀 Developed as part of the ISRO Hackathon.

---

## Overview

Detecting exoplanets from astronomical light curves is challenging due to stellar variability, instrumental systematics, flares, eclipsing binaries, and other sources of noise that often lead to false positives.

This project proposes an end-to-end machine learning pipeline that automatically prepares training data, characterizes stellar noise, extracts astrophysically meaningful features, and classifies exoplanet candidates with improved robustness and interpretability.

---

## Key Features

- Automated TESS light curve acquisition
- Automatic dataset generation using TOI cross-matching
- Noise fingerprinting before classification
- Physics-aware feature extraction
- Multi-view CNN architecture
- Stellar parameter fusion (TIC metadata)
- Explainable AI using SHAP
- Diagnostic plots and confidence estimation
- PDF and CSV report generation

---

## Proposed Pipeline

```
TESS Light Curves
        │
        ▼
Data Validation & Preprocessing
        │
        ▼
Transit Detection (BLS)
        │
        ▼
Candidate Extraction
        │
        ▼
Noise Fingerprinting
        │
        ▼
Physics-aware Feature Extraction
        │
        ▼
Multi-view CNN + Stellar Parameter Fusion
        │
        ▼
Planet Candidate Classification
        │
        ▼
Explainability & Report Generation
```

---

## Technologies Used

### Programming Language

- Python

### Astronomy Libraries

- Lightkurve
- Astroquery
- Astropy

### Machine Learning

- TensorFlow
- PyTorch
- Scikit-learn

### Scientific Computing

- NumPy
- SciPy
- Pandas

### Visualization

- Matplotlib
- Plotly
- SHAP

---

## Repository Structure

```
project/
│
├── data/
├── notebooks/
├── preprocessing/
├── feature_extraction/
├── noise_fingerprinting/
├── models/
├── evaluation/
├── reports/
├── utils/
├── requirements.txt
└── README.md
```

---

## Current Status

🚧 **Prototype Under Active Development**

Current progress includes:

- Dataset preparation pipeline
- Model training
- Noise fingerprinting research
- Architecture implementation

Upcoming work:

- Model evaluation
- Explainability integration
- User interface
- Performance benchmarking

---

## Future Enhancements

- Transformer-based models
- Self-supervised representation learning
- Multi-mission support (Kepler, PLATO)
- Real-time inference
- Cloud deployment
- Web dashboard

---

## Team

Developed for the **ISRO Hackathon**.

---

## Disclaimer

This repository is currently under active development. Features, architecture, and implementation details may evolve as the project progresses.
