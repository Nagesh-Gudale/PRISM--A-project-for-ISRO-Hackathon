# AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

An AI-powered pipeline for robust exoplanet detection from noisy TESS light curves using **noise fingerprinting**, **physics-aware feature engineering**, and **deep learning**.

> 🚀 Developed for the ISRO Hackathon.

---

## Overview

This project aims to improve exoplanet detection by combining astronomical signal processing with machine learning. The pipeline automates data collection from TESS, preprocesses light curves, trains deep learning models, and performs inference while reducing false positives caused by stellar noise.

---

## Features

- Automated TESS light curve collection
- Automatic dataset generation
- Data preprocessing and normalization
- Deep learning model training
- Exoplanet candidate inference
- Visualization utilities
- Model checkpoint saving
- Local execution without cloud dependency

---

## Repository Structure

```text
isro_v2/
│
├── dataset/
│   ├── lightcurves/
│   ├── balanced_training_dataset.csv
│   └── labeled_toi_dataset.csv
│
├── results/
│
├── best_model.keras
│
├── data_collector.py      # Downloads and prepares TESS data
├── preprocess.py          # Data preprocessing
├── model.py               # Neural network architecture
├── train.py               # Model training
├── inference.py           # Prediction on new light curves
├── visualize.py           # Plotting and visualization
└── main.py                # Main pipeline entry point
```

---

## Workflow

```
Collect TESS Data
        │
        ▼
Preprocess Light Curves
        │
        ▼
Generate Training Dataset
        │
        ▼
Train Deep Learning Model
        │
        ▼
Predict Exoplanet Candidates
        │
        ▼
Visualize Results
```

---

## Requirements

- Python 3.10+
- TensorFlow
- NumPy
- Pandas
- SciPy
- Lightkurve
- Astroquery
- Matplotlib
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Collect dataset

```bash
python data_collector.py
```

Train the model

```bash
python train.py
```

Run inference

```bash
python inference.py
```

Launch the complete pipeline

```bash
python main.py
```

---

## Current Status

🚧 Prototype Under Active Development

Current implementation includes:

- Dataset generation
- Model training
- Inference pipeline
- Visualization tools

Upcoming work:

- Noise fingerprinting module
- Stellar parameter fusion
- Explainable AI (SHAP)
- Performance benchmarking
- Interactive dashboard

---

## Technologies

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- SciPy
- Lightkurve
- Astroquery
- Matplotlib
- Pandas

---

## License

This repository is intended for research and educational purposes as part of the ISRO Hackathon.
