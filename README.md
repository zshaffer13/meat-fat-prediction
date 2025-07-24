# Tecator Spectral Regression

This project builds a machine learning pipeline to predict fat content in meat samples using near-infrared (NIR) spectra from the classic Tecator dataset. It includes preprocessing, exploratory visualization, model training (including LazyRegressor and custom regressors), and performance evaluation with actual vs. predicted plots.

---

## 📊 Dataset Overview

- **Source**: Tecator meat samples dataset (from `.arff` file)
- **Features**: Absorbance spectra from 100 wavelengths (850–1050 nm)
- **Targets**: 
  - `fat` (primary)
  - `moisture`, `protein` (optional)
- **Samples**:
  - 172 for training
  - 43 for testing
  - 25 held out for extrapolation analysis

---

## 🛠️ Pipeline Components

### 🔹 Data Preprocessing (`dataPreprocessing.py`)
- **Smoothing**: Savitzky-Golay (window=9, polyorder=3)
- **Baseline Correction**: Asymmetric Least Squares (ASLS)
- **Normalization**: Min-Max scaling to [0, 1]
- **Output**: Raw and preprocessed `ramanspy.Spectrum` objects

### 🔹 Visualization (`dataPlotting.py`, `plotting.py`)
- Spectra plots colored by fat content
- Mean spectra per fat group
- Actual vs. predicted scatter plots for multiple models

### 🔹 Modeling (`modelTraining.py`)
- **Quick benchmark** with `LazyRegressor` (40+ models)
- Custom regressors:
  - RidgeCV
  - PLSRegression
  - Random Forest
- Automatic plotting of top models by R²

---

## 🚀 How to Run

```bash
# Ensure dependencies
pip install -r requirements.txt

# Run the main pipeline
python main.py
