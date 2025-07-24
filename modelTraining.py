# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:31:10 2025

@author: zshaf
"""

# modelTraining.py

import numpy as np
import pandas as pd
import ramanspy as rp
from sklearn.utils import shuffle
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Use samples 1-172 for training
#173-215 for testing
#Last 25 for further extrapolation


def run_lazy_regression(preprocessed_spectra: rp.Spectrum, meta_data: pd.DataFrame,
                        target_col: str = 'fat', verbose: bool = True,predictions=True):
    train_data = preprocessed_spectra[:172, :]
    test_data = preprocessed_spectra[172:215, :]

    y_train = meta_data.iloc[:172][target_col]
    y_test = meta_data.iloc[172:215][target_col]

    X_train, y_train = shuffle(train_data.flat.spectral_data, y_train)

    clf = LazyRegressor(verbose=verbose, ignore_warnings=True, custom_metric=None,predictions=True)
    models_test, predictions_test = clf.fit(X_train, test_data.spectral_data, y_train, y_test)

    return models_test, predictions_test


def train_ridge_regression(preprocessed_spectra: rp.Spectrum, meta_data: pd.DataFrame,
                           target_col: str = 'fat'):
    train_data = preprocessed_spectra[:172, :]
    test_data = preprocessed_spectra[172:215, :]

    y_train = meta_data.iloc[:172][target_col]
    y_test = meta_data.iloc[172:215][target_col]

    X_train, y_train = shuffle(train_data.flat.spectral_data, y_train)
    X_test = test_data.spectral_data

    model = RidgeCV(alphas=np.logspace(-3, 3, 50))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return _report_results("RidgeCV", y_test, y_pred)


def train_pls_regression(preprocessed_spectra: rp.Spectrum, meta_data: pd.DataFrame,
                         target_col: str = 'fat', n_components: int = 10):
    train_data = preprocessed_spectra[:172, :]
    test_data = preprocessed_spectra[172:215, :]

    y_train = meta_data.iloc[:172][target_col]
    y_test = meta_data.iloc[172:215][target_col]

    X_train, y_train = shuffle(train_data.flat.spectral_data, y_train)
    X_test = test_data.spectral_data

    model = PLSRegression(n_components=n_components)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return _report_results("PLSRegression", y_test, y_pred)


def train_random_forest(preprocessed_spectra: rp.Spectrum, meta_data: pd.DataFrame,
                        target_col: str = 'fat', n_estimators: int = 100):
    train_data = preprocessed_spectra[:172, :]
    test_data = preprocessed_spectra[172:215, :]

    y_train = meta_data.iloc[:172][target_col]
    y_test = meta_data.iloc[172:215][target_col]

    X_train, y_train = shuffle(train_data.flat.spectral_data, y_train)
    X_test = test_data.spectral_data

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return _report_results("RandomForestRegressor", y_test, y_pred)


def _report_results(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Manually compute RMSE
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")

    return {
        "model": model_name,
        "r2": r2,
        "rmse": rmse,
        "y_true": y_true,
        "y_pred": y_pred
    }

