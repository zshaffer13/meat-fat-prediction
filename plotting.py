# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:07:44 2025

@author: zshaf
"""

# plotting.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Union
import pandas as pd

def plot_actual_vs_predicted(y_true: Union[np.ndarray, pd.Series],
                              y_pred: Union[np.ndarray, pd.Series],
                              model_name: str = "",
                              target_label: str = "Target",
                              show_identity_line: bool = True,
                              figsize=(8, 6),
                              save_path: str = None):
    """
    Plot Actual vs Predicted values for a regression model.

    Parameters:
    - y_true: array-like of true target values
    - y_pred: array-like of predicted target values
    - model_name: name of the model (used in title)
    - target_label: name of the target variable
    - show_identity_line: if True, plots y = x line
    - figsize: tuple for figure size
    - save_path: if set, saves plot to this path
    """

    plt.figure(figsize=figsize)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.8)
    if show_identity_line:
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='Ideal Fit')

    plt.xlabel(f"Actual {target_label}")
    plt.ylabel(f"Predicted {target_label}")
    title = f"Actual vs Predicted - {model_name}" if model_name else "Actual vs Predicted"
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_top_lazy_models(y_true, models_df, predictions_dict, top_n=3, target_label="Fat Content (%)"):
    """
    Automatically plots Actual vs Predicted for top N models using predictions dataframe.

    Parameters:
    - y_true: ground truth values
    - models_df: output from LazyRegressor (model scores)
    - predictions_df: DataFrame where each column = model's predictions
    - top_n: number of top models to visualize
    """

    top_models = models_df.sort_values(by='R-Squared', ascending=False).head(top_n).index.tolist()

    for model_name in top_models:
        if model_name in predictions_dict.columns:
            y_pred = predictions_dict[model_name]
            print(f"\nPlotting {model_name} (LazyRegressor prediction)")
            y_true = pd.Series(y_true).reset_index(drop=True)
            y_pred = pd.Series(y_pred).reset_index(drop=True)
            y_true.name = None
            y_pred.name = None
            plot_actual_vs_predicted(
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                target_label=target_label
            )
        else:
            print(f"⚠️ No predictions available for: {model_name}")
