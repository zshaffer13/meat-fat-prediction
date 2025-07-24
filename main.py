# -*- coding: utf-8 -*-
"""
Main script to run the Tecator spectral regression pipeline.
Performs preprocessing, plotting, model training (default and tuned), and evaluation.
"""

from dataPreprocessing import dataPreprocess
from dataPlotting import plot_spectra_by_target, plot_mean_spectra_by_group
from modelTraining import (
    run_lazy_regression,
    train_ridge_regression,
    train_pls_regression,
    train_random_forest,
    train_ridge_with_gridsearch,
    train_pls_with_gridsearch,
    train_rf_with_gridsearch
)
from plotting import plot_actual_vs_predicted, plot_top_lazy_models


def run_all_plots(preprocessed_spectra, spectral_data, spectral_axis, meta_data, target='fat'):
    """Run all target and mean plots for raw and preprocessed spectra."""
    plot_spectra_by_target(spectral_data=spectral_data, spectral_axis=spectral_axis,
                           meta_data=meta_data, target=target)

    plot_spectra_by_target(spectral_data=preprocessed_spectra,
                           spectral_axis=preprocessed_spectra.spectral_axis,
                           meta_data=meta_data, target=target)

    plot_mean_spectra_by_group(spectral_data=spectral_data, spectral_axis=spectral_axis,
                               meta_data=meta_data, target=target)

    plot_mean_spectra_by_group(spectral_data=preprocessed_spectra,
                               spectral_axis=preprocessed_spectra.spectral_axis,
                               meta_data=meta_data, target=target)


def run_all_models(preprocessed_spectra, meta_data, target='fat'):
    """Train and evaluate all models and plot predictions."""
    # LazyPredict
    models, preds = run_lazy_regression(preprocessed_spectra, meta_data, target_col=target, predictions=True)
    y_test = meta_data.iloc[172:215][target]
    plot_top_lazy_models(y_true=y_test, models_df=models, predictions_dict=preds, top_n=3, target_label='Fat Content (%)')

    # Standard Models
    for train_func in [train_ridge_regression, train_pls_regression, train_random_forest]:
        results = train_func(preprocessed_spectra, meta_data, target_col=target)
        plot_actual_vs_predicted(y_true=results['y_true'], y_pred=results['y_pred'],
                                 model_name=results['model'], target_label='Fat Content (%)')

    # Tuned Models
    for train_func in [train_ridge_with_gridsearch, train_pls_with_gridsearch, train_rf_with_gridsearch]:
        results = train_func(preprocessed_spectra, meta_data, target_col=target)
        plot_actual_vs_predicted(y_true=results['y_true'], y_pred=results['y_pred'],
                                 model_name=results['model'], target_label='Fat Content (%)')


def main():
    dataLoc = './data/tecator.arff'
    preprocessed_spectra, meta_data, pca_data, spectral_data, spectral_axis = dataPreprocess(dataLoc)

    run_all_plots(preprocessed_spectra, spectral_data, spectral_axis, meta_data, target='fat')
    run_all_models(preprocessed_spectra, meta_data, target='fat')


if __name__ == "__main__":
    main()
