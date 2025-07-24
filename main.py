# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:31:19 2025

@author: zshaf
"""
from dataPreprocessing import dataPreprocess
from dataPlotting import plot_spectra_by_target,plot_mean_spectra_by_group
from modelTraining import (
    run_lazy_regression,
    train_ridge_regression,
    train_pls_regression,
    train_random_forest
)
from plotting import plot_actual_vs_predicted, plot_top_lazy_models




def main():
    dataLoc = './data/tecator.arff'
    
    #Prepare data
    preprocessed_spectra, meta_data, pca_data, spectral_data, spectral_axis = dataPreprocess(dataLoc)
    
    #Plot Raw data
    plot_spectra_by_target(spectral_data=spectral_data,
                       spectral_axis=spectral_axis,
                       meta_data=meta_data,
                       target='fat')    
    #Plot Preprocessed data
    plot_spectra_by_target(spectral_data=preprocessed_spectra,
                       spectral_axis=preprocessed_spectra.spectral_axis,
                       meta_data=meta_data,
                       target='fat')
    
    #Plot Raw data
    plot_mean_spectra_by_group(spectral_data=spectral_data,
                       spectral_axis=spectral_axis,
                       meta_data=meta_data,
                       target='fat')    
    #Plot Preprocessed data
    plot_mean_spectra_by_group(spectral_data=preprocessed_spectra,
                       spectral_axis=preprocessed_spectra.spectral_axis,
                       meta_data=meta_data,
                       target='fat')
    
    print('hold')
    # Assume preprocessed_spectra and meta_data are defined already
    models, preds = run_lazy_regression(preprocessed_spectra, meta_data, target_col='fat',predictions=True)
    
    # Extract y_test manually (since run_lazy_regression doesn't return it yet)
    y_test = meta_data.iloc[172:215]['fat']

    plot_top_lazy_models(y_true=y_test,
                         models_df=models,
                         predictions_dict=preds,
                         top_n=3,
                         target_label='Fat Content (%)')
    
    ridge_results = train_ridge_regression(preprocessed_spectra, meta_data, target_col='fat')
    
    
    plot_actual_vs_predicted(y_true=ridge_results['y_true'],
                             y_pred=ridge_results['y_pred'],
                             model_name=ridge_results['model'],
                             target_label='Fat Content (%)')
    
    pls_results = train_pls_regression(preprocessed_spectra, meta_data, target_col='fat', n_components=10)
    
    plot_actual_vs_predicted(y_true=pls_results['y_true'],
                             y_pred=pls_results['y_pred'],
                             model_name=pls_results['model'],
                             target_label='Fat Content (%)')
    
    rf_results = train_random_forest(preprocessed_spectra, meta_data, target_col='fat')
    
    plot_actual_vs_predicted(y_true=rf_results['y_true'],
                             y_pred=rf_results['y_pred'],
                             model_name=rf_results['model'],
                             target_label='Fat Content (%)')
    
if __name__ == "__main__":
    main()
    