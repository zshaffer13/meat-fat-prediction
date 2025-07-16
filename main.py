# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:31:19 2025

@author: zshaf
"""
from dataPreprocessing import dataPreprocess
from dataPlotting import plot_spectra_by_target,plot_mean_spectra_by_group



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
    
    
if __name__ == "__main__":
    main()
    