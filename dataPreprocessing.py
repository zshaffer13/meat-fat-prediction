# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:41:22 2025

@author: zshaf
"""

import pandas as pd
from scipy.io import arff
import ramanspy as rp
import numpy as np

def dataPreprocess(dataLoc: str = './data/tecator.arff'):

    data = arff.loadarff(dataLoc)
    df = pd.DataFrame(data[0])

    spectral_data = df.iloc[:,0:100]
    pca_data = df.iloc[:,100:122]
    meta_data = df.iloc[:,122:]

    pipe = rp.preprocessing.Pipeline([rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
                                      rp.preprocessing.baseline.ASLS(),
                                      rp.preprocessing.normalise.MinMax(pixelwise = True)])

    spectral_axis = np.linspace(9523.8,11764.7,100)

    raman_spectrum = rp.Spectrum(spectral_data,spectral_axis)

    preprocessed_spectra = pipe.apply(raman_spectrum)

    
    return preprocessed_spectra, meta_data, pca_data, spectral_data, spectral_axis

