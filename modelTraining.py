# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:31:10 2025

@author: zshaf
"""

import numpy as np
import pandas as pd
import ramanspy as rp
from sklearn.utils import shuffle
from lazypredict.Supervised import LazyRegressor


#Use samples 1-172 for training
#173-215 for testing
#Last 25 for further extrapolation

train_data = preprocessed_spectra[:172,:]
test_data = preprocessed_spectra[172:215,:]
ext_1_data = preprocessed_spectra[215:223,:]
ext_2_data = preprocessed_spectra[223:,:]

y_train = meta_data.iloc[:172,1]
y_test = meta_data.iloc[172:215,1]
ext_1_y = meta_data.iloc[215:223,1]
ext_2_y = meta_data.iloc[223:,1]

X_train, y_train = shuffle(train_data.flat.spectral_data,y_train)

clf = LazyRegressor()
models_test, predictions_test = clf.fit(X_train, test_data.spectral_data, y_train, y_test)