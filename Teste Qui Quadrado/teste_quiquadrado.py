#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:26:02 2018

@author: anarocha
"""
from scipy.stats import chisquare
import pandas as pd
# scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)[source]
#f_obs : array_like
#    Observed frequencies in each category.
#f_exp : array_like, optional
#    Expected frequencies in each category. By default the categories are assumed to be equally likely.

# =============================================================================
# 2017-2018
# =============================================================================
data=pd.read_csv('quiquadrado_dataset_TRT13_2G_2017-2018.csv', sep=';',index_col=False)    
chisquare(data['Observado'],data['Esperado'])
#Power_divergenceResult(statistic=107.05153616069705, pvalue=5.615716058144861e-09)

# =============================================================================
# 2016-2017
# =============================================================================
data=pd.read_csv('quiquadrado_dataset_TRT13_2G_2016-2017.csv', sep=';',index_col=False)    
chisquare(data['Observado'],data['Esperado'])
#Power_divergenceResult(statistic=29.63399139503627, pvalue=0.764171604286429)

# =============================================================================
# 2016-2018
# =============================================================================
data=pd.read_csv('quiquadrado_dataset_TRT13_2G_2016-2018.csv', sep=';',index_col=False)    
chisquare(data['Observado'],data['Esperado'])
#Power_divergenceResult(statistic=49.08150558237491, pvalue=0.07170722181749253)