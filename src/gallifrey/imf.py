#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:19:27 2023

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.logspace(-2,2,int(1e+7))

from scipy.stats import rv_continuous

class Chabrier(rv_continuous):
    def __init__(self, slope= 1.35, normalisation= 0.4196249611479172,
                 mean=0.2, variance = 0.6,
                 lower_mass_limit = 0.08, upper_mass_limit=100,):
        
        super().__init__(a=lower_mass_limit,
                          b=upper_mass_limit)  # set domain boundaries
        
        self.slope = slope
        self.normalisation = normalisation
        self.mean = mean
        self.variance = variance
        
        self.constant = (self.normalisation 
                          * np.exp((np.log10(self.mean)**2)/self.variance))
        self.ln10    = np.log(10)
        
        
    def _pdf(self, m):
        m   = np.asarray(m)
        pdf = np.empty_like(m) 
        
        mask = m>1
        not_mask = ~mask
        
        pdf[mask]  = self.normalisation/self.ln10 * m[mask]**(-(self.slope+1))
        pdf[not_mask] = (self.constant/self.ln10 * 1/m[not_mask]
                      * np.exp(-(np.log10(m[not_mask]/self.mean)**2)/self.variance))
        #breakpoint()
        
        return(pdf)
            
            
        
from scipy.special import erf
from scipy.stats import lognorm

o = Chabrier()

mu = 0.2
sigma = np.sqrt(np.log(10)*0.3)

k = lognorm(s=sigma, scale = mu)

# calculate cdf
# check again
# add type hints, comments etc