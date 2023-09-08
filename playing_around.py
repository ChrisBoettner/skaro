#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:55:06 2023

@author: chris
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yt

from gallifrey.particles import rotated_dataset
from gallifrey.setup import data_setup

#%%
num_embryos = 50
host_star_mass = (0.7, 1)
ds, mw, stellar_model, imf, planet_model = data_setup(ngpps_num_embryos=num_embryos,
                                                      ngpps_star_masses=host_star_mass)
