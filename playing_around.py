#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:55:06 2023

@author: chris
"""

import matplotlib.pyplot as plt
import numpy as np
import yt

from gallifrey.particles import rotated_dataset
from gallifrey.setup import data_setup
from gallifrey.utilities.math import calculate_pca

ds, mw, stellar_model, imf, planet_model = data_setup()

#%%

lifetime = stellar_model.lifetime(0.8) # basically all stars

mask_age = ds.r["stars","stellar_age"] <= lifetime

metallicities = ds.r["stars","[Fe/H]"][mask_age]