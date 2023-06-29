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
from gallifrey.utilities.math import calculate_pca
from notebooks.setup import data_setup

ds, mw, stellar_model, imf, planet_model = data_setup()

#%%
radius = 60
normal_vector = calculate_pca(
    mw.sphere(radius=(10, "kpc"))["stars", "Coordinates"]
).components_[-1]

disk_data = rotated_dataset(
    mw.disk(
        radius=ds.quan(radius, "kpc"), height=ds.quan(0.5, "kpc"), normal=normal_vector
    ),
    mw.centre(),
    normal_vector,
    [
        ("stars", "planets"),
        ("stars", "main_sequence_stars"),
    ],
)
#%%

lifetime = stellar_model.lifetime(0.8) # basically all stars

mask_age = ds.r["stars","stellar_age"] <= lifetime

metallicities = ds.r["stars","[Fe/H]"][mask_age]