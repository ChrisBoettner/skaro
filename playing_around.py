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
from gallifrey.utilities.math import calculate_pca

ds, mw, stellar_model, imf, planet_model = data_setup()

#%%
category =  "Earth"
samples = int(1e5)

variables = pd.DataFrame(
    np.linspace(*planet_model.systems.bounds["[Fe/H]"], samples), columns=["[Fe/H]"]
)
variables["age"] = int(1e9)
result = planet_model.prediction(category, variables, return_full=True)

result["planets_binned"] = pd.cut(result[category], bins=4)
sns.pairplot(
    result.drop(columns=["age", category]), hue="planets_binned", kind="hist"
)

#%%
print("make original data analysis in own notebook")

categories = [
    category
    for category in planet_model.categories
    if category not in ["Dwarf", "D-Burner"]
]
original_variables = planet_model.systems.variables.reset_index(drop=True)
original_variables["age"] = int(1e8)
original_sample = planet_model.prediction(
    categories, original_variables, return_full=True, neighbors=1
)

corr_matrix = original_sample.corr(method="kendall").drop(
    columns=planet_model.features, index=["age", *categories]
)

plt.figure()
sns.pairplot(
    original_sample.drop(columns=["age", *[c for c in categories if c != "Earth"]]),
    hue="Earth")

plt.figure()
sns.heatmap(corr_matrix, vmin=-1, vmax=1, square=True, annot=True, cmap="vlag")

#%%
print("remember ridge plots, and other sns plots")