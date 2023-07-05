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

print("check if the rescaling is correct, maybe compare actual values with results in paper")
print("currently m_g prop m_star, r_in prop m_star^1/3, [Fe/H] and mwind const")
print("reasoning in paper, write down. actually make notes while reading all papers")

print("check if imf integration works")
#%%
category =  "Earth"
host_star_mass = 1
samples = int(1e5)

population_id = planet_model.get_population_id(num_embryos, host_star_mass)
systems = planet_model.get_systems(population_id)

variables = pd.DataFrame(
    np.linspace(*systems.bounds["[Fe/H]"], samples), columns=["[Fe/H]"]
)
variables["age"] = int(1e9)
result = planet_model.prediction(category, variables, host_star_mass = host_star_mass, 
                                 return_full=True)

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
original_variables = systems.variables.reset_index(drop=True)
original_variables["age"] = int(1e8)
original_sample = planet_model.prediction(
    categories, original_variables, host_star_mass = host_star_mass, 
    return_full=True, neighbors=1
)

corr_matrix = original_sample.corr(method="kendall").drop(
    columns=planet_model.features, index=["age", *categories]
)

sns.pairplot(
    original_sample.drop(columns=["age", *[c for c in categories if c != "Earth"]]),
    hue="Earth")

plt.figure()
sns.heatmap(corr_matrix, vmin=-1, vmax=1, square=True, annot=True, cmap="vlag")

#%%
print("remember ridge plots, and other sns plots")