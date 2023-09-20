#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:55:06 2023

@author: chris
"""

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yt

sys.path.append(str(pathlib.Path(os.getcwd()).joinpath("src")))

from gallifrey.particles import rotated_dataset
from gallifrey.setup import data_setup

#%%
num_embryos = 50
host_star_mass = (0.7, 1)
ds, mw, stellar_model, imf, planet_model, path = data_setup(ngpps_num_embryos=num_embryos,
                                                            ngpps_star_masses=host_star_mass)

#%%
# f_path = path + "/snapshot_127" 

# from gallifrey.decomposition.mordor import galaxy_components

# k = galaxy_components(mw, f_path)

# #%%
# sphere = mw.sphere(radius=0.1*mw.virial_radius())

# #%%

# def create_component_mask(galaxy_compotent_dataframe, star_particle_IDs, component):
#     component_ids = galaxy_compotent_dataframe["ParticleIDs"][
#         galaxy_compotent_dataframe["Component"]==component].to_numpy()
    
#     return np.isin(star_particle_IDs, component_ids, assume_unique=True)

# t = k["ParticleIDs"][k["Component"]==1].to_numpy()

# o = sphere["stars", "ParticleIDs"].astype(int)

#r = np.isin(o, t)