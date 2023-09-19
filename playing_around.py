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

import pathlib
import sys
import os

sys.path.append(str(pathlib.Path(os.getcwd()).parent.joinpath("src")))

from gallifrey.particles import rotated_dataset
from gallifrey.setup import data_setup

#%%
num_embryos = 50
host_star_mass = (0.7, 1)
ds, mw, stellar_model, imf, planet_model = data_setup(ngpps_num_embryos=num_embryos,
                                                      ngpps_star_masses=host_star_mass)

#%%
#mw_ids = pd.read_csv("test_mw_ids")

#%%
import h5py
import pynbody
from gallifrey.data.paths import Path

path = str(Path.raw_data("snapdir_127/snapshot_127"))

#s = pynbody.load(path)

#%%

#centre = mw.centre().to("code_length").value
#radius = ds.quan(25, "kpc").to("code_length").value

#sphere = s[pynbody.filt.Sphere(radius, centre)]

from gallifrey.decomposition.mordor import galaxy_components

galaxy_components(path, mw, radius=ds.quan(1, "kpc"))