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
from notebooks.setup import data_setup

ds, mw, stellar_model, imf, planet_model = data_setup()

#%%%

da = mw.sphere(radius=ds.quan(30,"kpc"))
vec = da.quantities.angular_momentum_vector()

new_ds = rotated_dataset(da, mw.centre(), vec, [('stars','planets')])
    