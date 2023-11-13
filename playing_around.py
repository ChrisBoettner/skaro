#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:55:06 2023

@author: chris
"""
# flake8: noqa

# %%
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yt

from gallifrey.model import Model

resolution = 8192
sim_id = "37_11"
num_embryos = 50
host_star_mass = 1

model = Model(
    resolution=resolution,
    sim_id=sim_id,
    ngpps_num_embryos=num_embryos,
    ngpps_star_masses=host_star_mass,
)

# %%
