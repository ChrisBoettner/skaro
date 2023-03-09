#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:34:02 2023

@author: chris
"""
import pathlib

#%%
# load local version before pip installed version, for debugging
import sys

sys.path.append(pathlib.Path(__file__).parent.joinpath("src"))

#%%
import yt

from gallifrey.data.load import load_snapshot
from gallifrey.halo import MainHalo
from gallifrey.utilities.time import Timer
from gallifrey.visualization.manager import DefaultFigureManager as fm

#%%
with Timer("load"):
    ds = load_snapshot(127, 4096)
    mw = MainHalo("MW", 4096, ds)
    
#%%
with Timer('add'):
    from gallifrey.fields import Fields
    from gallifrey.filter import Filter
    
    filters = Filter(ds)
    filters.add_stars()
    
    fields = Fields(ds)
    fields.add_stellar_age()