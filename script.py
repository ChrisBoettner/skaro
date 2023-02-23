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
# import numpy as np
# o = ds.all_data()
# t = [380726040, 357706557]
# a = np.in1d(o['PartType0', 'ParticleIDs'].value, t, assume_unique=True)

#%%
# @yt.particle_filter(requires=["ParticleIDs"], filtered_type="PartType0")
# def fil(pfilter, data):
#     filter = data[(pfilter.filtered_type, "ParticleIDs")] == t[0]
#     return filter

# ds.add_particle_filter("fil")

# for field in ds.derived_field_list:
#     if field[0] == "fil":
#         print(field)
        
# def create_ID_filter(ID_list, filtered_type="PartType0"):
    
#     @yt.particle_filter(requires=["ParticleIDs"], filtered_type=filtered_type)
#     def fil(pfilter, data):
#         filter = np.in1d(data[pfilter.filtered_type, 'ParticleIDs'].value, ID_list,
#                           assume_unique=True)
#         #filter = data[(pfilter.filtered_type, "ParticleIDs")] == t[0]
#         return filter
    
#     return fil

# f = create_ID_filter(t)
# ds.add_particle_filter("fil")

# for field in ds.derived_field_list:
#     if field[0] == "fil":
#         print(field)
        
        
# to do:
#     create filter and regions sub module of halo
#     implement box, disk for regions
#     add ds as attribute to halo, star, gas class
#     impelemt add star_filter, gas_filter for filter
    
    
#     general form for filter:
    
#     def add_star_filter(self, ID_list, ):
#         @yt.particle_filter(requires=["ParticleIDs"], filtered_type='PartType4')
#         def halo_stars(pfilter, data):
#             filter = np.in1d(data[pfilter.filtered_type, 'ParticleIDs'].value, ID_list,
#                               assume_unique=True)
#             return filter
        
#         self.ds.add_particle_fiter("halo_stars")