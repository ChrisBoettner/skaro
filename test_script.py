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
with Timer('stars'):
    from gallifrey.fields import Fields
    from gallifrey.filter import Filter
    
    filters = Filter(ds)
    filters.add_stars()
    
    fields = Fields(ds)
    fields.convert_stellar_age()
    
#%%
with Timer('planets'):
    import matplotlib.pyplot as plt
    import numpy as np

    from gallifrey.planets import PlanetModel, PlanetOccurenceModel
    from gallifrey.stars import ChabrierIMF, StellarModel
    data = mw.sphere()
    planet_model = PlanetModel()
    stellar_model = StellarModel()

    imf = ChabrierIMF()
    
    fields.add_planets(stellar_model, planet_model, imf)

#%%
# mass-weighted planet profile
planet_profile = yt.create_profile(
    data_source=mw.disk(radius=ds.quan(20,'kpc')),
    bin_fields=[("stars", "particle_radius")],
    fields=[("stars", "planets")],
    n_bins=100,
    units={('stars', 'particle_radius'): 'kpc'},
    weight_field=('stars', 'Masses'),
)

plot = yt.ProfilePlot.from_profiles(planet_profile)
plot.set_log(('stars', 'particle_radius'), False)
plot.set_log(('stars','planets'), False)
    
