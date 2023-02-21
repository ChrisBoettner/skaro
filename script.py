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
    mw = MainHalo("MW", 4096, "09_18")

#%%
fname = ds.add_deposited_particle_field(("PartType4", "GFM_Metallicity"), method="cic")

p = yt.ProjectionPlot(
    ds,
    #normal='z',
    normal=mw.sphere(ds).quantities.angular_momentum_vector().value,
    fields=fname,
    data_source=mw.sphere(ds),
    center=mw.centre(ds),
    width=(55, "kpc"),
)

# p.set_unit(("PartType0", "GFM_Metallicity"), "Zsun")
# p.set_log(("PartType0", "GFM_Metallicity"), False)
p.show()