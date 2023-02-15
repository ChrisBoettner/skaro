#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:34:02 2023

@author: chris
"""
#%%
# load local version before pip installed version, for debugging
import os
import sys
sys.path.append(os.path.join(sys.path[0],'src'))

#%%
import yt

from gallifrey.data.load import load_snapshot
from gallifrey.halo import MainHalo

from gallifrey.visualization.utilities import FigureManager

man = FigureManager()


#%%
ds = load_snapshot(127, 4096)
mw = MainHalo("MW", 4096, "09_18")


p = yt.ProjectionPlot(ds, normal = mw.sphere(ds).quantities.angular_momentum_vector().value,
                 fields=[('gas','density'), ('gas','temperature')], data_source=mw.sphere(ds),
                      center = mw.centre(ds), 
                      width =(5000, 'kpc') )
p.set_zlim(("gas", "density"), zmin=(1e-8, "g/cm**2"), zmax=(1e-2, "g/cm**2"))

man.show(p)