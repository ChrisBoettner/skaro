#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:34:02 2023

@author: chris
"""
import yt
yt.toggle_interactivity()

from gallifrey.data.load import load_snapshot
from gallifrey.halo import MainHalo

ds = load_snapshot(127, 4096)
mw = MainHalo("MW", 4096, "09_18")


p = yt.ProjectionPlot(ds, normal = mw.sphere(ds).quantities.angular_momentum_vector().value,
                 fields=('gas','density'), data_source=mw.sphere(ds),
                      center = mw.centre(ds), 
                      width =(5000, 'kpc') )
p.set_zlim(("gas", "density"), zmin=(1e-8, "g/cm**2"), zmax=(1e-2, "g/cm**2"))
p.save()



# p = yt.ProjectionPlot(ds, "y", ("gas", "density"), 
#                       center=([46.791, 49.064, 49.887], "Mpc"),
#                       width = (1, "Mpc"),
#                       buff_size=(1000,1000),
#                       )
# p.show()