#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:34:02 2023

@author: chris
"""
import yt

yt.toggle_interactivity()

from src.gallifrey.data.load import load_snapshot

ds = load_snapshot(1)

p = yt.ProjectionPlot(ds, "y", ("gas", "density"), 
                      center=([46.791, 49.064, 49.887], "Mpc"),
                      width = (1, "Mpc"),
                      buff_size=(1000,1000),
                      )
p.show()