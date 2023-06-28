#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:49:07 2023

@author: chris
"""
import numpy as np
import pandas as pd

from dace_query.population import Population
columns = Population.get_columns('ng75', output_format="pandas")

snapshot_ages = Population.get_snapshot_ages()

snapshots = Population.get_snapshots('ng75', 5, 
                                     columns=["semi_major_axis", 
                                              "total_mass", "total_radius", 
                                              "valid", "system_id", "planet_id", 
                                              "status"],
                                     output_format="pandas")

# open problem: how to get system information

# status : status (0=active, <0 accreted by another planet, 
#                  2=ejected, 3=accreted by host star)

# for one stellar particle:
#     choose simulation (star mass, number of embryos)
#     look at time, find clostest snapshots
#     look at metallicity (or other properties), find clostest match
#     define categorisation (classes, within 1AU, habitable zone etc, paper table 5)
#     get query (pay attention to planet status, should be only 0)