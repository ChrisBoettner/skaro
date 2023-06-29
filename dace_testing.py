#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:49:07 2023

@author: chris
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dace_query.population import Population

from gallifrey.data.paths import Path

columns = Population.get_columns('ng75', output_format="pandas")

snapshot_ages = Population.get_snapshot_ages()

snapshots = Population.get_snapshots('ng75', int(5e+9), 
                                     columns=columns["name"],
                                     output_format="pandas")

# status : status (0=active, <0 accreted by another planet, 
#                  2=ejected, 3=accreted by host star)

# for one stellar particle:
#     choose simulation (star mass, number of embryos)
#     look at time, find clostest snapshots
#     look at metallicity (or other properties), find clostest match
#     define categorisation (classes, within 1AU, habitable zone etc, paper table 5)
#     get query (pay attention to planet status, should be only 0)

#%%
# read system properties

# column names
columns = ['system_id', 'mstar', 'sigma', 'expo', 'ain', 'aout', 'fpg', 'mwind']

# read data file with pandas
df = pd.read_csv(Path().external_data('NGPPS_variables.txt'), delimiter='\s+', 
                 names=columns)

# modify the 'SIM' column to remove 'SIM' prefix
df['SIM'] = df['SIM'].str[3:].astype(int)

#convert columns with scientific notation to float
for col in columns[1:]:
    df[col] = df[col].map(lambda x: float(x.split('=')[1]))

#%%