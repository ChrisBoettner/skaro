#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:21:08 2023

@author: chris
"""
import numpy as np
import yt
import matplotlib.pyplot as plt

from gallifrey.data.paths import Path

def plot_maps(planet_categories, data, normal="z", plot_width=(42, "kpc"), 
                cmap="kelp", figsize=(18.5, 10.5), subplot_columns= 3, 
                deposition_method="cic", weight_field=None,
                global_normalisation=True, colorbar_percentiles=(1,99), save=False):
    
    # create fields
    fields = [("stars", category) for category in planet_categories]
    
    # create plots using yt
    plot = yt.ParticleProjectionPlot(
        ds=data,
        fields=fields,
        normal=normal,
        width=plot_width,
        deposition=deposition_method,
        weight_field=weight_field,
        density=True,
    )

    # choose colorbar label
    if weight_field is None:
        label = r"s $\left(1/\mathrm{pc}^2\right)$"
    elif weight_field == ("stars", "number"):
        label = r"s Per Star $\left(1/\mathrm{pc}^2\right)$"
    else:
         label = (f"s (weighted by {weight_field[-1]})"+
                  r" $\left(1/\mathrm{pc}^2\right)$")
    
    # change density units (need to do first to properly calculate colorbar percentiles)
    for field in fields:
        plot.set_unit(field, "1/pc**2")        
         
    if global_normalisation:
        image_values = np.array([np.array(plot.frb[field]) for field 
                                 in fields]).flatten()
        percentiles = np.nanpercentile(image_values[image_values > 0], 
                                       colorbar_percentiles)
        

    for field in fields:
        if not global_normalisation:
            image_values = np.array(plot.frb[field]).flatten()
            percentiles = np.nanpercentile(image_values[image_values > 0], 
                                           colorbar_percentiles)            

        plot.set_cmap(field, cmap)
        plot.set_colorbar_label(field, field[-1] + label)
        plot.set_zlim(field, *percentiles)
    
    # convert to matplotlib figure
    fig = plot.export_to_mpl_figure((np.ceil(len(fields)/subplot_columns).astype(int),
                                     subplot_columns))
    fig.set_size_inches(*figsize)
    fig.tight_layout()
    
    if save:
        fig.savefig(Path().figures(f"planets/planets_maps_{normal}.pdf"))
    return plot, fig