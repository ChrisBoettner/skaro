#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:21:08 2023

@author: chris
"""
import yt
from gallifrey.utilities.structures import flatten_list
from gallifrey.data.paths import Path

def plot_and_show(
    ds,
    fields,
    axis="z",
    deposition="cic",
    colormap="kelp",
    width=(42, "kpc"),
    weight_field=None,
    units=None,
    zlims=None,
    logs=None,
    zlabels=None,
    density=None,
    **kwargs,
):
    plot = yt.ParticleProjectionPlot(
        ds=ds,
        fields=fields,
        axis=axis,
        width=width,
        deposition=deposition,
        weight_field=weight_field,
        density=density,
        **kwargs,
    )

    if units is not None:
        for field, unit in units.items():
            plot.set_unit(field, unit)

    if zlims is not None:
        for field, zlim in zlims.items():
            plot.set_zlim(field, zmin=zlim[0], zmax=zlim[1])

    if logs is not None:
        for field, log in logs.items():
            plot.set_log(field, log)

    if zlabels is not None:
        for field, zlabel in zlabels.items():
            plot.set_colorbar_label(field, zlabel)

    for field in fields:
        plot.set_cmap(field, colormap)

    plot.show()
    return plot


def plot_maps(ds, axis = "z", save=False, **kwargs):
    total_planet = plot_planets = plot_and_show(
        ds,
        fields=[("stars", "planets")],
        axis = axis,
        units={("stars", "planets"): "1/pc**2"},
        zlims={("stars", "planets"): ((1, "1/pc**2"), (2e3, "1/pc**2"))},
        zlabels={("stars", "planets"): r"Planets $\left(1/\mathrm{pc}^2\right)$"},
        density=True,
        **kwargs,
    )
    
    

    relative_planet = plots_weighted_planets = plot_and_show(
        ds,
        fields=[("stars", "mass_weighted_planets"), ("stars", "star_weighted_planets")],
        axis = axis,
        weight_field=("stars", "particle_ones"),
        zlims={
            ("stars", "star_weighted_planets"): ((0.1, ""), (0.5, "")),
            ("stars", "mass_weighted_planets"): ((0.5, "1/Msun"), (1.2, "1/Msun")),
        },
        logs={
            ("stars", "star_weighted_planets"): False,
            ("stars", "mass_weighted_planets"): False,
        },
        zlabels={
            ("stars", "star_weighted_planets"): "Star Weighted Planets",
            ("stars", "mass_weighted_planets"): r"Mass Weighted Planets "
            r"$\left(1/\mathrm{M_\odot}\right)$",
        },
        **kwargs,
    )
    
    if save:
        figs = flatten_list([[pl.figure for pl in list(plot.plots.values())]
                for plot in [total_planet,relative_planet]])
        names = ["planets", "sw_planets", "mw_planets"]
        for fig, name in zip(figs, names):
            fig.savefig(Path().figures(f"planets/maps_{axis}_{name}.pdf"))
    
    return plot_planets, plots_weighted_planets
