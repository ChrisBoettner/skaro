#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:21:08 2023

@author: chris
"""

import yt


def plot_and_show(
    ds,
    fields,
    axis="z",
    deposition="cic",
    colormap="kelp",
    width=(55, "kpc"),
    weight_field=None,
    units=None,
    zlims=None,
    logs=None,
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

    for field in fields:
        plot.set_cmap(field, colormap)

    plot.show()
    return plot


def plot_maps(ds, **kwargs):
    plot_planets = plot_and_show(
        ds,
        fields=[("stars", "planets")],
        units={("stars", "planets"): "1/pc**2"},
        zlims={("stars", "planets"): ((1, "1/pc**2"), (2e3, "1/pc**2"))},
        density=True,
        **kwargs,
    )

    plots_weighted_planets = plot_and_show(
        ds,
        fields=[("stars", "mass_weighted_planets"), ("stars", "star_weighted_planets")],
        weight_field=("stars", "particle_ones"),
        zlims={
            ("stars", "star_weighted_planets"): ((0.0, ""), (0.5, "")),
            ("stars", "mass_weighted_planets"): ((0.5, "1/Msun"), (1.2, "1/Msun")),
        },
        logs={
            ("stars", "star_weighted_planets"): False,
            ("stars", "mass_weighted_planets"): False,
        },
        **kwargs,
    )
    return plot_planets, plots_weighted_planets
