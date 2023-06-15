#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:16:45 2023

@author: chris
"""
import yt
from gallifrey.utilities.structures import flatten_list
from gallifrey.data.paths import Path

def create_profile(data_source, bin_fields, bins, fields, units, logs, weight_field):
    """Create a profile using given parameters."""
    return yt.create_profile(
        data_source,
        bin_fields,
        n_bins=bins,
        fields=fields,
        units=units,
        logs=logs,
        weight_field=weight_field,
        deposition="cic",
    )


def create_plot(profile, field, logs, field_log, x_range, y_range, sun_reference_coord):
    """Create a plot using a profile and other provided parameters."""
    plot = yt.PhasePlot.from_profile(profile)
    plot.annotate_text(*sun_reference_coord, "☉")
    plot.set_log(("stars", "particle_radius"), False)
    for log_name, log_val in logs.items():
        plot.set_log(log_name, log_val)
    if field_log is False:
        plot.set_log(field, field_log)
    plot.set_xlim(*x_range)
    plot.set_ylim(*y_range)
    plot.set_xlabel("Shell Radius (kpc)")
    plot.set_cmap(field, "kelp")
    return plot


def generate_plots(
    data_source,
    bin_fields,
    bins,
    units,
    logs,
    fields_info,
    x_range,
    y_range,
    sun_reference_coord,
):
    """Generate a set of plots using the provided parameters."""
    plots = []
    for info in fields_info:
        profile = create_profile(
            data_source,
            bin_fields,
            bins,
            info["fields"],
            units,
            logs,
            info.get("weight_field"),
        )
        plot = create_plot(
            profile,
            info["fields"][0],
            logs,
            info.get("field_log_status", False),
            x_range,
            y_range,
            sun_reference_coord,
        )
        if "z_lims" in info:
            plot.set_zlim(
                info["fields"][0], zmin=info["z_lims"][0], zmax=info["z_lims"][1]
            )
        plot.show()
        plots.append(plot)
    return plots


def plot_2dprofiles(ds, quantity, bins=[200, 200], save=False):
    """Generate 2D plots for the given quantity."""
    units = {("stars", "particle_radius"): "kpc"}
    logs = {("stars", "particle_radius"): False}
    fields_info = [
        {
            "fields": [("stars", "planets")],
            "weight_field": None,
            "field_log_status": True,
        },
        {
            "fields": [("stars", "star_weighted_planets")],
            "weight_field": ("stars", "particle_ones"),
            "field_log_status": False,
            "z_lims": (0.1, 0.5),
        },
        {
            "fields": [("stars", "mass_weighted_planets")],
            "weight_field": ("stars", "particle_ones"),
            "field_log_status": False,
            "z_lims": (0.5, 1.2),
        },
    ]
    
    x_range = (0, 25)
    if quantity == "stellar_age":
        bin_fields = [("stars", "particle_radius"), ("stars", "stellar_age")]
        logs[("stars", "stellar_age")] = False
        y_range = (0.1, 13.7)
        sun_reference_coord = (8.2, 4.6)
    elif quantity == "[Fe/H]":
        bin_fields = [("stars", "particle_radius"), ("stars", "[Fe/H]")]
        logs[("stars", "[Fe/H]")] = False
        y_range = (-2.5, 0.8)
        sun_reference_coord = (8.2, 0)
    else:
        raise ValueError(
            "Unknown quantity, please choose either 'stellar_age' or '[Fe/H]'."
        )
        
    plots = generate_plots(ds, bin_fields, bins, units, logs, fields_info, 
                           x_range, y_range, sun_reference_coord)

    if save:
        figs = flatten_list([[pl.figure for pl in list(plot.plots.values())]
                for plot in plots])
        names = ["planets", "sw_planets", "mw_planets"]
        for fig, name in zip(figs, names):
            fig.savefig(Path().figures(f"planets/2d_profile_{quantity}_{name}.pdf"))
    return plots