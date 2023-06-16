#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:17:47 2023

@author: chris
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yt

from gallifrey.data.paths import Path
from gallifrey.utilities.math import calculate_smoothing_line

color_palette = ["#1E0C78", "#754682", "#978489", "#F1EE88", "#B2CB9E"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_and_show(
    x,
    y,
    xlabel,
    ylabel,
    xscale="log",
    yscale="log",
    figsize=(10, 6),
    scatter_color="#1E0C78",
    scatter_size=60,
    line_color="#754682",
    line_width=4,
    scatter_alpha=0.4,
    label_fontsize=20,
    tick_fontsize=16,
    text_fontsize=20,
    smoothness=0,
    xlim=[0, 30],
    ylim=None,
    shading=None,
    extra_line=None,
):
    # Create a figure with specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Smoothed line
    smoothed = calculate_smoothing_line(x, y, fraction=smoothness)
    ax.plot(*smoothed.T, color=line_color, linewidth=line_width)

    # Scatter plot
    ax.scatter(x, y, c=scatter_color, s=scatter_size, alpha=scatter_alpha)
    ax.set_xlim(xlim)

    # add shaded regions
    if shading:
        ax.axvspan(xlim[0], shading[0], alpha=0.4, color="#754682")
        ax.text(
            (xlim[0] + shading[0]) / 2 / (xlim[1] - xlim[0]),
            0.05,
            "BULGE",
            transform=ax.transAxes,
            ha="center",
            size=text_fontsize,
        )
        ax.axvspan(shading[0], shading[1], alpha=0.3, color="#754682")
        ax.text(
            (shading[0] + shading[1]) / 2 / (xlim[1] - xlim[0]),
            0.05,
            "DISK",
            transform=ax.transAxes,
            ha="center",
            size=text_fontsize,
        )
        ax.axvspan(shading[1], xlim[1], alpha=0.2, color="#754682")
        ax.text(
            (shading[1] + xlim[1]) / 2 / (xlim[1] - xlim[0]),
            0.05,
            "HALO",
            transform=ax.transAxes,
            ha="center",
            size=text_fontsize,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Change the size of tick labels
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # add limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if extra_line:
        for line in extra_line:
            ax.axvline(line, linewidth=int(line_width / 3), color="black")

    # Adjust the padding
    plt.tight_layout()
    return fig, ax


def make_1dprofiles(data_source, bins=100):
    # planet profiles
    total_planet_profile = yt.create_profile(
        data_source=data_source,
        bin_fields=[("stars", "perp_radius")],
        fields=("stars", "planets"),
        n_bins=bins,
        units={("stars", "perp_radius"): "kpc"},
        logs={("stars", "perp_radius"): False},
        weight_field=None,
        deposition="cic",
    )

    relative_planet_profiles = yt.create_profile(
        data_source=data_source,
        bin_fields=[("stars", "perp_radius")],
        fields=[("stars", "star_weighted_planets"), ("stars", "mass_weighted_planets")],
        n_bins=bins,
        units={("stars", "perp_radius"): "kpc"},
        logs={("stars", "perp_radius"): False},
        weight_field=("stars", "particle_ones"),
        deposition="cic",
    )
    return total_planet_profile, relative_planet_profiles


def plot_1dprofiles(
    data_source,
    halo,
    disk_height,
    bins=100,
    save=False,
    smoothness=(0.08, 0.15),
    no_dwarfs=False,
):
    total_planet_profile, relative_planet_profiles = make_1dprofiles(data_source, bins)

    # calculate bin volumes (cylinder)
    bin_sizes = total_planet_profile.x_bins.to("pc")
    bin_volume = (
        np.pi * 2 * disk_height.to("pc") * (bin_sizes[1:] ** 2 - bin_sizes[:-1] ** 2)
    )

    # planet density
    fig1, ax1 = plot_and_show(
        total_planet_profile.x,
        total_planet_profile[("stars", "planets")] / bin_volume,
        xlabel="Disk Radius (kpc)",
        ylabel=r"Planets $\left(1/\mathrm{pc}^3\right)$",
        xscale="linear",
        yscale="log",
        smoothness=smoothness[0],
        shading=[halo.BULGE_END, halo.DISK_END],
    )

    fig2, ax2 = plot_and_show(
        relative_planet_profiles.x,
        relative_planet_profiles[("stars", "star_weighted_planets")],
        xlabel="Disk Radius (kpc)",
        ylabel="Star Weighted Planets",
        xscale="linear",
        yscale="linear",
        smoothness=smoothness[1],
        shading=[halo.BULGE_END, halo.DISK_END],
    )

    fig3, ax3 = plot_and_show(
        relative_planet_profiles.x,
        relative_planet_profiles[("stars", "mass_weighted_planets")],
        xlabel="Disk Radius (kpc)",
        ylabel=r"Mass Weighted Planets $\left(1/\mathrm{M_\odot}\right)$",
        xscale="linear",
        yscale="linear",
        smoothness=smoothness[1],
        shading=[halo.BULGE_END, halo.DISK_END],
    )

    # cummulative planets
    cummulative_planets = np.cumsum(
        total_planet_profile[("stars", "planets")]
    ) / np.sum(total_planet_profile[("stars", "planets")])
    fig4, ax4 = plot_and_show(
        total_planet_profile.x,
        cummulative_planets,
        xlabel="Disk Radius (kpc)",
        ylabel=r"Planets Fraction",
        xscale="linear",
        yscale="linear",
        smoothness=smoothness[0],
        shading=[halo.BULGE_END, halo.DISK_END],
        extra_line=[8.2],
    )

    figs = (fig1, fig2, fig3, fig4)
    axes = (ax1, ax2, ax3, ax4)

    if save:
        names = ["planets", "sw_planets", "mw_planets", "cummulative_planets"]
        if no_dwarfs:
            names = [name + "_no_dwarfs" for name in names]
        for fig, name in zip(figs, names):
            fig.savefig(Path().figures(f"planets/1d_profile_{name}.pdf"))

    return figs, axes
