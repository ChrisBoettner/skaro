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
from scipy.interpolate import UnivariateSpline

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
    scatter_size=56,
    line_color="#754682",
    line_width=4,
    scatter_alpha=0.4,
    label_fontsize=20,
    tick_fontsize=16,
    text_fontsize=20,
    smoothness=0,
    xlim=[0, 25],
    ylim=None,
    shading=None,
):
    # Create a figure with specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Smoothed line
    spline = UnivariateSpline(x, y, s=smoothness)
    xnew = np.linspace(min(x), max(x), 500)
    ax.plot(xnew, spline(xnew), color=line_color, linewidth=line_width)

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

    # Adjust the padding
    plt.tight_layout()
    return fig, ax


def make_1dprofiles(data_source, bins=100):
    # planet profiles
    total_planet_profile = yt.create_profile(
        data_source=data_source,
        bin_fields=[("stars", "particle_radius")],
        fields=("stars", "planets"),
        n_bins=bins,
        units={("stars", "particle_radius"): "kpc"},
        logs={("stars", "particle_radius"): False},
        weight_field=None,
        deposition="cic",
    )

    relative_planet_profiles = yt.create_profile(
        data_source=data_source,
        bin_fields=[("stars", "particle_radius")],
        fields=[("stars", "star_weighted_planets"), ("stars", "mass_weighted_planets")],
        n_bins=bins,
        units={("stars", "particle_radius"): "kpc"},
        logs={("stars", "particle_radius"): False},
        weight_field=("stars", "particle_ones"),
        deposition="cic",
    )
    return total_planet_profile, relative_planet_profiles


def plot_1dprofiles(data_source, halo, bins=100):
    total_planet_profile, relative_planet_profiles = make_1dprofiles(data_source, bins)

    # calculate bin volumes
    bin_sizes = total_planet_profile.x_bins.to("pc")
    bin_volume = 4 / 3 * np.pi * bin_sizes[1:] ** 3 - bin_sizes[:-1] ** 3
    fig1, ax1 = plot_and_show(
        total_planet_profile.x,
        total_planet_profile[("stars", "planets")] / bin_volume,
        xlabel="Shell Radius (kpc)",
        ylabel=r"Planets $\left[\frac{1}{\mathrm{pc}^3}\right]$",
        xscale="linear",
        yscale="log",
        smoothness=0,
        shading=[halo.BULGE_END, halo.DISK_END],
    )

    fig2, ax2 = plot_and_show(
        relative_planet_profiles.x,
        relative_planet_profiles[("stars", "star_weighted_planets")],
        xlabel="Shell Radius (kpc)",
        ylabel="Star Weighted Planets",
        xscale="linear",
        yscale="linear",
        smoothness=0.00022,
        shading=[halo.BULGE_END, halo.DISK_END],
    )
    fig3, ax3 = plot_and_show(
        relative_planet_profiles.x,
        relative_planet_profiles[("stars", "mass_weighted_planets")],
        xlabel="Shell Radius (kpc)",
        ylabel="Mass Weighted Planets",
        xscale="linear",
        yscale="linear",
        smoothness=0.0015,
        shading=[halo.BULGE_END, halo.DISK_END],
    )

    return (fig1, fig2, fig3), (ax1, ax2, ax3)
