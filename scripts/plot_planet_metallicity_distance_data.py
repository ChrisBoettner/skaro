#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:06:19 2023

@author: chris
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from gallifrey.data.paths import Path
from gallifrey.planets import PlanetModel


def plot_planet_metallicity_distance_data() -> tuple[Figure, plt.Axes]:
    """
    Plot planet distance - stellar metallicity data with errorbars and a
    critical model line.

    Returns
    -------
    fig : Figure
        matplotlib figure.
    ax : plt.Axes
        matplotlib ax.

    """

    # load data
    data = pd.read_csv(
        Path().external_data(r"observed_exoplanet_data.csv"), sep=",", comment="#"
    )

    # Extract the relevant columns from the data
    x_data = data["pl_orbsmax"]
    y_data = data["st_met"]
    x_err = data[["pl_orbsmaxerr2", "pl_orbsmaxerr1"]].abs().T
    y_err = data[["st_meterr2", "st_meterr1"]].abs().T

    # Minimum metallicity line from Johnson2012
    metallicities = np.linspace(-5, y_data.max(), 100)
    rs = PlanetModel.critical_formation_distance(metallicities)

    # Create a new figure and axes with increased size
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create the errorbar plot with cleaner marker style
    ax.errorbar(
        x_data,
        y_data,
        xerr=x_err,
        yerr=y_err,
        fmt=".",
        color="darkblue",
        alpha=0.6,
        label="Observed Exoplanets",
    )

    # Add the lower bound line
    ax.plot(rs, metallicities, color="darkred", label="Lower Limit")

    # Shade the area below the model line
    ax.fill_between(
        rs, metallicities, y2=ax.get_ylim()[0], color="lightgray", alpha=0.5
    )

    # Set the scale and limits
    ax.set_xscale("log")
    ax.set_xlim(0.005, 15)
    ax.set_ylim(-2.85, 1.2)

    # Add labels with latex formatting and increased font size
    ax.set_xlabel("Orbit Semi-Major Axis [au]", fontsize=20)
    ax.set_ylabel("[Fe/H]", fontsize=20)

    # Add the 'Forbidden Zone' text to the plot
    ax.text(1.05, -1.96, "Forbidden Zone", fontsize=18, color="darkred", alpha=0.8)

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.6)

    # Increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adjust layout for better spacing
    fig.tight_layout()

    return fig, ax


# Use the function
if __name__ == "__main__":
    fig, ax = plot_planet_metallicity_distance_data()
    fig.savefig(Path().figures("planet_metallicity_distance_observations.pdf"))
