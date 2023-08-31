#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:24:24 2023

@author: chris
"""
import os
from typing import Any, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.pyplot import Axes, Figure
from seaborn.palettes import _ColorPalette

from gallifrey.data.paths import Path


def set_plot_defaults() -> None:
    """
    Set plot defaults.

    """
    sns.set_theme(
        style="whitegrid",
        palette="pastel",
        font_scale=2,
        rc={
            "figure.figsize": (18.5, 10.5),
            "axes.grid": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.usetex": True,
        },
    )


def get_palette(
    n_colors: int = 6,
    start: float = 0,
    rot: float = 0.4,
    reverse: bool = False,
    as_cmap: bool = False,
    **kwargs: Any,
) -> _ColorPalette | ListedColormap:
    """
    Create cubehelix color palette using seaborn.

    Parameters
    ----------
    n_colors : int, optional
        Number of colors in the palette. The default is 6.
    start : float, optional
        The hue value at the start of the helix. The default is -.2.
    rot : float, optional
        Rotations around the hue wheel over the range of the palette. The default is
        0.6.
    reverse: bool, optional
        If True, the palette will go from dark to light. The default is False.
    as_cmap: bool, optional
        If True, colormap is returned as matplotlib ListedColormap object. Otherwise
        its a seaborn ColorPalette. The default is False
    **kwargs : Any
        Additional parameter.

    Returns
    -------
    ColorPalette | ListedColormap
        Return colormap either as seaborn color palette or matplotlib colormap.

    """
    return sns.cubehelix_palette(
        n_colors=n_colors,
        start=start,
        rot=rot,
        reverse=reverse,
        as_cmap=as_cmap,
        **kwargs,
    )


class FigureProcessor:
    """
    Class to handle figures created by seaborn
    """

    def __init__(self, figure: Figure, process: bool = True) -> None:
        """
        Load in and (optionally) process seaborn figure.

        Parameters
        ----------
        seaborn_plot : Figure
            The matplotlib figure object.
        process : bool, optional
            If True, calls process function which further processes the plot. The
            default is True.

        """
        self.figure = figure

        if process:
            self.process()

    def process(self) -> None:
        """
        Process image.

        """
        # align y labels
        self.figure.align_ylabels()

    def save(self, file_name: str, sub_directory: Optional[str] = None) -> None:
        """
        Save figure in figures directory.

        Parameters
        ----------
        file_name : str
            Name and relative path of file in figures directory.
        sub_directory : Optional[str], optional
            Name of directory within figure directory. The default is None.

        """

        if sub_directory:
            path = Path().figures(f"{sub_directory}/{file_name}")
        else:
            path = Path().figures(f"{file_name}")

        # create directory if it doesn't exist already
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        self.figure.savefig(path, bbox_inches="tight", pad_inches=0)


def contour_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    reshaping_bins: int,
    cmap: Optional[ListedColormap] = None,
    colorbar_label: Optional[str] = None,
    square_aspect_ratio: bool = False,
    outline: bool = False,
    prune_lowest: bool = False,
    background_color: str = "black",
    kws: Optional[dict[str, Any]] = None,
    okws: Optional[dict[str, Any]] = None,
) -> tuple[Figure, list[Axes]]:
    """
    Create contour plot from dataframe. The x- and y-columns should be the variable
    pairs that together construct a meshgrid. The color is determined by the hue-column.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the data.
    x : str
        Name of column containing the x values.
    y : str
        Name of column containing the y values.
    hue : str
        Name of column containing the z values.
    reshaping_bins : int
        Number of bins of underlying grid, needed to reconstruct meshgrid.
    cmap : Optional[ListedColormap], optional
        The colormap to be used. The default is None.
    colorbar_label : Optional[str], optional
        Alternative label for the colorbar If None, use hue. The default is None.
    square_aspect_ratio : bool, optional
        If True, returns figure with square aspect ratio. The default is False.
    outline : bool, optional
        If True, contours have outlines. The default is False.
    prune_lowest: bool, optional
        If True, mask out lowest values so that background color is shown. The default
        is False.
    background_color: str, optional
        The plot background color. The default is black.
    kws : dict[str, Any]
        Dict of optional parameters passed to plt.contourf.
    okws : dict[str, Any]
        Dict of optional parameters passed to plt.contour, for outlines.

    Returns
    -------
    Figure
        The matplotlib figure object and list of axes in order
        [contour_ax, cbar_ax, histogram_ax].

    """
    # reshape dataframe to meshgrids for plotting
    xx, yy, zz = [
        data[key].to_numpy().reshape(2 * [reshaping_bins]) for key in (x, y, hue)
    ]

    # calculate histogram
    x_range = xx[0, :]
    z_sum = np.sum(zz, axis=0)
    bar_width = x_range[1] - x_range[0]

    # choose aspect ratio / figure size
    figsize = rcParams["figure.figsize"]
    if square_aspect_ratio:
        figsize = 2 * [1.2 * min(figsize)]

    # prune lowest values if True by adding appropriate keyword
    for keywords in [kws, okws]:
        keywords = {} if keywords is None else keywords
        if prune_lowest and ("locator" not in keywords.keys()):
            keywords["locator"] = ticker.MaxNLocator(prune="lower")

    # set up the figure and the gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[16, 1],
        height_ratios=[1, 8],
        hspace=0,
    )

    # add main contour plot
    contour_ax = fig.add_subplot(gs[1, 0])
    contour = contour_ax.contourf(xx, yy, zz, cmap=cmap, **kws)
    contour_ax.set_facecolor(background_color)  # set background color
    contour_ax.set_xlabel(x)
    contour_ax.set_ylabel(y)

    # add colorbar
    cbar_ax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label(hue if colorbar_label is None else colorbar_label)

    # add top histogram by summing over y axis
    histogram_ax = fig.add_subplot(gs[0, 0], sharex=contour_ax)
    norm = Normalize(vmin=0, vmax=max(z_sum))  # normalizer for bar colors
    color = "black" if cmap is None else cmap(norm(z_sum))  # colors based on bar values
    histogram_ax.bar(x_range, z_sum, width=bar_width, align="edge", color=color)
    histogram_ax.set_xlim(x_range[0], x_range[-1])
    histogram_ax.axis("off")

    # add outline
    if outline:
        contour = contour_ax.contour(xx, yy, zz, cmap=cmap, **okws)

    # adapt layout
    fig.tight_layout()
    return fig, [contour_ax, cbar_ax, histogram_ax]
