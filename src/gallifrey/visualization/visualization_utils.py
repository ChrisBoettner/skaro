#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:24:24 2023

@author: chris
"""
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Figure
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
        Rotations around the hue wheel over the range of the palette. The default is .6.
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
    x: str,
    y: str,
    hue: str,
    data: pd.DataFrame,
    reshaping_bins: int,
    colorbar_label: Optional[str] = None,
    **kwargs: Any,
) -> Figure:
    """
    Create contour plot from dataframe. The x- and y-columns should be the variable
    pairs that together construct a meshgrid. The color is determined by the hue-column.

    Parameters
    ----------
    x : str
        Name of column containing the x values.
    y : str
        Name of column containing the y values.
    hue : str
        Name of column containing the z values.
    data : pd.DataFrame
        The dataframe containing the data.
    reshaping_bins : int
        Number of bins of underlying grid, needed to reconstruct meshgrid.
    colorbar_label : Optional[str], optional
        Alternative label for the colorbar If None, use hue. The default is None.
    **kwargs : Any
        Further parameters passed to plt.contourf.

    Returns
    -------
    Figure
        The matplotlib figure object.

    """
    xx, yy, zz = [
        data[key].to_numpy().reshape(2 * [reshaping_bins]) for key in (x, y, hue)
    ]

    fig, ax = plt.subplots()
    contour = ax.contourf(xx, yy, zz, **kwargs)

    cbar = fig.colorbar(contour, pad=0)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if colorbar_label is None:
        colorbar_label = hue

    cbar.ax.set_ylabel(colorbar_label)

    fig.tight_layout()

    return fig
