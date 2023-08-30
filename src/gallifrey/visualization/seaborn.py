#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:24:24 2023

@author: chris
"""
import os
from typing import Any, Optional

import seaborn as sns
from matplotlib import rc
from matplotlib.colors import ListedColormap
from seaborn.axisgrid import Grid
from seaborn.palettes import _ColorPalette

from gallifrey.data.paths import Path


def set_plot_defaults() -> None:
    """
    Set seaborn plot defaults.

    """

    # Seaborn options
    sns.set(rc={"figure.figsize": (18.5, 10.5)}, font_scale=2)
    sns.set_style("whitegrid", {"axes.grid": False})

    # Set the font
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
    rc("text", usetex=True)


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
        n_colors=n_colors, start=start, rot=rot, as_cmap=as_cmap, **kwargs
    )


class SeabornFigure:
    """
    Class to handle figures created by seaborn
    """

    def __init__(self, seaborn_plot: Grid, process: bool = True) -> None:
        """
        Load in and (optionally) process seaborn figure.

        Parameters
        ----------
        seaborn_plot : Grid
            The seaborn figure object.
        process : bool, optional
            If True, calls process function which further processes the plot. The
            default is True.

        """
        self.seaborn_plot = seaborn_plot

        if process:
            self.process()

    def process(self) -> None:
        """
        Process image.

        """
        # align y labels
        self.seaborn_plot.figure.align_ylabels()

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

        self.seaborn_plot.figure.savefig(path)
