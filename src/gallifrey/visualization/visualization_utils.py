#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:24:24 2023

@author: chris
"""
import os
from typing import Any, Callable, Optional

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
from gallifrey.utilities.logging import logger

# create Logger
logger = logger(__name__)


def set_plot_defaults() -> None:
    """
    Set plot defaults.

    """
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="pastel",
        font_scale=3.2,
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
    diverging: bool = False,
    as_cmap: bool = False,
    second_palette_start: float = 1.65,
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
        0.4 .
    diverging : bool, optional
        If True, create a diverging colormap. The default is False.
    as_cmap: bool, optional
        If True, colormap is returned as matplotlib ListedColormap object. Otherwise
        its a seaborn ColorPalette. The default is False.
    second_palette_start : float, optional
        Starting point for second part of colormap, if diverging is True. The default
        is 1.65 .
    **kwargs : Any
        Additional parameters for cubehelix_palette. Ignored if diverging is True.

    Returns
    -------
    ColorPalette | ListedColormap
        Return colormap either as seaborn color palette or matplotlib colormap.

    """

    if not diverging:
        return sns.cubehelix_palette(
            n_colors=n_colors,
            start=start,
            rot=rot,
            as_cmap=as_cmap,
            **kwargs,
        )

    # for diverging colormap, create two maps and combine them
    else:
        # if output is ListedColormap, use max number of colors
        if as_cmap:
            n_colors = 256

        # create palettes
        palette_one = sns.cubehelix_palette(
            n_colors=(
                n_colors // 2 if n_colors % 2 else n_colors // 2 + 1
            ),  # odd number handling
            start=second_palette_start,
            rot=rot,
            light=0.95,
            as_cmap=False,
            reverse=True,
        )
        palette_two = sns.cubehelix_palette(
            n_colors=n_colors // 2,
            start=start,
            rot=rot,
            light=0.95,
            as_cmap=False,
        )
        diverging_palette = palette_one + palette_two

        if as_cmap:
            return ListedColormap(diverging_palette)
        return diverging_palette


def adjust_legend(ax: Axes, ncols: int = 3, pad: float = 1) -> Axes:
    """
    Adjust plot to accomodate for legend. Can increase the number of columns for the
    legend, and add extra space at top of legend for legend.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object.
    ncols : int, optional
        Number of columns for the legend. The default is 3.
    pad : float, optional
        Additional padding at top of plot (multiple of ymax). The default is 1,
        i.e. no change.

    Returns
    -------
    Axes
        Matplotlib Axes object with adjusted for the legend.

    """

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * pad)
    ax.legend(ncols=3)
    return ax


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

    def save(
        self,
        file_name: str,
        sub_directory: Optional[str] = None,
        save: Optional[bool] = True,
    ) -> None:
        """
        Save figure in figures directory.

        Parameters
        ----------
        file_name : str
            Name and relative path of file in figures directory.
        sub_directory : Optional[str], optional
            Name of directory within figure directory. The default is None.
        save: Optional[bool], optional
            If False, only create file structure and don't actually save image. The
            default is True.


        """
        if sub_directory:
            path = Path().figures(f"{sub_directory}/{file_name}")
        else:
            path = Path().figures(f"{file_name}")

        # create directory if it doesn't exist already
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if save:
            self.figure.savefig(path, bbox_inches="tight", pad_inches=0)


def contourplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    reshaping_bins: int,
    bin_window: int = 1,
    additional_contours: Optional[str] = None,
    cmap: Optional[ListedColormap] = None,
    colorbar_label: Optional[str] = None,
    square_aspect_ratio: bool = False,
    outline: bool = False,
    prune_lowest: bool = False,
    background_color: str = "black",
    contour_label_fmt: str | Callable = "%1.1f",
    kws: Optional[dict[str, Any]] = None,
    okws: Optional[dict[str, Any]] = None,
    ackws: Optional[dict[str, Any]] = None,
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
    bin_window : int, optional
        Choose how many bins are combined into single bar of top barplot.
    additional_contours : Optional[str], optional
        Optional name of seperate column to make additional contours. The default
        is None
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
    contour_label_fmt: str | Callable, optional
        Label formatter for additional contours. The default is  "%1.1f", which return
        values with one decimal.
    kws : dict[str, Any]
        Dict of optional parameters passed to plt.contourf.
    okws : dict[str, Any]
        Dict of optional parameters passed to plt.contour, for outlines.
    ackws : dict[str, Any]
        Dict of optional parameters passed to plt.contour, for additional contours.

    Returns
    -------
    Figure
        The matplotlib figure object and list of axes in order
        [contour_ax, cbar_ax, histogram_ax].

    """
    kws = {} if kws is None else kws
    okws = {} if okws is None else okws
    ackws = {} if ackws is None else ackws

    # reshape dataframe to meshgrids for plotting
    xx, yy, zz = [
        data[key].to_numpy().reshape(2 * [reshaping_bins]) for key in (x, y, hue)
    ]

    # calculate barplot
    if (reshaping_bins % bin_window) != 0:
        raise ValueError("reshaping_bins must be divisible by bin_windows.")
    x_range = xx[0, ::bin_window]
    z_sum = np.sum(zz, axis=0)
    bin_values = np.sum(z_sum.reshape(-1, bin_window), axis=1)  # combine bins
    bar_width = x_range[1] - x_range[0]

    # choose aspect ratio / figure size
    figsize = rcParams["figure.figsize"]
    if square_aspect_ratio:
        figsize = 2 * [1.2 * min(figsize)]

    # prune lowest values if True by adding appropriate keyword
    for keywords in [kws, okws, ackws]:
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
        wspace=0,
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
    cbar.outline.set_visible(False)  # type: ignore

    # add top histogram by summing over y axis
    histogram_ax = fig.add_subplot(gs[0, 0], sharex=contour_ax)
    norm = Normalize(vmin=0, vmax=max(1.1 * bin_values))  # normalizer for bar colors
    color = (
        "black" if cmap is None else cmap(norm(bin_values))
    )  # colors based on bar values
    histogram_ax.bar(x_range, bin_values, width=bar_width, align="edge", color=color)
    histogram_ax.set_xlim(x_range[0], x_range[-1])
    histogram_ax.axis("off")

    # add outline
    if outline:
        contour = contour_ax.contour(xx, yy, zz, cmap=cmap, **okws)

    if additional_contours is not None:
        ww = data[additional_contours].to_numpy().reshape(2 * [reshaping_bins])

        add_contours = contour_ax.contour(xx, yy, ww, **ackws)
        contour_ax.clabel(add_contours, inline=True, fmt=contour_label_fmt)

    # adapt layout
    fig.tight_layout()
    return fig, [contour_ax, cbar_ax, histogram_ax]


def ridgeplot(
    data: pd.DataFrame,
    x: str,
    row: str,
    height: float = 3,
    aspect: float = 7,
    hspace: float = -0.45,
    palette: Optional[_ColorPalette] = None,
    font_scale: float = 5,
    label_position: tuple[float, float] = (1, 0.2),
) -> sns.FacetGrid:
    """
    Create Ridgeplot using Seaborn FacetGrid.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the data.
    x : str
        Name of column containing the x values.
    row : str
        Name of column by which the FacetGrid is build, should be categorical.
    height : float, optional
        Height (in inches) of each facet. The default is 3.
    aspect : float, optional
        Aspect ratio of each facet, so that aspect * height gives the width of each
        facet in inches. The default is 7.
    hspace : float, optional
        Distance between ridgle plots, negative values lead to overlap. The
        default is -.45.
    palette : Optional[_ColorPalette], optional
        The color palette. The default is None.
    font_scale : float, optional
        Multiplier for font scale. The default is 5.
    label_position : tuple[float, float], optional
        Position of labels for every facet, in axes coordinates. The default is (1, .2).

    Returns
    -------
    grid : FacetGrid
        The FacetGrid object. Access matplotlib figure using figure attribute and
        axes using axes attribute.

    """
    # set font_scale and make background transparent, for this plot only
    with sns.plotting_context("paper", font_scale=font_scale):
        with plt.rc_context(rc={"axes.facecolor": (0, 0, 0, 0)}):
            if data[row].nunique() > 10:
                logger.warn(
                    "VISUALIZATION: Ridgeplot tries to create more than 10 "
                    "rows. This might take a while. Are you sure you "
                    "specificed the right dataframe column for the rows? It "
                    "should contain categorical data."
                )

            # create grid
            grid = sns.FacetGrid(
                data,
                row=row,
                hue=row,
                height=height,
                aspect=aspect,
                palette=palette,
                sharex=True,
            )

            # Draw kde plot and white outline
            grid.map(sns.kdeplot, x, clip_on=False, fill=True, alpha=1, linewidth=1.5)
            grid.map(sns.kdeplot, x, clip_on=False, color="white", lw=3)

            # add reference line bottom,
            # passing color=None to refline() uses the hue mapping
            grid.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

            # Define and use a simple function to label the plot in axes coordinates
            def label(x: str, color: str, label: str) -> None:
                ax = plt.gca()
                ax.text(
                    *label_position,
                    label,
                    fontweight="bold",
                    color="black",
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                )

            grid.map(label, x)

            # Set the subplots to overlap
            grid.figure.subplots_adjust(hspace=hspace)

            # Remove axes details that don't play well with overlap
            grid.set_titles("")
            grid.set(yticks=[], ylabel="")
            grid.despine(bottom=True, left=True)

            # Add a frame around the entire plot
            grid.fig.tight_layout

    return grid
