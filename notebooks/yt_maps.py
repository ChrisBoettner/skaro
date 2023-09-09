#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:21:08 2023

@author: chris
"""
import os
import warnings
from typing import Any, Optional

import mpl_toolkits
import numpy as np
import yt
from matplotlib.pyplot import Axes, Figure
from yt import ParticleProjectionPlot
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.data.paths import Path
from gallifrey.visualization.visualization_utils import get_palette


def plot_component_maps(
    galaxy_component_data: dict[str, YTDataContainerDataset],
    field_value: Optional[str] = "total_number",
    normal: str = "z",
    weight_field: Optional[tuple[str, str]] = None,
    colorbar_percentiles: tuple[float, float] = (1, 99),
    width: tuple[float, str] = (43.3, "kpc"),
    depth: tuple[float, str] = (43.3, "kpc"),
    cmap: Optional[str] = None,
    figsize: tuple[float, float] = (12.6, 12.6),
    deposition_method: str = "cic",
    density_unit: str = "1/pc**2",
    font_dict: dict[str, Any] = {"size": 40},
    hide_colorbar: bool = True,
    hide_axes: bool = False,
    save: bool = False,
    figure_name_addon: Optional[str] = None,
) -> tuple[list[ParticleProjectionPlot], list[Figure]]:
    """
    Create maps of galaxy components.

    Parameters
    ----------
    galaxy_component_data : dict[str, YTDataContainerDataset]
        Dictonary of data containers, must be of form {component_name: component_data}.
    field_value:
        The field value to be plotted. The default is "total_number", which corresponds
        to the number of stars.
    normal : str, optional
        Direction of normal vector for projection. The default is "z".
    weight_field : Optional[tuple[str,str]], optional
        Field to weight planet fields by. The default is None.
    colorbar_percentiles : tuple[float,float], optional
        DESCRIPTION. The default is (1, 99).
    width : tuple[float, str], optional
        Plot limits for map. The default is (43.3, "kpc").
    depth : tuple[float, str], optional
        Depth of the projected, centered on domain center. The default is (43.3, "kpc").
    cmap : str, optional
        Colormap to use. The default is "None", which loads the color palette defined
        in gallifrey.visualization.seaborn.
    figsize : tuple[float,float], optional
        Size of figure. The default is (12.6, 12.6).
    deposition_method : str, optional
        Deposition method for fixed resolution buffer. The default is "cic".
    density_unit : str, optional
        Unit for spatial density. The default is "1/pc**2". Changing this value will
        lead to an incorrect colorbar, which needs to be adjusted manually.
    font_dict : dict[str, Any], optional
        Dictonary of additional font properties. The default is {"size": 16}.
    save : bool, optional
        If True, save figure to Figures directory. The default is False.
    figure_name_addon : str, optional
        Optional addon to default figure name. The default is None.

    Returns
    -------
    tuple[list[ParticleProjectionPlot], list[Figure]]
        Lists of ParticleProjectionPlot and Figure objects containing the figures.

    """

    if cmap is None:
        cmap = get_palette(rot=0.6, as_cmap=True, reverse=True)

    plots, figs = [], []
    for component, data in galaxy_component_data.items():
        field = (f"{component}_stars", field_value)

        # create plots using yt
        plot = yt.ParticleProjectionPlot(
            ds=data,
            fields=field,
            normal=normal,
            width=width,
            depth=depth,
            deposition=deposition_method,
            weight_field=weight_field,
            density=True,
        )

        # general plot configurations
        plot_configurations(
            plot,
            [field],
            cmap,
            density_unit,
            figsize,
            font_dict,
            "individual",
            1,
        )

        # set colorbar limits
        set_colorbar_limits(
            plot,
            [field],
            "individual",
            colorbar_percentiles,
            1,
            1,
        )

        cbar_mode = "each"
        if hide_colorbar:
            plot.hide_colorbar()
            cbar_mode = None

        if hide_axes:
            plot.hide_axes(draw_frame=True)
            plot.annotate_scale(coeff=5, text_args=font_dict)

        # export tp matplotlib
        fig = plot.export_to_mpl_figure((1, 1), cbar_mode=cbar_mode)

        if save:
            file_name = f"Galaxy_component_maps/{component}_map_{normal}"
            if figure_name_addon:
                file_name = f"{file_name}_{figure_name_addon}.pdf"
            else:
                file_name = f"{file_name}.pdf"
            path = Path().figures(f"{file_name}")

            # create directory if it doesn't exist already
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            fig.savefig(
                path,
                bbox_inches="tight",
            )
        plots.append(plot)
        figs.append(fig)

    return plots, figs


def plot_planet_maps(
    planet_categories: list[str],
    data: YTDataContainerDataset,
    normal: str = "z",
    weight_field: Optional[tuple[str, str]] = None,
    colorbar_normalisation: str = "inidvidual",
    colorbar_percentiles: tuple[float, float] = (1, 99),
    width: tuple[float, str] = (43.3, "kpc"),
    depth: tuple[float, str] = (43.3, "kpc"),
    cmap: Optional[str] = None,
    figsize: tuple[float, float] = (18.5, 10.5),
    deposition_method: str = "cic",
    density_unit: str = "1/pc**2",
    font_dict: dict[str, Any] = {"size": 18},
    subplot_columns: int = 3,
    subplot_pad: float | tuple[float, float] = (0, 0),
    save: bool = False,
    figure_name_addon: Optional[str] = None,
) -> tuple[ParticleProjectionPlot, Figure]:
    """
    Create maps of planet distributions.

    Parameters
    ----------
    planet_categories : list[str]
        List of planet categories.
    data : YTDataContainerDataset
        Hestia data object.
    normal : str, optional
        Direction of normal vector for projection. The default is "z".
    weight_field : Optional[tuple[str,str]], optional
        Field to weight planet fields by. The default is None.
    colorbar_normalisation : str, optional
        Normalisation of colorbars, can be "individual", "row" or "global". The
        default is "inidvidual".
    colorbar_percentiles : tuple[float,float], optional
        DESCRIPTION. The default is (1, 99).
    width : tuple[float, str], optional
        Plot limits for map. The default is (43.3, "kpc").
    depth : tuple[float, str], optional
        Depth of the projected, centered on domain center. The default is (43.3, "kpc").
    cmap : str, optional
        Colormap to use. The default is "None", which loads the color palette defined
        in gallifrey.visualization.seaborn.
    figsize : tuple[float,float], optional
        Size of figure. The default is (18.5, 10.5).
    deposition_method : str, optional
        Deposition method for fixed resolution buffer. The default is "cic".
    density_unit : str, optional
        Unit for spatial density. The default is "1/pc**2". Changing this value will
        lead to an incorrect colorbar, which needs to be adjusted manually.
    font_dict : dict[str, Any], optional
        Dictonary of additional font properties. The default is {"size": 16}.
    subplot_columns : int, optional
        Number of subplot columns. Rows are infered from length of planet_categories
        list. The default is 3.
    subplot_pad : float | tuple[float, float], optional
        Padding between subplots. The default is 1.
    save : bool, optional
        If True, save figure to Figures directory. The default is False.
    figure_name_addon : str, optional
        Optional addon to default figure name. The default is None.

    Returns
    -------
    tuple[ParticleProjectionPlot, Figure]
        ParticleProjectionPlot and Figure object containing the figure.

    """
    if cmap is None:
        cmap = get_palette(rot=0.6, as_cmap=True, reverse=True)

    # figure layout
    subplot_rows = np.ceil(len(planet_categories) / subplot_columns).astype(int)

    # create fields
    fields = [("stars", category) for category in planet_categories]

    # create plots using yt
    plot = yt.ParticleProjectionPlot(
        ds=data,
        fields=fields,
        normal=normal,
        width=width,
        depth=depth,
        deposition=deposition_method,
        weight_field=weight_field,
        density=True,
    )

    # general plot configurations
    plot_configurations(
        plot,
        fields,
        cmap,
        density_unit,
        figsize,
        font_dict,
        colorbar_normalisation,
        subplot_columns,
    )

    # set colorbar limits
    set_colorbar_limits(
        plot,
        fields,
        colorbar_normalisation,
        colorbar_percentiles,
        subplot_rows,
        subplot_columns,
    )

    # export to matplotlib figure
    if colorbar_normalisation == "global":
        cbar_mode = "single"
    else:
        cbar_mode = "each"

    fig = plot.export_to_mpl_figure(
        (subplot_rows, subplot_columns), cbar_mode=cbar_mode, axes_pad=subplot_pad
    )

    if colorbar_normalisation == "row":
        remove_central_colorbar_ticks(fig, subplot_columns)

    # add labels
    labels = create_labels(planet_categories, weight_field)
    add_labels(fig, labels)
    fig.set_size_inches(*figsize)

    if save:
        if weight_field is None:
            weight = "None"
        else:
            weight = weight_field[-1]

        file_name = f"Planet_maps/planet_map_{normal}_weight={weight}"
        if figure_name_addon:
            file_name = f"{file_name}_{figure_name_addon}.pdf"
        else:
            file_name = f"{file_name}.pdf"
        path = Path().figures(f"{file_name}")

        # create directory if it doesn't exist already
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        fig.savefig(
            path,
            bbox_inches="tight",
        )
    return plot, fig


def planet_figure_name_formatting(
    host_star_masses: int | float | tuple | list | np.ndarray,
) -> str:
    """
    Create appropriate figure naming scheme dependent on input host_star_masses variable
    type.

    Parameters
    ----------
    host_star_masses : int | float | tuple | list | np.ndarray
        The masses of host star considered.

    Returns
    -------
    str
        Formatted naming scheme addon.

    """

    match host_star_masses:
        case int() | float():
            return f"masses={host_star_masses}"
        case tuple() | list() | np.ndarray():
            min_mass = np.amin(host_star_masses)
            max_mass = np.amax(host_star_masses)
            return f"masses={min_mass}-{max_mass}"
        case _:
            raise ValueError(
                "host_star_masses must either be a number of a list of " "numbers."
            )


def plot_configurations(
    plot: ParticleProjectionPlot,
    fields: list[tuple[str, str]],
    cmap: str,
    density_unit: str,
    figsize: tuple[float, float],
    fontdict: dict[str, Any],
    colorbar_normalisation: str,
    subplot_columns: int,
) -> None:
    """
    General plotting configuration

    Parameters
    ----------
    plot : ParticleProjectionPlot
        The yt plot object.
    fields : list[tuple[str,str]]
        List of fields.
    density_unit : str, optional
        Unit for spatial density.
    font_dict : dict[str, Any], optional
        Additional parameter for fonts.
    figsize : tuple[float, float]
        Size of figure.
    fontdict : dict[str, Any]
        Additional parameter for fonts.
    colorbar_normalisation : str, optional
        Normalisation of colorbars.
    subplot_columns : int, optional
        Number of subplot columns.

    """
    for i, field in enumerate(fields):
        plot.set_cmap(field, cmap)
        plot.set_unit(field, density_unit)
        if density_unit != "1/pc**2":
            warnings.warn("density_unit is not default. Need to adjust colorbar label.")

        if colorbar_normalisation == "row" and ((i + 1) % subplot_columns != 0):
            plot.set_colorbar_label(field, r"")
        else:
            plot.set_colorbar_label(
                field, r"Surface Density $\left(1/\mathrm{pc}^2\right)$"
            )
        plot.set_background_color(field, "black")
        plot.set_figure_size(figsize)
        plot.set_font(fontdict)


def set_colorbar_limits(
    plot: ParticleProjectionPlot,
    fields: list[tuple[str, str]],
    colorbar_normalisation: str,
    colorbar_percentiles: tuple[float, float],
    subplot_rows: int,
    subplot_columns: int,
) -> None:
    """
    Set limits for colorbar using percentiles of data, ignores 0s. Following modes:
        inidividual:
            Calculate limit for each subplot individually.
        row:
            Calculate value along rows of subplots.
        global:
            Calculate values for all subplots.

    Parameters
    ----------
    plot : ParticleProjectionPlot
        The yt plot object.
    fields : list[tuple[str,str]]
        List of fields.
    colorbar_normalisation : str
        The normalisation mode, must be 'individual', 'row' or 'global'.
    colorbar_percentiles : tuple[float, float]
        Percentiles to calculate on data, used as limits.
    subplot_rows : int
        Number of subplots rows.
    subplot_columns : int
        Number of subplot columns.

    """
    image_values = np.array([np.array(plot.frb[field]).flatten() for field in fields])
    image_values[image_values == 0] = np.nan

    if colorbar_normalisation == "individual":
        colorbar_limits = np.nanpercentile(image_values, colorbar_percentiles, axis=1).T

    elif colorbar_normalisation == "row":
        if len(image_values) != subplot_rows * subplot_columns:
            raise NotImplementedError(
                "Row wise colorbars currently only work if all rows are filled."
            )
        image_values = image_values.reshape(subplot_rows, subplot_columns, -1).reshape(
            subplot_rows, -1
        )
        percentiles = np.nanpercentile(image_values, colorbar_percentiles, axis=1)
        colorbar_limits = np.repeat(percentiles, 3, axis=1).T

    elif colorbar_normalisation == "global":
        colorbar_limits = (
            np.repeat(np.nanpercentile(image_values, colorbar_percentiles), len(fields))
            .reshape(2, -1)
            .T
        )
    else:
        raise ValueError("colorbar mode must be 'individual', 'row' or 'global'.")

    for field, limits in zip(fields, colorbar_limits):
        plot.set_zlim(field, *limits)


def remove_central_colorbar_ticks(fig: Figure, subplot_columns: int) -> None:
    """
    Deactivate ticks for all colorbars that are not on the outer right edge of the row.

    Parameters
    ----------
    fig : Figure
        The matplotlib figuure object.
    subplot_columns : int
        Number of subplot columns.

    """
    _, cbar_axes = filter_subplot_axes(fig)
    for i, ax in enumerate(cbar_axes):
        # if not at end of row, delete colorbar
        if (i + 1) % subplot_columns != 0:
            ax.set_xticks([])
            ax.set_yticks([])


def create_labels(
    planet_categories: list[str],
    weight_field: Optional[tuple[str, str]] = None,
) -> list[str]:
    """
    Create labels for each subplot, adapted by weight_field used.

    Parameters
    ----------
    planet_categories : list[str]
        List of planet categories.
    weight_field : Optional[tuple[str,str]], optional
        Field to weight planet fields by. The default is None.

    Returns
    -------
    list[str]
        List of labels for each plot.

    """
    if weight_field is None:
        label_appendix = r"s"
    elif weight_field == ("stars", "number"):
        label_appendix = r"s Per Star"
    else:
        label_appendix = f"s (weighted by {weight_field[-1]})"
    return [f"{category}{label_appendix}" for category in planet_categories]


def add_labels(fig: Figure, labels: list[str]) -> None:
    """
    Add labels to plots as text in the upper left corner. Matches font and fontsize of
    labels.

    Parameters
    ----------
    fig : Figure
        The matplotlib figuure object.
    list[str]
        List of labels for each plot.

    """
    subplot_axes, _ = filter_subplot_axes(fig)
    for ax, label in zip(subplot_axes, labels):
        label_font = ax.xaxis.get_label().get_fontproperties()
        ax.text(
            0.03,
            0.97,
            label,
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=label_font.get_size(),
            fontname=label_font.get_name(),
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round"),
        )


def filter_subplot_axes(fig: Figure) -> tuple[list[Axes], list[Axes]]:
    """
    Filter out axes that are subplots and colorbars from fig object.

    Parameters
    ----------
    fig : Figure
        The matplotlib figuure object.

    Returns
    -------
    subplot_axes : list[Axes]
        List of subplot axes.
    cbar_axes : list [Axes]
        List of colorbar axes.

    """
    subplot_axes = []
    cbar_axes = []
    for ax in fig.axes:
        # check for colorbars
        if isinstance(ax, mpl_toolkits.axes_grid1.axes_grid.CbarAxesBase):
            cbar_axes.append(ax)
        #  check for plot axes
        elif isinstance(ax, mpl_toolkits.axes_grid1.mpl_axes.Axes):
            subplot_axes.append(ax)
        else:
            raise RuntimeError(
                "Axis type not understood, must either be mpl_axes.Axes or "
                "axes_grid.CbarAxesBase."
            )
    return subplot_axes, cbar_axes
