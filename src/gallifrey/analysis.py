#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:00:52 2023

@author: chris
"""
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yt

from skaro.model import Model
from skaro.utilities.dataframe import aggregated_dataframe, rename_entries
from skaro.utilities.logging import logger

# create Logger
logger = logger(__name__)


def count_planets(
    model: "Model",
    data_creator: Callable,
    planet_categories: str | list[str],
    normalize_by: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    components: Optional[str | list[str]] = None,
    long_format: Optional[bool] = False,
    variable_name: str = "Planet Type",
    value_name: str = "Number of Planets",
    rename_components: bool = True,
    description: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Calculate occurence rate of planet of planet types in sphere around the model halo.
    The planet_type fields must have been created for the HESTIA snapshot in question.

    Parameters
    ----------
    model : Model
        The model object that contains the snapshot and halo.
    data_creator : Callable
        A function that creates the data yt source object. Must be Callable, rather than
        a variable, since model.update_fields doesn't update previously created
        data_source objects.
    planet_categories : str | list[str]
        The planet types to calculate the occurence rate for.
    normalize_by: Optional[str], optional
        If given, normalize number of planets by dividing through summed field value
        named by normalized_by, e.g. 'planet_hosting_number' for occurence rates. The
        default is None.
    model_config : Optional[dict[str, Any]], optional
        An optional model config that updates some of the model parameter and the
        corresponding field values for the planet types. The default is None.
    components : Optional[str | list[str]], optional
        The galaxy components components to calculate the occurence rates for. The
        default is None, which defaults to the standard decomposition. Use "stars" for
        all star particles within the sphere.
    long_format: bool, optional
        Choose if dataframe should be returned in long format or not. The default is
        False.
    variable_name: str, optional
        Name of the variable field, if the dataframe is returned in long format. The
        default is "Planet Type".
    value_name: str, optional
        Name of the value field, if the dataframe is returned in long format. The
        default is "Number of Planets".
    rename_components: bool, optional
        If True, rename components using skaro.utilities.dataframe.rename_entries.
    description: Optional[tuple[str, Any]], optional
        Adds an optional description of the dataframe by appending columns with a
        decriptor value. Must be of form {column name, entry value}. The default is
        None.

    Returns
    -------
    planet_counts : pd.DataFrame
        The dataframe containing the planet counts (or occurence rates, if
        occurence_rates = True). In long format if long_format=True.

    """
    # make defaults iterable
    if isinstance(planet_categories, str):
        planet_categories = [planet_categories]

    if model_config is None:
        model_config = {}

    if components is None:
        components = [
            "bulge_stars",
            "thin_disk_stars",
            "thick_disk_stars",
            "halo_stars",
        ]
    if isinstance(components, str):
        components = [components]

    # update fields and create data objects
    model.update_fields(**model_config)

    data_source = data_creator()

    # query data object for planets and create dataframe
    column_list = [*planet_categories]
    if normalize_by is not None:
        column_list.append(normalize_by)

    data = aggregated_dataframe(
        components,
        column_list,
        data_source=data_source,
        type_name="Component",
    )

    if rename_components:
        # rename components, e.g. halo_stars to Halo, etc.
        data = rename_entries(data)

    # count planets (per component)
    planet_counts = data.groupby("Component")[planet_categories].sum()

    if normalize_by is not None:
        # normalise by seperate column
        normalization = data.groupby("Component")[normalize_by].sum()
        planet_counts = planet_counts.div(normalization, axis=0).reset_index()

    if long_format:
        # return in long format
        planet_counts = planet_counts.melt(
            id_vars="Component",
            var_name=variable_name,
            value_name=value_name,
        )

    if description is not None:
        # add additional description columns
        for key, value in description.items():
            planet_counts[key] = value

    return planet_counts


def bin_planets(
    model: "Model",
    data_creator: Callable,
    planet_categories: str | list[str],
    bin_field: str = "particle_radius",
    model_config: Optional[dict[str, Any]] = None,
    num_bins: int = 100,
    bin_limits: Optional[tuple[float, float]] = None,
    log_bins: bool = False,
    density: Optional[str | Callable] = None,
    normalize: Optional[str] = None,
    cumulative: bool = False,
    component: str = "stars",
    base_units: str = "galactic",
    bin_field_units: Optional[str] = None,
    long_format: Optional[bool] = False,
    variable_name: str = "Planet Type",
    value_name: str = "Number of Planets",
    description: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin planet fields according to some bin_field using the yt create_profiles function
    and return dataframe containing binned profiles.The planet_type fields must have
    been created for the HESTIA snapshot in question.

    Parameters
    ----------
    model : Model
        The model object that contains the snapshot and halo.
    data_creator : Callable
        A function that creates the data yt source object. Must be Callable, rather than
        a variable, since model.update_fields doesn't update previously created
        data_source objects.
    planet_categories : str | list[str]
        The planet types to calculate the occurence rate for.
    bin_field : str, optional
        The field by which to bin the values by. The default is "particle_radius".
    model_config : Optional[dict[str, Any]], optional
        An optional model config that updates some of the model parameter and the
        corresponding field values for the planet types. The default is None.
    num_bins : int, optional
        The number of bins. The default is 100.
    bin_limits: Optional[tuple(float, float)], optional
        Limits of the bins. If given, must be of form (lower limit, upper limit). The
        default is None, which uses the data limits.
    log_bins : bool, False
        Choose if bins should be created in logspace or linear space. The default is
        False.
    density: Optional[str | Callable], optional
        If given a string, must be either "width" or "spherical". In case of "width",
        divide value of bin by width of bin. If "spherical", divide by volume of
        spherical shell using bin width. Alternatively, a function can be passed
        that takes the bin edges as input and returns the volume as a numpy array.
        The default is None.
    normalize: Optional[str], optional
        If given a string, must be either "first_bin" or "area". If "first_bin", divide
        columns by value in first bin, so that the profiles start at 1. If "area",
        divide by total area. The default is False.
    cumulative: bool optional
        If True, accumulate values along bins. The default is False
    component : str, optional
        The galaxy component to look at (main field name in yt data source). The
        default is "stars".
    base_units : str, optional
        The base unit system to convert return the bin field values in. The default
        is "galactic".
    bin_field_units : Optional[str], optional
        Custom units for the bin field, if base units are not wanted. The default is
        None.
    long_format: bool, optional
        Choose if dataframe should be returned in long format or not. The default is
        False.
    variable_name: str, optionbin_limitsal
        Name of the variable field, if the dataframe is returned in long format. The
        default is "Planet Type".
    value_name: str, optional
        Name of the value field, if the dataframe is returned in long format. The
        default is "Number of Planets".
    description: Optional[tuple[str, Any]], optional
        Adds an optional description of the dataframe by appending columns with a
        decriptor value. Must be of form {column name, entry value}. The default is
        None.
    **kwargs : Any
        Additional parameter passed to yt.create_profile.

    Returns
    -------
    profiles_dataframe : pd.DataFrame
        The profile values for each planet type in the planet categories.
    bins_dataframe : pd.DataFrame
        The lower and upper edges of the bin. The bin centres are the index.

    """
    # make defaults iterable
    if isinstance(planet_categories, str):
        planet_categories = [planet_categories]

    if model_config is None:
        model_config = {}

    # update fields and create data container
    model.update_fields(**model_config)

    data_source = data_creator()

    if "weight_field" not in kwargs.keys():
        # set default weight_field to None, else error occurs since actual default is
        # ('gas', 'mass')
        kwargs["weight_field"] = None

    # make bin limits understandable to yt
    if bin_limits is not None:
        extrema: Optional[dict] = {(component, bin_field): bin_limits}
    else:
        extrema = bin_limits

    # get correct units for bin field
    if not bin_field_units:
        bin_field_units = data_source[(component, bin_field)].in_base(base_units).units

    # create profiles using yt
    yt_profiles = yt.create_profile(
        data_source,
        [(component, bin_field)],
        [(component, planet_type) for planet_type in planet_categories],
        n_bins=num_bins,
        extrema=extrema,
        logs={(component, bin_field): log_bins},
        units={(component, bin_field): bin_field_units},
        **kwargs,
    )

    # turn profiles into array
    profiles = np.array(
        [yt_profiles[planet_type].value for planet_type in planet_categories]
    ).T

    # get bins
    bin_centres = yt_profiles.x
    bin_edges = yt_profiles.x_bins

    # normalize by some value, if wanted
    volume: Optional[np.ndarray] = None
    if density is not None:
        if isinstance(density, str) and density.lower() == "width":
            volume = bin_edges[1:] - bin_edges[:-1]
        elif isinstance(density, str) and density.lower() == "spherical":
            volume = 4 / 3 * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
        elif callable(density):
            volume = density(bin_edges)
            if not isinstance(volume, np.ndarray):
                raise ValueError("'density' callable must return numpy array.")
        else:
            raise ValueError("'density' keyword not known.")
        if isinstance(volume, np.ndarray):
            profiles = profiles / volume[:, np.newaxis]
        else:
            raise ValueError("Calculated volumes are not a numpy array.")

    # create profile and bin dataframes
    profiles_dataframe = pd.DataFrame(profiles, columns=planet_categories)

    # normalize columns
    if normalize is not None:
        if normalize.lower() == "first_bin":
            # divide by first row
            profiles_dataframe = profiles_dataframe.div(profiles_dataframe.iloc[0])
        elif normalize.lower() == "area":
            # divide by whole area
            if density is not None:
                logger.warn(
                    "WARNING: 'density' keyword has no effect if normalize='area'."
                )
                if isinstance(volume, np.ndarray):
                    profiles = profiles * volume[:, np.newaxis]
                else:
                    raise ValueError("Calculated volumes are not a numpy array.")
            area = profiles.sum(axis=0)
            profiles_dataframe = profiles_dataframe.div(area)
        else:
            raise ValueError("'normalize' keyword not known.")

    if cumulative:
        # accumulate along bins
        profiles_dataframe = profiles_dataframe.cumsum(axis=0)

    # add bin fields and create bin dataframe
    profiles_dataframe[bin_field] = bin_centres.value
    bins_dataframe = pd.DataFrame(
        {
            bin_field: bin_centres,
            "lower_bin_edge": bin_edges[:-1],
            "upper_bin_edge": bin_edges[1:],
        }
    ).drop_duplicates(subset=[bin_field])
    bins_dataframe = bins_dataframe.set_index(bin_field)

    if long_format:
        # return in long format
        profiles_dataframe = profiles_dataframe.melt(
            id_vars=bin_field,
            var_name=variable_name,
            value_name=value_name,
        )

    if description is not None:
        # add additional description column
        for key, value in description.items():
            profiles_dataframe[key] = value

    return profiles_dataframe, bins_dataframe
