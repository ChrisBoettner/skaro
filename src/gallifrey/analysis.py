#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:00:52 2023

@author: chris
"""
from typing import Any, Callable, Optional

import pandas as pd

from gallifrey.model import Model
from gallifrey.utilities.dataframe import aggregated_dataframe, rename_entries


def count_planets(
    model: "Model",
    data_creator: Callable,
    planet_categories: str | list[str],
    normalize_by: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    components: Optional[str | list[str]] = None,
    long_format: Optional[bool] = False,
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
    rename_components: bool, optional
        If True, rename components using gallifrey.utilities.dataframe.rename_entries.
    description: Optional[tuple[str, Any]], optional
        Adds an optional description of the dataframe by appending columns with a
        decriptor value. Must be of form {column name, entry value}. The default is
        None.

    Returns
    -------
    occurence_rate_long_format : pd.DataFrame
        The dataframe containing the planet counts (or occurence rates, if
        occurence_rates = True). In long format if long_format=True.

    """
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

    model.update_fields(**model_config)

    data_source = data_creator()

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
        data = rename_entries(data)

    planet_counts = data.groupby("Component")[planet_categories].sum()

    if normalize_by is not None:
        host_counts = data.groupby("Component")[normalize_by].sum()
        planet_counts = planet_counts.div(host_counts, axis=0).reset_index()

    if long_format:
        planet_counts = planet_counts.melt(
            id_vars="Component",
            var_name="Planet Type",
            value_name="Occurence Rate",
        )

    if description is not None:
        for key, value in description.items():
            planet_counts[key] = value

    return planet_counts
