#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:02:46 2023

@author: chris
"""
from typing import Optional

import pandas as pd
from yt.data_objects.data_containers import YTDataContainer
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.utilities.structures import flatten_list


def aggregated_dataframe(
    particle_types: str | list[str],
    field_values: str | list[str],
    data_source: YTDataContainer | YTDataContainerDataset,
    type_name: str = "Type",
    base_units: str = "galactic",
    custom_units: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Create dataframe with different field values, annotated by particle type from
    yt data source.

    Parameters
    ----------
    particle_types : str| list[str]
        The particle type(s) to include.
    field_values : str| list[str]
        The field value(s) to fetch.
    data_source : YTDataContainer|YTDataContainerDataset
        The data source.
    type_name : str, optional
        Name of the column that contains the particle type. The default is 'Type'.
    base_units : str, optional
        Sets the unit system in which data should be returned by unyt. The default is
        'galactic', corresponding to kpc, Myr, Msun, etc.
    custom_units : dict[str,str], optional
        Change units for some specific field_values. Input must be a dictionary with
        the key being the field_value name and the value being the units. The default is
        None.

    Returns
    -------
    dataframe : pd.DataFrame
        The aggregated dataframe.

    """
    # make input iterable
    if isinstance(particle_types, str):
        particle_types = [particle_types]
    if isinstance(field_values, str):
        field_values = [field_values]

    # create dataframe from first value in such a way that the dataframe gets an
    # additional column "Type" that informs about the particle type
    dataframes = [
        pd.DataFrame(
            data_source[particle_type, field_values[0]].value,
            columns=[particle_type],
        ).melt(var_name=type_name, value_name=field_values[0])
        for particle_type in particle_types
    ]
    dataframe = pd.concat(dataframes)

    # add potential further field values
    for field_value in field_values:
        # use custon units if available
        if (custom_units is not None) and (field_value in custom_units.keys()):
            data = [
                data_source[particle_type, field_value]
                .to(custom_units[field_value])
                .value
                for particle_type in particle_types
            ]
        # otherwise use default units
        else:
            data = [
                data_source[particle_type, field_value].in_base(base_units).value
                for particle_type in particle_types
            ]

        dataframe[field_value] = flatten_list(data)
    return dataframe


def rename_labels(
    dataframe: pd.DataFrame, mapping_dict: Optional[dict] = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rename columns of a dataframe based on a mapping dictionary and return dataframe
    with changed labels, as well as a list of labels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe whose columns are renamed.
    mapping_dict : Optional[dict], optional
        The mapping dictonary containing old column names as keys and new values as
        labels. The default is None, in which case a default dictonary for the parameter
        names is used.

    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe with renamed columns.
    label_dict : list
        A dictionary containing the old label (keys) and new
        labels (columns).

    """

    if mapping_dict is None:
        mapping_dict = {
            "log_initial_mass": r"log $M_\mathrm{g}$ ($M_\odot$)",
            "[Fe/H]": "[Fe/H]",
            "log_inner_edge": r"log $r_\mathrm{in}$ (AU)",
            "log_photoevaporation": (
                r"log $\dot{M}_\mathrm{wind}$"
                r"$\left(\frac{M_\odot}{\mathrm{yr}}\right)$"
            ),
            "log_solid_mass": r"log $M_\mathrm{s}$ ($M_\mathrm{Jupiter}$)",
            "[alpha/Fe]": r"[$\mathrm{\alpha}$/Fe]",
            "stellar_age": "Stellar Age (Gyr)",
            "particle_radius": "Distance (kpc)",
        }

    dataframe = dataframe.rename(columns=mapping_dict)
    label_dict = {
        key: value for key, value in mapping_dict.items() if value in dataframe.columns
    }
    return dataframe, label_dict


def rename_entries(
    dataframe: pd.DataFrame,
    column: str = "Component",
    mapping_dict: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Rename entries in column of a dataframe based on a mapping dictionary
    and return dataframe with changed names. Convenience function to translate between
    yt field names to names that are nicer for plotting. The defaults are set to rename
    the galaxy components.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe whose entries are renamed.
    column : str, optional
        The name of the column that contains the galaxy component names. The default is
        "Component".
    mapping_dict : dict[str,str], optional
        The dictonary that translates between yt field names and new names. The default
        is None, in which case a default dictonary for the galaxy component names is
        used.


    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe with renamed galaxy component entries.

    """
    if mapping_dict is None:
        mapping_dict = {
            "stars": "Stars",
            "bulge_stars": "Bulge",
            "thin_disk_stars": "Thin Disk",
            "thick_disk_stars": "Thick Disk",
            "halo_stars": "Halo",
        }

    dataframe[column] = dataframe[column].apply(lambda entry: mapping_dict[entry])
    return dataframe


def within_bounds(
    dataframe: pd.DataFrame,
    columns: list[str],
    bounds: dict[str, tuple[float, float]],
    condition: str,
) -> pd.Series:
    """
    Check if values in a dataframe fall within some given bounds.

    The bounds condition can be
        'none':
            No bound conditions are applied, returns Series of True.
        'lower':
            Series contains True if all values in row are above the lower bound,
            otherwise False.
        'upper':
            Series contains True if all values in row are below the upper bound,
            otherwise False.
        'both':
            Series contains True if all values in row are between the bounds,
            otherwise False.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe.
    columns : list[str]
        A list of columns that are checked for falling within the bounds.
    bounds : dict[str, tuple[float, float]]
        A dictonary of bounds, must have the column names as keys and corresponding
        bounds as values in form of a tuple (lower_bound, upper_bound).
    condition : str
        The condition that the dataframe values are checked against. Must be 'none',
        'lower', 'upper' or 'both'.

    Returns
    -------
    pd.Series
        A pandas series containing 'True' if all row values fall within bounds and
        'False' if they do not.

    """

    if not all([column in dataframe.columns for column in columns]):
        raise ValueError("All values in 'columns' must be columns in the DataFrame")

    if not isinstance(bounds, dict):
        raise ValueError(
            "bounds must be a dict, with the columns as keys and "
            "a tuple (lower_bound, upper_bound) as values."
        )

    if not all([column in bounds.keys() for column in columns]):
        raise ValueError(
            "All values in 'column' must have corresponding bounds in "
            "the bounds dictionary."
        )

    if (not isinstance(condition, str)) or (
        condition.lower() not in ["none", "lower", "upper", "both"]
    ):
        raise ValueError(
            "bounds condition must be 'none', 'lower', 'upper', or 'both'."
        )
    condition = condition.lower()  # make case-insensitive

    if condition == "none":
        return pd.Series([True] * len(dataframe), index=dataframe.index)

    is_within = []
    for column in columns:
        if condition == "lower":
            is_within.append(dataframe[column] > bounds[column][0])
        elif condition == "upper":
            is_within.append(dataframe[column] < bounds[column][1])
        elif condition == "both":
            is_within.append(dataframe[column].between(*bounds[column]))

    return pd.concat(is_within, axis=1).all(axis=1)
