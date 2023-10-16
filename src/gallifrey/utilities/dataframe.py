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


def rename_galaxy_components(
    dataframe: pd.DataFrame,
    key: str = "Component",
    mapping_dict: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Rename galaxy component names of a dataframe based on a mapping dictionary
    and return dataframe with changed names. Convenience function to translate between
    yt field names to names that are nicer for plotting.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe whose entries are renamed.
    key : str, optional
        The name of the column that contains the galaxy component names. The default is
        "Component".
    mapping_dict : dict[str,str], optional
        The dictonary that translates between yt field names and new names. The default
        is None, in which case a default dictonary for the parameter
        names is used.


    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe with renamed galaxy component entries.

    """
    if mapping_dict is None:
        mapping_dict = {
            "bulge_stars": "Bulge",
            "thin_disk_stars": "Thin Disk",
            "thick_disk_stars": "Thick Disk",
            "halo_stars": "Halo",
        }

    dataframe[key] = dataframe[key].apply(lambda entry: mapping_dict[entry])
    return dataframe
