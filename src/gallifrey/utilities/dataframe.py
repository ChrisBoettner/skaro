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
        Name of the column that contains the particle type.

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
            data_source[particle_type, field_values[0]], columns=[particle_type]
        ).melt(var_name=type_name, value_name=field_values[0])
        for particle_type in particle_types
    ]
    dataframe = pd.concat(dataframes)

    # add potential further field values
    if len(field_values) > 1:
        for field_value in field_values[1:]:
            dataframe[field_value] = flatten_list(
                [
                    data_source[particle_type, field_value].value
                    for particle_type in particle_types
                ]
            )
    return dataframe


def rename_labels(
    dataframe: pd.DataFrame, mapping_dict: Optional[dict] = None
) -> tuple[pd.DataFrame, list]:
    """
    Rename columns of a dataframe based on a mapping dictionary and return dataframe
    with changed labels, as well as a list of labels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe whose columns are renamed.
    mapping_dict : Optional[dict], optional
        The mapping dictonary containing old column names as keys and new values as
        labels. The default is None, in which case a dictonary for the parameter names
        is used.

    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe with renamed columns.
    labels : list
        A list of the applied new column names.

    """

    if mapping_dict is None:
        mapping_dict = {
            "log_initial_mass": r"log $M_\mathrm{g}$ [$M_\odot$]",
            "[Fe/H]": "[Fe/H]",
            "[alpha/Fe]": r"[$\mathrm{\alpha}$/Fe]",
            "log_inner_edge": r"log $r_\mathrm{in}$ [AU]",
            "log_photoevaporation": (
                r"log $\dot{M}_\mathrm{wind}$ [$\frac{M_\odot}{\mathrm{yr}}$]"
            ),
            "log_solid_mass": r"log $M_\mathrm{s}$ [$M_\mathrm{Jupiter}$]",
        }

    dataframe = dataframe.rename(columns=mapping_dict)
    labels = [label for label in dataframe.columns if label in mapping_dict.values()]
    return dataframe, labels
