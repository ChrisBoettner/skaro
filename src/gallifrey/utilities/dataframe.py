#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:02:46 2023

@author: chris
"""
from typing import Optional

import pandas as pd


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
            "log_inner_edge": r"log $r_\mathrm{in}$ [AU]",
            "log_photoevaporation": r"log $\dot{M}_\mathrm{wind}$ [$M_\odot$/yr]",
        }

    dataframe = dataframe.rename(columns=mapping_dict)
    labels = [label for label in dataframe.columns if label in mapping_dict.values()]
    return dataframe, labels
