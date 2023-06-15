#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:02:09 2023

@author: chris
"""
from typing import Any

def flatten_list(nested_list: list[list]) -> list[Any]:
    """
    Flatten sublists in list.

    Parameters
    ----------
    nested_list : list[list]
        Nested list.

    Returns
    -------
    list[Any]
        Flattened list.

    """
    return [item for sublist in nested_list for item in sublist]