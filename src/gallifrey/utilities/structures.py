#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:02:09 2023

@author: chris
"""
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


def find_closest(
    value_array: ArrayLike,
    reference_array: ArrayLike,
    is_sorted: bool = False,
) -> NDArray:
    """
    Find value in reference_array that is clostest to value in value_array.

    Parameters
    ----------
    value_array : ArrayLike
        The array of values to be matched to reference_array.
    reference_array : ArrayLike
        The array of reference values.
    is_sorted : bool, optional
        The reference array needs to be sorted. If False, sort array first. The
        default is False.

    Returns
    -------
    NDArray
        DESCRIPTION.

    """
    value_array = np.asarray(value_array)
    reference_array = np.asarray(reference_array)
    if not is_sorted:
        reference_array = np.sort(reference_array)

    # searchsorted looks for the spot in which one would need to insert the value
    # to keep the array sorted
    indices = np.searchsorted(reference_array, value_array)
    # clip so no value falls outside
    indices = np.clip(indices, 1, len(reference_array) - 1)

    # get left and right neighbours, look which one is closer to value_array
    left_neighbour = reference_array[indices - 1]
    right_neighbour = reference_array[indices]
    closest_index = np.where(
        np.abs(left_neighbour - value_array) <= np.abs(right_neighbour - value_array),
        indices - 1,
        indices,
    )
    return reference_array[closest_index]
