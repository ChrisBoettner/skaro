#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:02:09 2023

@author: chris
"""
from collections import Counter
from functools import lru_cache
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


@lru_cache(maxsize=8)
def count_list_occurences(input_list: list[Any]) -> Counter:
    """
    Create Counter object that counts how often elements occur in list.

    Parameters
    ----------
    input_list : list[Any]
        The input list.

    Returns
    -------
    Counter
        Counter object. Call on list items to check how often item has occured in list.

    """
    return Counter(input_list)


def find_closest(
    value_array: ArrayLike,
    reference_array: ArrayLike,
    is_sorted: bool = False,
    return_index: bool = False,
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
    return_index: bool, optional
        If True, return index of clostest value rather than clostest value itself. The
        default is False.

    Returns
    -------
    NDArray
        Clostest matching values or indices of clostest matching values.

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

    if return_index:
        return closest_index
    return reference_array[closest_index]


@lru_cache(maxsize=8)
def make_geomspace(
    start: tuple | float,
    stop: tuple | float,
    num: int = 50,
    **kwargs: Any,
) -> ArrayLike:
    """
    Create geometric space using lru cache for efficiency. For this to work, input
    needs to be hashable, meaning arrays need to be converted to tuples before passing

    Parameters
    ----------
    start : tuple|float
        The starting value of the sequence.
    stop : tuple|float
        The final value of the sequence, unless endpoint is False. In that case,
        num + 1 values are spaced over the interval in log-space, of which all but
        the last (a sequence of length num) are returned.
    num : int, optional
        Number of samples to generate. The default is 50.
    **kwargs : Any
        Additional arguments passed to np.geomspace.

    Returns
    -------
    ArrayLike
        num samples, equally spaced on a log scale.

    """
    return np.geomspace(start, stop, num, **kwargs)


def make_meshgrid(
    bounds: list[tuple],
    num_bins: int = 10,
    as_list: bool = False,
) -> list[np.ndarray]:
    """
    Make meshgrid based on list of bounds. bounds must be a list of 2-tuples, where
    the first value is the lower bound and the second value is the upper bound.

    Parameters
    ----------
    bounds : list[tuple]
        List of bounds to create the grid from.
    num_bins : int, optional
        Number of bins for all dimensions. The default is 10.
    as_list : bool, optional
        If True, return a array of coordinate pairs, rather than meshgrid. The
        default is False.

    Returns
    -------
    meshgrid : list[np.ndarray]
        The list of arrays that make up the meshgrid if as_list is False. If as_list
        is True, returns list of coordinate pairs.

    """
    # create linspaces from bounds
    ranges = [np.linspace(b[0], b[1], num_bins) for b in bounds]

    # create the meshgrid
    meshgrid = np.meshgrid(*ranges)

    if as_list:
        # convert to list of coordinate pairs
        meshgrid = list(np.column_stack([m.ravel() for m in meshgrid]))
    return meshgrid
