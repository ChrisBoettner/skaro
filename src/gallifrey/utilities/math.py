#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:23:00 2023

@author: chris
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess


def calculate_rotation_matrix(
    initial_vector: NDArray,
    target_vector: NDArray,
) -> NDArray:
    """
    Calculate the rotation matrix that transforms initial vector to final vector.

    Parameters
    ----------
    initial_vector : ArrayLike
        The initial (3D) vector.
    target_vector : ArrayLike
        The target (3D) vector.

    Returns
    -------
    NDArray
        The 3x3 rotation matrix transforming the initial to the target vector.

    """
    initial_vector = np.asarray(initial_vector)
    target_vector = np.asarray(target_vector)

    # Normalize the input vectors
    initial_vector = initial_vector / np.linalg.norm(initial_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    # Calculate the rotation axis
    axis = np.cross(initial_vector, target_vector)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # The vectors are parallel; no rotation needed.
        return np.eye(3)
    axis = axis / axis_norm

    # Calculate the rotation angle
    angle = np.arccos(np.clip(np.dot(initial_vector, target_vector), -1.0, 1.0))

    # Form the rotation matrix using the axis-angle formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return rotation_matrix


def calculate_pca(data: NDArray, **kwargs: dict[str, Any]) -> PCA:
    """
    Calculate PCA on dataset.

    Parameters
    ----------
    data : NDArray
        The dataset.
    **kwargs : dict[str, Any]
        Additional parameter passed to sklearn PCA.

    Returns
    -------
    PCA
        Fitted PCA result.

    """
    pca = PCA(**kwargs)
    return pca.fit(data)


def calculate_smoothing_line(
    x: ArrayLike, y: ArrayLike, fraction: float = 0.1, **kwargs: dict[str, Any]
) -> NDArray:
    """
    Calculate PCA on dataset.

    Parameters
    ----------
    x : ArrayLike
        The x data.
    y : ArrayLike
        The y data.
    fraction : float
        The fraction of points to be included in the local estimate, controls
        smoothness.
    **kwargs : dict[str, Any]
        Additional parameter passed to statsmodels lowess.

    Returns
    -------
    NDArray
        2D array with smoothed line. first column contains the (sorted) x data, second
        line contains the smoothed y data.

    """
    return lowess(y, x, frac=fraction)
