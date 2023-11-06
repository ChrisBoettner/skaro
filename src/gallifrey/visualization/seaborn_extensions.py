#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:16:24 2023

@author: chris
"""

import copy
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn.regression import _RegressionPlotter


class _LogYRegression(_RegressionPlotter):
    """
    Custom plotter for numeric independent variables with regression model, used for
    logyregplot.

    """

    def __init__(self, *args: Any, logy: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logy = logy

    def fit_regression(
        self,
        ax: plt.Axes = None,
        x_range: Optional[tuple[float, float]] = None,
        grid: Optional[np.ndarray] = None,
    ) -> tuple[Iterable, np.ndarray, np.ndarray | None]:
        """
        Pass regression calculation to super class method after converting y to
        log10(y), then revert after regression calculation.

        Parameters
        ----------
        ax : plt.Axes, optional
            The Axes object containing the plot. The default is None.
        x_range : Optional[tuple(float, float)], optional
            The range over which to calculate the regression, must be (lower, upper).
            The default is None.
        grid : Optional[np.ndarray], optional
            The grid points to calculate the fitted regression on. The default is None.

        Returns
        -------
        grid : Iterable
            The grid points to calculate the fitted regression on.
        yhat : np.ndarray
            The regression estimator for the y values (fitted in log space).
        err_bands : np.ndarray
           The regression estimator for the y error bands (fitted in log space).

        """
        if self.logy:
            self.y: np.ndarray = np.log10(self.y)

        grid, yhat, err_bands = super().fit_regression(
            ax=ax, x_range=x_range, grid=grid
        )

        if not isinstance(grid, Iterable):
            raise ValueError("'grid' should be an iterable.")

        if self.logy:
            self.y = np.power(10, self.y)
            yhat = np.power(10, yhat)
            if err_bands is not None:
                err_bands = np.power(10, err_bands)

        return grid, yhat, err_bands


def logyregplot(
    data: pd.DataFrame,
    **kwargs: Any,
) -> plt.Axes:
    """
    Extension of seaborn regplot, but fits y values in logspace. Only works with polyfit
    (of any order). For other regression techniques, use default seaborn regplot.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy (“long-form”) dataframe where each column is a variable and each row is an
        observation..
    **kwargs : Any
        Further parameter passed to _LogYRegression (which inherits from
        _RegressionPlotter). See seaborn.regplot for more details.

    Returns
    -------
    ax : plt.Axes
        The Axes object containing the plot.

    """
    kwargs.setdefault("truncate", True)  # for consistency with seaborn regplot

    plotter = _LogYRegression(
        data=data,
        **{
            key: kwargs[key]
            for key in kwargs.keys()
            if key not in ["marker", "scatter_kws", "line_kws", "ax"]
        },
    )

    ax = kwargs.get("ax") or plt.gca()
    # kwargs.get("ax") returns None if "ax" keys is not present or corresponding value
    # is None. In any case, if kwargs.get("ax") is None, return plt.gca() instead.

    scatter_kws = copy.copy(kwargs.get("scatter_kws", {}))
    scatter_kws["marker"] = "o" if "marker" not in kwargs else kwargs["marker"]
    line_kws = copy.copy(kwargs.get("line_kws", {}))
    plotter.plot(ax, scatter_kws, line_kws)
    return ax
