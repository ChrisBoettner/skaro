#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:16:24 2023

@author: chris
"""

import copy
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.algorithms as algo
import seaborn.utils as utils
from seaborn.regression import _RegressionPlotter


class _LogYRegression(_RegressionPlotter):
    """
    Custom plotter for numeric independent variables with regression model, used for
    logyregplot.
    """

    def __init__(self, logy: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logy = logy

        if any((self.logistic, self.robust, self.lowess, self.logx)):
            raise ValueError(
                "logyregplot and _LogYRegression only work with polynomial regression "
                "and logx=False."
            )

    def fit_regression(
        self,
        ax: Optional[plt.Axes] = None,
        x_range: Optional[np.ndarray] = None,
        grid: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fit the regression model.

        For a list of parameters and descriptions, see seaborn regplot documentation.

        """
        # Create the grid for the regression
        if grid is None:
            if self.truncate:
                x_min, x_max = self.x_range
            else:
                if ax is None:
                    if x_range is None:
                        raise ValueError("x_range must be (x_min, x_max) not None.")
                    x_min, x_max = x_range
                else:
                    x_min, x_max = ax.get_xlim()
            grid = np.linspace(x_min, x_max, 100)
        ci = self.ci

        x = np.log10(self.x) if self.logx else self.x
        y = np.log10(self.y) if self.logy else self.y

        yhat, yhat_boots = self.fit_poly(x, y, grid, self.order)

        if self.logy:
            yhat = np.power(10, yhat)
            if yhat_boots is not None:
                yhat_boots = np.power(10, yhat_boots)

        # Compute the confidence interval at each grid point
        if ci is None:
            err_bands: Optional[np.ndarray] = None
        else:
            err_bands = utils.ci(yhat_boots, ci, axis=0)

        return grid, yhat, err_bands

    def fit_poly(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        order: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Custom polyfit function.

        Parameters
        ----------
        x : np.ndarray
            Independent values.
        y : np.ndarray
            Dependent values.
        grid : np.ndarray
            Grid to evaluate on fit on for plotting.
        order : int
            Order of polynomial.

        Returns
        -------
        np.ndarray
            Fit estimates.
        np.ndarray
            Bootstrapped confidence interval estimates.

        """

        def reg_func(_x: np.ndarray, _y: np.ndarray) -> np.ndarray:
            return np.polyval(np.polyfit(_x, _y, order), grid)

        yhat = reg_func(x, y)
        if self.ci is None:
            return yhat, None

        yhat_boots = algo.bootstrap(
            x, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed
        )
        return yhat, yhat_boots


def logyregplot(
    data: Optional[pd.DataFrame] = None,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    x_estimator: Optional[Callable] = None,
    x_bins: Optional[np.ndarray] = None,
    x_ci: str = "ci",
    scatter: bool = True,
    fit_reg: bool = True,
    ci: int = 95,
    n_boot: int = 1000,
    units: Optional[str] = None,
    seed: Optional[int] = None,
    order: int = 1,
    x_partial: Optional[str] = None,
    y_partial: Optional[str] = None,
    truncate: bool = True,
    dropna: bool = True,
    x_jitter: Optional[float] = None,
    y_jitter: Optional[float] = None,
    label: Optional[str] = None,
    color: Optional[Any] = None,
    marker: Optional[str] = "o",
    scatter_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Extension of seaborn regplot, but fits y values in logspace. Only works with polyfit
    (of any order). For other regression techiques, use default seaborn regplot.

    For a list of parameters and descriptions, see seaborn regplot documentation.

    """
    plotter = _LogYRegression(
        logy=True,
        x=x,
        y=y,
        data=data,
        x_estimator=x_estimator,
        x_bins=x_bins,
        x_ci=x_ci,
        scatter=scatter,
        fit_reg=fit_reg,
        ci=ci,
        n_boot=n_boot,
        units=units,
        seed=seed,
        order=order,
        logistic=False,
        lowess=False,
        robust=False,
        logx=False,
        x_partial=x_partial,
        y_partial=y_partial,
        truncate=truncate,
        dropna=dropna,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
        color=color,
        label=label,
    )

    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax
