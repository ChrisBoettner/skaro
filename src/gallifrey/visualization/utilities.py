#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:34:21 2023

@author: chris
"""

from typing import Type

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt import FigureManagerQT
from yt.visualization.plot_window import NormalPlot


class FigureManager:
    """Custom Figure Manager to deal with yt plot interactively."""

    def __init__(self) -> None:
        """
        Initilize empty list of managers and counter for figures.
        """
        self.num = 0
        self.managers: list[FigureManagerQT] = []

    def show(self, ytPlot: Type[NormalPlot]) -> None:
        """
        Show plots by creating new figure manager and appending it to list. Works
        also if yt Plot object contains multiple plots.

        Parameters
        ----------
        ytPlot : Type[NormalPlot]
            yt Plot object.

        """
        plots = list(getattr(ytPlot, "plots").values())
        for plot in plots:
            # create new figure manager
            new_manager = plt._backend_mod.new_figure_manager_given_figure(
                num=self.num, figure=plot.figure
            )
            # connect manager and figure
            new_manager.canvas.figure = plot.figure
            plot.figure.set_canvas(new_manager.canvas)
            # append manager to list of managers
            self.managers.append(new_manager)
            # increment counter
            self.num += 1

    def close(self, which: str = "latest") -> None:
        """
        Closes figure. Either 'latest' or 'all'

        Parameters
        ----------
        which : str, optional
            Choose which figure to close, 'latest' or 'all'. The default is 'latest'.

        """
        if len(self.managers) == 0:
            raise AttributeError("No plots found in FigureManager instance.")

        if which == "latest":
            self.managers[-1].destroy()
            self.managers.pop()

        elif which == "all":
            while len(self.managers) > 0:
                self.close("latest")

        else:
            raise ValueError("Choose either 'latest' or 'all'.")
