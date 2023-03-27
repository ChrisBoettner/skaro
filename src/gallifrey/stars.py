#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:04:22 2023

@author: chris
"""
import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from gallifrey.data.paths import Path


class StellarModel:
    def __init__(self) -> None:
        """
        Initialize.
        """

        # input data
        path = Path().external_data(r"stellar_main_sequence_parameter.csv")

        self.stellar_parameter = pd.read_csv(path)

        self.log_stellar_parameter = self.stellar_parameter.copy()
        self.log_stellar_parameter.iloc[:, 1:] = np.log10(
            self.stellar_parameter.iloc[:, 1:]
        )

    def lifetime(self, m: ArrayLike) -> float | NDArray:
        """
        Calculate lifetime from mass.

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        float | NDArray
            Lifetime of stars.

        """
        return self.calculate_interpolated_quantity(m, target_quantity="lifetime")

    def luminosity(self, m: ArrayLike) -> float | NDArray:
        """
        Calculate luminosity from mass.

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        float | NDArray
            Luminosity of stars.

        """
        return self.calculate_interpolated_quantity(m, target_quantity="luminosity")

    def temperature(self, m: ArrayLike) -> float | NDArray:
        """
        Calculate temperature from mass.

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        float | NDArray
            Temperature of stars.

        """
        return self.calculate_interpolated_quantity(m, target_quantity="temperature")

    def mass_from_lifetime(self, lifetime: ArrayLike) -> float | NDArray:
        """
        Calculate mass from lifetime.

        Parameters
        ----------
        lifetime : ArrayLike
            Lifetime of stars.

        Returns
        -------
        float | NDArray
            Mass of stars.

        """
        return self.calculate_interpolated_quantity(
            lifetime, target_quantity="mass", input_quantity="lifetime", reverse=True
        )

    def mass_from_luminosity(self, lifetime: ArrayLike) -> float | NDArray:
        """
        Calculate mass from luminosity.

        Parameters
        ----------
        luminosity : ArrayLike
            Luminosity of stars.

        Returns
        -------
        float | NDArray
            Mass of stars.

        """
        return self.calculate_interpolated_quantity(
            lifetime, target_quantity="mass", input_quantity="luminosity"
        )

    def mass_from_temperature(self, lifetime: ArrayLike) -> float | NDArray:
        """
        Calculate mass from temperature.

        Parameters
        ----------
        temperature : ArrayLike
            Temperature of stars.

        Returns
        -------
        float | NDArray
            Mass of stars.

        """
        return self.calculate_interpolated_quantity(
            lifetime, target_quantity="mass", input_quantity="temperature"
        )

    def calculate_interpolated_quantity(
        self,
        value: ArrayLike,
        target_quantity: str,
        input_quantity: str = "mass",
        reverse: bool = False,
    ) -> float | NDArray:
        """


        Parameters
        ----------
        value : ArrayLike
            Input values.
        target_quantity : str
            Name of output quantity.
        input_quantity : str, optional
            Name of input quantity. The default is "mass".
        reverse : bool, optional
            Choose if arrays should be reversed. Necessary if input_quantity is lifetime
            since the lifetime decreases with mass, and np.interp requires strictly
            increasing values. The default is False.

        Returns
        -------
        float | NDArray
            Output array.

        """
        log_value = np.log10(value)
        if input_quantity == "lifetime" and (not reverse):
            warnings.warn(
                "If input quantity is lifetime, reverse should be true in "
                "order for interpolation to work."
            )

        if reverse:
            log_quantity = np.interp(
                log_value,
                self.log_stellar_parameter[input_quantity][::-1],
                self.log_stellar_parameter[target_quantity][::-1],
            )
        else:
            log_quantity = np.interp(
                log_value,
                self.log_stellar_parameter[input_quantity],
                self.log_stellar_parameter[target_quantity],
            )
        return np.power(10, log_quantity)
