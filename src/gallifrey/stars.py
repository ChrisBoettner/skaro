#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:04:22 2023

@author: chris
"""
import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from gallifrey.data.paths import Path


class StellarModel:
    """
    Stellar Model.
    """

    def __init__(self) -> None:
        """
        Initialize.
        """

        # stellar parameter
        self.stellar_parameter = pd.read_csv(
            Path().external_data(r"stellar_main_sequence_parameter.csv")
        )

        self.log_stellar_parameter = self.stellar_parameter.copy()
        self.log_stellar_parameter.iloc[:, 1:] = np.log10(
            self.stellar_parameter.iloc[:, 1:]
        )

        # HZ parameter
        self.hz_parameter = np.loadtxt(Path().external_data(r"HZ_coefficients.dat"))
        self.hz_data = np.loadtxt(Path().interim_data(r"HZ_distances.txt"))

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
        return self.calculate_interpolated_stellar_parameter(
            m, target_quantity="lifetime"
        )

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
        return self.calculate_interpolated_stellar_parameter(
            m, target_quantity="luminosity"
        )

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
        return self.calculate_interpolated_stellar_parameter(
            m, target_quantity="temperature"
        )

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
        return self.calculate_interpolated_stellar_parameter(
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
        return self.calculate_interpolated_stellar_parameter(
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
        return self.calculate_interpolated_stellar_parameter(
            lifetime, target_quantity="mass", input_quantity="temperature"
        )

    def inner_HZ(self, m: ArrayLike) -> NDArray:
        """
        Calculate (conservative estimate) of inner limit of HZ, based on Kopparapu 2014
        estimate for Runaway Greenhouse limit for 1 Earth-massed planet.

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        NDArray
            Inner HZ limit in AU.

        """
        inner_hz_parameter = self.hz_parameter[:, 1]
        return self.calculate_habitable_zone(m, inner_hz_parameter)

    def inner_HZ_inverse(
        self,
        dist: float | NDArray,
        precalculated: bool = True,
        **kwargs: Any,
    ) -> NDArray:
        """
        Inverse relation for the inner HZ calculation, returns mass of star for a
        given inner HZ radius.

        NOTE: If precalculated is True and distance is outside of interpolation range,
        it uses the

        Parameters
        ----------
        dist : ArrayLike
            Distance to inner HZ edge.
        precalculated : bool, optional
            Decide if pre-calculated table should be interpolated or not. Currently only
            interpolation is implemented. The default is True.
        kwargs: dict[Any, Any], optional
            Additional arguments passed to np.interp .

        Returns
        -------
        NDArray
            Infered masses of stars for given inner HZ distances.

        """
        if precalculated:
            return np.interp(dist, self.hz_data[:, 1], self.hz_data[:, 0], **kwargs)

        else:
            raise NotImplementedError(
                "Direct inner HZ inverse calculation not yet implemented."
            )

    def outer_HZ(self, m: ArrayLike) -> NDArray:
        """
        Calculate (conservative estimate) of outer limit of HZ, based on Kopparapu 2014
        estimate for Maximum Greenhouse limit for 1 Earth-massed planet.

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        NDArray
            Outer HZ limit in AU.

        """
        outer_hz_parameter = self.hz_parameter[:, 2]
        return self.calculate_habitable_zone(m, outer_hz_parameter)

    def calculate_interpolated_stellar_parameter(
        self,
        value: ArrayLike,
        target_quantity: str,
        input_quantity: str = "mass",
        reverse: bool = False,
    ) -> float | NDArray:
        """
        Calculate target quantity as function of input (and input quantity) by
        linearly interpolating the safed stellar parameter. The interpolation occurs in
        log space.

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

        input_data = self.log_stellar_parameter[input_quantity]
        target_data = self.log_stellar_parameter[target_quantity]
        if reverse:
            input_data = input_data[::-1]
            target_data = target_data[::-1]

        log_quantity = np.interp(log_value, input_data, target_data)
        return np.power(10, log_quantity)

    def calculate_habitable_zone(
        self, m: ArrayLike, parameter: NDArray, temperature_sun: float = 5780
    ) -> NDArray:
        """
        Calculate edges of habitable zone for given mass of star and a set of parameter,
        based on Kopparapu 2014.

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.
        parameter : NDArray
            Parameter for effective flux calculation.
        temperature_sun : float, optional
            Temperature of the su. The default is 5780.

        Returns
        -------
        NDArray
            Distance to edge of habitable zone in AU.

        """
        # calculate luminosity and effective temperature
        # (relative to temperature of sun)
        lum = self.luminosity(m)
        temp = self.temperature(m)
        if np.any(temp > 7200 or temp < 2600):
            raise ValueError(
                "HZ calculation only can only be done in temperature "
                "range = [2600,7200]K (mass range = [0.08, 1.68] M_sun)."
            )

        temp_eff = temp - temperature_sun
        # calculate effective stellar flux
        temp_powers = np.array(
            [
                np.ones_like(temp_eff),
                temp_eff,
                temp_eff**2,
                temp_eff**3,
                temp_eff**4,
            ]
        ).T
        S_eff = temp_powers.dot(parameter)

        # calculate distance in AU
        dist = np.sqrt(lum / S_eff)
        return dist
