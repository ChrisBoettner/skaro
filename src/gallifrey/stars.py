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
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import rv_continuous

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
            Path().external_data("stellar_main_sequence_parameter.csv")
        )

        self.log_stellar_parameter = self.stellar_parameter.copy()
        self.log_stellar_parameter.iloc[:, 1:] = np.log10(
            self.stellar_parameter.iloc[:, 1:]
        )

        # HZ parameter
        self.hz_parameter = np.loadtxt(Path().external_data(r"HZ_coefficients.dat"))
        self.hz_data = np.loadtxt(Path().interim_data(r"HZ_distances.txt"))

    def lifetime(self, m: float | NDArray) -> float | NDArray:
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

    def luminosity(self, m: float | NDArray) -> float | NDArray:
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

    def temperature(self, m: float | NDArray) -> float | NDArray:
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

    def mass_from_lifetime(self, lifetime: float | NDArray) -> float | NDArray:
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

    def mass_from_luminosity(self, lifetime: float | NDArray) -> float | NDArray:
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

    def mass_from_temperature(self, lifetime: float | NDArray) -> float | NDArray:
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

    def inner_HZ(self, m: float | NDArray) -> NDArray:
        """
        Calculate (conservative estimate) of inner limit of HZ, based on Kopparapu2014
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
        left: float = 0.08,
        right: float = 100,
        **kwargs: Any,
    ) -> NDArray:
        """
        Inverse relation for the inner HZ calculation, returns mass of star for a
        given inner HZ radius.

        NOTE: If precalculated is True and distance is outside of interpolation range,
        it uses left and right argument to fill the values.

        Parameters
        ----------
        dist : ArrayLike
            Distance to inner HZ edge.
        precalculated : bool, optional
            Decide if pre-calculated table should be interpolated or not. Currently only
            interpolation is implemented. The default is True.
        left: float, optional
            Mass returned if distance is below interpolation rate. The default is 0.08.
        right: float, optional
            Mass returned if distance is above interpolation rate. The default is 100.
        kwargs: dict[Any, Any], optional
            Additional arguments passed to np.interp .

        Returns
        -------
        NDArray
            Infered masses of stars for given inner HZ distances.

        """
        if precalculated:
            return np.interp(
                dist,
                self.hz_data[:, 1],
                self.hz_data[:, 0],
                left=left,
                right=right,
                **kwargs,
            )

        else:
            raise NotImplementedError(
                "Direct inner HZ inverse calculation not yet implemented."
            )

    def outer_HZ(self, m: float | NDArray) -> NDArray:
        """
        Calculate (conservative estimate) of outer limit of HZ, based on Kopparapu2014
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
        value: float | NDArray,
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
        log_value = np.where(value > 0, np.ma.log10(value), -np.inf)

        if input_quantity == "lifetime" and (not reverse):
            warnings.warn(
                "If input quantity is lifetime, reverse should be true in "
                "order for interpolation to work.",
                stacklevel=2,
            )

        input_data = self.log_stellar_parameter[input_quantity]
        target_data = self.log_stellar_parameter[target_quantity]
        if reverse:
            input_data = input_data[::-1]
            target_data = target_data[::-1]

        log_quantity = np.interp(log_value, input_data, target_data)
        return np.power(10, log_quantity)

    def calculate_habitable_zone(
        self, m: float | NDArray, parameter: NDArray, solar_temperature: float = 5780
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
        solar_temperature: float
            Solar effective temperature in Kelvin.

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

        temp_eff = temp - solar_temperature
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


class ChabrierIMF(rv_continuous):
    """
    A probability distribution function for the Chabrier IMF.

    NOTE: The distribution is normalised so that the integral over the pdf is equal
    to 1. This disagrees with the common normalisation which is that the integral over
    m*pdf(x) is equal to one. In order to calculate the number of stars for a given,
    one must compensate for that (which is done e.g. in the number_of_stars method).
    """

    def __init__(
        self,
        slope: float = 1.35,
        m_c: float = 0.2,
        variance: float = 0.6,
        lower_mass_limit: float = 0.08,
        upper_mass_limit: float = 100,
        transition_point: float = 1,
        mass_normalisation: float = 0.4196249611479187,
    ) -> None:
        """
        Initialize.

        Parameters
        ----------
        slope : float, optional
            Slope of power law section. The default is 1.35.
        m_c : float, optional
            Mean of lognormal section, defined so that the exponential reads
            ((log(m)-log(m_c))**2). The default is 0.2.
        variance : float, optional
            Variance of lognormal section. The default is 0.6.
        lower_mass_limit : float, optional
            Lower end of support of distribution. The default is 0.08.
        upper_mass_limit : float, optional
            Upper end of support of distribution. The default is 100.
        transition_point : float, optional
            Transition point between lognormal and power law section. The default is 1.
        mass_normalisation : float, optional
            Overall normalisation of the pdf so that integral over support is equal to
            1. The default is 0.4196249611479187, which is the value pre-calculated for
            all default parameter.
        """
        # set support range
        super().__init__(a=lower_mass_limit, b=upper_mass_limit)

        # set parameter
        if np.any(
            [
                slope != 1.35,
                m_c != 0.2,
                variance != 0.6,
                lower_mass_limit != 0.08,
                upper_mass_limit != 100,
                transition_point != 1,
            ]
        ):
            warnings.warn(
                "If any of the default parameter of the Chabrier IMF are "
                "changed, the normalisation should be updated using "
                "update_normalisation method.",
                stacklevel=2,
            )

        self.slope = slope
        self.m_c = m_c
        self.variance = variance
        self.transition_point = transition_point
        self.mass_normalisation = mass_normalisation

        # This is the alternative normalisation that ensures that the integeral over
        # m*chabrier(m) is unity. This normalisation is needed to calculate the number
        # of stars created for a given amount of total stellar mass (e.g. in
        # number_of_stars method).
        self.number_normalisation = 0.6039226412905723 / self.mass_normalisation

        # recurring constants (pre-calculated for speed)
        # =============================================================================
        self.ln10 = np.log(10)
        # constant for lognormal (to connect continously to powerlaw)
        self.constant_1 = self.mass_normalisation * np.exp(
            (np.log10(self.m_c) ** 2) / self.variance
        )
        # costant for lognormal anti-derivative
        self.sigma_2 = self.ln10 * np.sqrt(self.variance)  # converts between log and ln
        self.constant_2 = (
            0.5 * np.sqrt(np.pi) * self.constant_1 / self.ln10 * self.sigma_2
        )
        # edges of piecewise definition
        self.constant_lognormal_008 = self._antiderivative_lognormal(0.08)
        self.constant_lognormal_1 = self._antiderivative_lognormal(1)
        self.constant_powerlaw_1 = self._antiderivative_powerlaw(1)
        # =============================================================================

    def _pdf(self, m: float | NDArray) -> NDArray:
        """
        Calculate value of Chabrier probability density function (normalised to 1).

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        NDArray
            Value of Chabrier pdf at m.

        """
        pdf = np.where(
            m > self.transition_point,
            # power law for m > transition point
            self.mass_normalisation / self.ln10 * m ** (-(self.slope + 1)),
            # lognormal for m < transition point
            (
                (self.constant_1 / self.ln10)
                * (1 / m)
                * np.exp(-(np.log10(m / self.m_c) ** 2) / self.variance)
            ),
        )
        return pdf

    def _cdf(self, m: float | NDArray) -> NDArray:
        """
        Calculate value of Chabrier cumulative distribution function (normalised to 1).

        Parameters
        ----------
        m : ArrayLike
            Mass of stars.

        Returns
        -------
        NDArray
            Value of Chabrier cdf at m.

        """
        cdf = np.where(
            m > self.transition_point,
            # lognormal and power law for m > transition point
            (
                (self.constant_lognormal_1 - self.constant_lognormal_008)
                + (self._antiderivative_powerlaw(m) - self.constant_powerlaw_1)
            ),
            # lognormal for m < transition point
            (self._antiderivative_lognormal(m) - self.constant_lognormal_008),
        )
        return cdf

    def number_density(
        self,
        M_star: float | NDArray,
        m: float | NDArray,
    ) -> float | NDArray:
        """
        Value of pdf at m, rescaled by total amount of stellar mass (and number
        normalisation).

        Parameters
        ----------
        M_star : float|NDArray
            Total stellar mass created.
        m : float|NDArray
            Points where to evaluate the pdf.

        Returns
        -------
        float | NDArray
            Number density of stars.

        """
        return self.number_normalisation * np.outer(M_star, self.pdf(m))

    def number_of_stars(
        self,
        M_star: float | NDArray,
        lower_bound: float | NDArray = 0.08,
        upper_bound: float | NDArray = 100,
    ) -> float | NDArray:
        """
        Calculate the number of stars expected from Chabrier IMF for a total amount
        of stellar mass created within the given bounds. Returns 0 if
        lower_bound >= upper_bound.

        Parameters
        ----------
        M_star : float|NDArray
            Total stellar mass created.
        lower_bound : float|NDArray, optional
            Lower bound of considered interval of IMF. The default is 0.08.
        upper_bound : float|NDArray, optional
            Upper bound of considered interval of IMF. The default is 100.

        Returns
        -------
        float | NDArray
            Number of stars born.

        """
        num_stars = np.where(
            lower_bound < upper_bound,
            # calculate number of stars using CDF and
            # correct normalisation
            (
                (M_star * self.number_normalisation)
                * (self.cdf(upper_bound) - self.cdf(lower_bound))
            ),
            # return 0 if lower_bound >= upper bound
            0,
        )

        return num_stars

    def update_normalisation(self) -> None:
        """
        Calculate new normalisations (mass and number) in case non-default parameter
        are used and pre-saved values are not valid.
        """
        new_normalisation = quad(self.pdf, self.a, self.b)[0]  # integrate over support
        self.number_normalisation *= self.mass_normalisation / new_normalisation
        self.mass_normalisation = new_normalisation

    def _antiderivative_lognormal(self, m: float | NDArray) -> float | NDArray:
        """
        Value of anti-derivative of lognormal distribution.

        Parameters
        ----------
        m : float|NDArray
            Mass of stars.

        Returns
        -------
        NDArray
            Value of anti-derivative.

        """
        return self.constant_2 * erf(np.log(np.asarray(m) / self.m_c) / self.sigma_2)

    def _antiderivative_powerlaw(self, m: float | NDArray) -> float | NDArray:
        """
        Value of anti-derivative of powerlaw distribution.

        Parameters
        ----------
        m : float|NDArray
            Mass of stars.

        Returns
        -------
        NDArray
            Value of anti-derivative.

        """
        return (
            (self.mass_normalisation / self.ln10)
            * (-1 / self.slope)
            * np.power(m, (-self.slope))
        )
