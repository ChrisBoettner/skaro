#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:19:27 2023

@author: chris
"""
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import rv_continuous


class Chabrier(rv_continuous):
    """
    A probability distribution function for the Chabrier IMF.

    Note: The distribution is normalised so that the integral over the pdf is equal
    to 1. This disagrees with the common normalisation which is that the integral over
    m*pdf(x) is equal to one. In order to calculate the number of stars for a given,
    one must compensate for that (which is done e.g. in the number_of_stars method).
    """

    def __init__(
        self,
        slope: float = 1.35,
        mean: float = 0.2,
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
        mean : float, optional
            Mean of lognormal section, defined so that the exponential reads
            ((log(m)-log(mean))**2). The default is 0.2.
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
                mean != 0.2,
                variance != 0.6,
                lower_mass_limit != 0.08,
                upper_mass_limit != 100,
                transition_point != 1,
            ]
        ):
            warnings.warn(
                "If any of the default parameter of the Chabrier IMF are "
                "changed, the normalisation should be updated using "
                "update_normalisation method."
            )
        self.slope = slope
        self.mean = mean
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
            (np.log10(self.mean) ** 2) / self.variance
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

    def _pdf(self, m: ArrayLike) -> NDArray:
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
        m = np.asarray(m)
        pdf = np.empty_like(m)

        # masks for masses above and below piecewise transition
        mask, inverse_mask = self.create_mask(m)

        # pdf for masses > transition point
        pdf[mask] = self.mass_normalisation / self.ln10 * m[mask] ** (-(self.slope + 1))

        # pdf for masses < transition point
        pdf[inverse_mask] = (
            (self.constant_1 / self.ln10)
            * (1 / m[inverse_mask])
            * np.exp(-(np.log10(m[inverse_mask] / self.mean) ** 2) / self.variance)
        )
        return pdf

    def _cdf(self, m: ArrayLike) -> NDArray:
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
        m = np.asarray(m)
        cdf = np.empty_like(m)

        # masks for masses above and below piecewise transition
        mask, inverse_mask = self.create_mask(m)

        # cdf for masses > transition point
        cdf[mask] = (self.constant_lognormal_1 - self.constant_lognormal_008) + (
            self._antiderivative_powerlaw(m[mask]) - self.constant_powerlaw_1
        )

        # cdf for masses < transition point
        cdf[inverse_mask] = (
            self._antiderivative_lognormal(m[inverse_mask])
            - self.constant_lognormal_008
        )

        return cdf

    def number_of_stars(
        self,
        M_star: float | NDArray,
        lower_bound: float | NDArray = 0.08,
        upper_bound: float | NDArray = 100,
    ) -> ArrayLike:
        """
        Calculate the number of stars expected from Chabrier IMF for a total amount
        of stellar mass created within the given bounds.

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
        ArrayLike
            Number of stars born.

        """
        return (M_star * self.number_normalisation) * (
            self.cdf(upper_bound) - self.cdf(lower_bound)
        )

    def create_mask(self, m: NDArray) -> tuple[NDArray, NDArray]:
        """
        Create mask for values of m above and below transition point.

        Parameters
        ----------
        m : NDArray
            Mass of stars.

        Returns
        -------
        mask: NDArray
            Mask for masses above transition point.
        inverse_mask: NDArray
            Mask for masses below transition point.

        """

        mask = m > self.transition_point
        inverse_mask = ~mask
        return mask, inverse_mask

    def update_normalisation(self) -> None:
        """
        Calculate new normalisations (mass and number) in case non-default parameter
        are used and pre-saved values are not valid.
        """
        new_normalisation = quad(self.pdf, self.a, self.b)[0]  # integrate over support
        self.number_normalisation *= self.mass_normalisation / new_normalisation
        self.mass_normalisation = new_normalisation

    def _antiderivative_lognormal(self, m: float | NDArray) -> NDArray:
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
        return self.constant_2 * erf(np.log(np.asarray(m) / self.mean) / self.sigma_2)

    def _antiderivative_powerlaw(self, m: float | NDArray) -> NDArray:
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
