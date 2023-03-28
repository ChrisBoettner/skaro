#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:34:58 2023

@author: chris
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray


class PlanetModel:
    """
    Stellar Model.
    """

    def __init__(
        self,
        planet_formation_time: float = 0.1,
        cutoff_temperature: float = 7200,
        occurence_rate: float = 0.5,
    ) -> None:
        """
        Initialize.

        Parameters
        -------
        planet_formation_time : float, optional
            Estimated time scale for rocky (habitable) planet formation in Gyr. The
            default is 0.1
        cutoff_temperature : float, optional
            Maximum stellar effective temperature for which planets are considered in K.
            We estimate the occurence rate for more massive stars at 0, since little
            data is available on occurence rates and habitable zones. The default
            is 7200.
        occurence_rate : float, optional
            Occurence rate of planets in habitable zone below temperature cutoff. The
            default value is 0.5 (for M and FGK spectral types), from Bryson2020
            and Hsu2020.

        """
        self.planet_formation_time = planet_formation_time  # in Gyr
        self.cutoff_temperature = cutoff_temperature  # in K
        self.occurence_rate = occurence_rate

    @staticmethod
    def critical_formation_distance(fe_fraction: ArrayLike) -> NDArray:
        """
        Critical distance for planet formation based on [Fe/H] estimated by Johnson2012.

        Parameters
        ----------
        fe_fraction : Arraylike
            Iron abundace [Fe/H] as estimator of metallicity.

        Returns
        -------
        NDArray
            Estimated maximum distance for planet formation.

        """

        return np.power(10, 1.5 + np.asarray(fe_fraction))
