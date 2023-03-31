#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:34:58 2023

@author: chris
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray

class PlanetOccurenceModel:
    def __init__(self, stellar_model, planet_model, imf):
        self.planet_model = planet_model
        self.stellar_model = stellar_model
        self.imf = imf

    def number_of_planets(self, data, lower_bound=0.08):
        mass_limit   = np.amin(self.mass_limits(data), axis=1) 
        stellar_ages = data["stars", "stellar_age"].value # in Gyr
        masses = data["stars", "Masses"].to("Msun").value
        
        # number of eligable stars
        star_number = self.imf.number_of_stars(masses, 
                                               upper_bound=mass_limit,
                                               lower_bound=lower_bound)
        
        # calculate number of planets by multiplying number of eligable stars with
        # planet occurence rate
        # set number of planets to 0 if stellar age is below planet formation time
        planet_number = np.where(stellar_ages>=self.planet_model.planet_formation_time,
                                 self.planet_model.occurence_rate * star_number,
                                 0)
        
        return planet_number
    
    def dominant_effect(self, data):
        # array of dominant effects
        # 0 means lifetime, 1 means metallicity, 2 means temperature, 3 means planet
        # formation time
        
        dominant_eff = np.argmin(self.mass_limits(data), axis=1)
        stellar_ages = data["stars", "stellar_age"].value # in Gyr
        
        dominant_eff[stellar_ages<self.planet_model.planet_formation_time] = 3
        return dominant_eff

    def mass_limits(self, data):
        limit_models = [self.mass_limit_from_lifetime, self.mass_limit_from_metallicity,
                  self.mass_limit_from_temperature]
        mass_limits = np.array([f(data) for f in limit_models]).T
        return mass_limits

    def mass_limit_from_lifetime(self, data):
        stellar_ages = data["stars", "stellar_age"].value # in Gyr
        m_from_lifetime = self.stellar_model.mass_from_lifetime(stellar_ages)
        return m_from_lifetime
        
    def mass_limit_from_metallicity(self, data):
        # calculate iron abundance for star particles
        fe_fraction = (data['stars','Fe_fraction'].value
                       /data['stars','H_fraction'].value)
        fe_fraction[fe_fraction<0] = 0 # due to numerical effects some values are < 0
        log_fe_fraction = np.where(fe_fraction>0, 
                             np.ma.log10(fe_fraction),
                             -np.inf)
        
        fe_abundance = log_fe_fraction - self.stellar_model.log_solar_fe_fraction
        
        # calculate maximum rocky planet formation distance
        crit_distance = self.planet_model.critical_formation_distance(fe_abundance)
        
        # match maximum formation distance to inner habitable zone distance
        m_from_metallicity = self.stellar_model.inner_HZ_inverse(crit_distance)
        return m_from_metallicity
    
    def mass_limit_from_temperature(self, data):
        cutoff_mass =  self.stellar_model.mass_from_temperature(
                            self.planet_model.cutoff_temperature)
        m_from_temp = np.full_like(data["stars", "stellar_age"].value, cutoff_mass)
        return m_from_temp

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
    def critical_formation_distance(iron_abundance: ArrayLike) -> NDArray:
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

        return np.power(10, 1.5 + np.asarray(iron_abundance))
