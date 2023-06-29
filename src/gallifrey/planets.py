#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:34:58 2023

@author: chris
"""
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.data.paths import Path
from gallifrey.stars import ChabrierIMF, StellarModel


class Population:
    """
    NGPPS planet population.

    """

    def __init__(self, population_id: str | int, age: int) -> None:
        """
        Initialize object, load dataframe and add planet categories.

        Parameters
        ----------
        population_id : str | int
            Name of the population, can be string or integer. If string, load population
            with that name. If int, load solar-like run with number of embryos
            correponding to that number.
        age : int
            Age of system at time of snapshot.

        """
        # dict mapping number of embryos to correponding population names for
        # solar-like run
        self.embryo_dict = {10: "ng96", 20: "ng74", 50: "ng75", 100: "ng76"}

        # if population is int, map number of embryos to corresponding solar-like
        # run
        if isinstance(population_id, int):
            if population_id not in self.embryo_dict.keys():
                raise ValueError("No population corresponding to number of embryos.")
            self.population_id = self.embryo_dict[population_id]
        elif isinstance(population_id, str):
            self.population_id = population_id
        else:
            raise ValueError("population_id must be either int or str.")

        # load populations
        self.population = pd.read_csv(
            Path().raw_data(f"NGPPS/{population_id}/snapshot_{age}.csv")
        )
        # add planet categories
        self.add_categories()

        # load system properties
        self.systems = self.load_system_properties()

    def add_categories(self) -> None:
        """
        Adds planet categories to columns.

        """

        categories = {
            "Dwarf": lambda row: row["total_mass"] < 0.5,
            "Earth": lambda row: 0.5 <= row["total_mass"] < 2,
            "SuperEarth": lambda row: 2 <= row["total_mass"] < 10,
            "Neptunian": lambda row: 10 <= row["total_mass"] < 30,
            "SubGiant": lambda row: 30 <= row["total_mass"] < 300,
            "Giant": lambda row: 300 <= row["total_mass"],
            "DBurning": lambda row: 4322 <= row["total_mass"],
        }

        # Apply each function to the DataFrame to create new columns
        for category, condition in categories.items():
            self.population[category] = self.population.apply(condition, axis=1)

    def load_system_properties(self) -> pd.DataFrame:
        """
        Loads system monte carlo variables (needed e.g. to calculate metallicity).

        Raises
        ------
        NotImplementedError
            Raised if population_id does not match one of the solar-like runs, since
            currently only those have the properties available.

        Returns
        -------
        properties : DataFrame
            Dataframe containing the system properties.

        """
        if self.population_id in self.embryo_dict.values():
            # column names
            columns = [
                "system_id",
                "mstar",
                "sigma",
                "expo",
                "ain",
                "aout",
                "fpg",
                "mwind",
            ]

            # read data file property data file
            properties = pd.read_csv(
                Path().external_data("NGPPS_variables.txt"),
                delimiter=r"\s+",
                names=columns,
            )

            # modify the 'system_id' column to remove 'system_id' prefix
            properties["system_id"] = properties["system_id"].str[3:].astype(int)

            # convert columns to float
            for col in columns[1:]:
                properties[col] = properties[col].map(lambda x: float(x.split("=")[1]))

        else:
            raise NotImplementedError(
                "Population ID does not much any solar-like run. If you "
                "use other runs with other stellar masses, the system "
                "properties won't match."
            )

        return properties


class PlanetModel:
    """
    Planet Model.

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


class PlanetOccurenceModel:
    """
    Model to assign planets to star particles.
    """

    def __init__(
        self,
        stellar_model: StellarModel,
        planet_model: PlanetModel,
        imf: ChabrierIMF,
    ) -> None:
        """
        Initialize.

        Parameters
        ----------
        stellar_model : StellarModel
            Stellar model that connects mass to other stellar parameter.
        planet_model : PlanetModel
            Planet model that contains relevant planet parameter.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.

        """

        self.planet_model = planet_model
        self.stellar_model = stellar_model
        self.imf = imf

    def number_of_planets(
        self,
        data: ArepoHDF5Dataset | YTDataContainerDataset,
        lower_bound: float = 0.08,
        mass_limits: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculate the number of planets associated with the star particles based on
        the mass of the star particle and including the different cut off effects from
        stellar lifetime, metallicity, temperature and planet formation time.

        Parameters
        ----------
        data : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation..
        lower_bound : float, optional
            Lower bound for the integration of the Chabrier IMF. The default is 0.08.
        mass_limits : NDArray, optional
            Mass limits used to choose integration limit from. If not provided,
            calculated using mass_limits method. Primarely implemented as argument to
            avoid having to calculate values multiple times when calling
            number_of_planets and dominant_effect methods. The default is None.

        Returns
        -------
        NDArray
            Number of planets associated with star particles.

        """
        stellar_ages = data["stars", "stellar_age"].value  # in Gyr
        masses = data["stars", "InitialMass"].to("Msun").value

        # calculate mass limits based on different effects, if mass limits are not
        # provided
        if not mass_limits:
            mass_limits = self.mass_limits(data)
        mass_limit = np.amin(mass_limits, axis=1)

        # based on mass limit, calculate number of eligable stars
        star_number = self.imf.number_of_stars(
            masses, upper_bound=mass_limit, lower_bound=lower_bound
        )

        # calculate number of planets by multiplying number of eligable stars with
        # planet occurence rate, set number of planets to 0 if stellar age is below
        # planet formation time
        planet_number = np.where(
            stellar_ages >= self.planet_model.planet_formation_time,
            self.planet_model.occurence_rate * star_number,
            0,
        )
        return planet_number

    def dominant_effect(
        self,
        data: ArepoHDF5Dataset | YTDataContainerDataset,
        mass_limits: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculate the dominant effect on the number of planets based on the different
        mass limits and planet formation time by calculating all effects and then
        choosing the relevant one.
        Returns array with values between 0 and 3, where the number indicates the
        dominant effect:
            0: lifetime
            1: metallicity
            2: temperature cut
            3: planet formation time

        Parameters
        ----------
        data : : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        mass_limits : NDArray, optional
            Mass limits used to choose integration limit from. If not provided,
            calculated using mass_limits method. Primarely implemented as argument to
            avoid having to calculate values multiple times when calling
            number_of_planets and dominant_effect methods. The default is None.

        Returns
        -------
        dominant_eff : NDArray
            Array containing the dominant effect on the planet number.

        """
        if not mass_limits:
            mass_limits = self.mass_limits(data)
        dominant_eff = np.argmin(mass_limits, axis=1)

        # add planet formation time effect
        stellar_ages = data["stars", "stellar_age"].value  # in Gyr
        dominant_eff[stellar_ages < self.planet_model.planet_formation_time] = 3
        return dominant_eff

    def mass_limits(
        self,
        data: ArepoHDF5Dataset | YTDataContainerDataset,
    ) -> NDArray:
        """
        Calculate maximum considered stellar mass limits based the different modelled
        effects.

        Parameters
        ----------
        data : : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.

        Returns
        -------
        mass_limits : NDArray
            Array of mass limits (size: [number of effects, number of star particles]).

        """
        limit_models = [
            self.mass_limit_from_lifetime,
            self.mass_limit_from_metallicity,
            self.mass_limit_from_temperature,
        ]
        mass_limits = np.array([func(data) for func in limit_models]).T
        return mass_limits

    def mass_limit_from_lifetime(
        self,
        data: ArepoHDF5Dataset | YTDataContainerDataset,
    ) -> NDArray:
        """
        Calculate maximum considered stellar mass based on the lifetime of the star
        particles.

        Parameters
        ----------
        data : : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.

        Returns
        -------
        m_from_lifetime : NDArray
            Array of mass limits.

        """
        stellar_ages = data["stars", "stellar_age"].value  # in Gyr
        m_from_lifetime = self.stellar_model.mass_from_lifetime(stellar_ages)
        return m_from_lifetime

    def mass_limit_from_metallicity(
        self,
        data: ArepoHDF5Dataset | YTDataContainerDataset,
    ) -> NDArray:
        """
        Calculate maximum considered stellar mass based on the metallicity of the
        stellar particle, by comparing maximum distance at which planets can
        form (Johnson2012) and inner edge of planetary HZ (Kopparapu2014).

        Parameters
        ----------
        data : : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.

        Returns
        -------
        m_from_metallicity : NDArray
            Array of mass limits.

        """
        fe_abundance = data["stars", "[Fe/H]"]

        # calculate maximum rocky planet formation distance
        crit_distance = self.planet_model.critical_formation_distance(fe_abundance)

        # match maximum formation distance to inner habitable zone distance
        m_from_metallicity = self.stellar_model.inner_HZ_inverse(crit_distance)
        return m_from_metallicity

    def mass_limit_from_temperature(
        self,
        data: ArepoHDF5Dataset | YTDataContainerDataset,
    ) -> NDArray:
        """
        Calculate maximum considered stellar mass based on the maximum
        stellar temperature.

        Parameters
        ----------
        data : : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.

        Returns
        -------
        m_from_temp : NDArray
            Array of mass limits.

        """
        cutoff_mass = self.stellar_model.mass_from_temperature(
            self.planet_model.cutoff_temperature
        )
        m_from_temp = np.full_like(data["stars", "stellar_age"].value, cutoff_mass)
        return m_from_temp
