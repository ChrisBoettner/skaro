#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:10:14 2023

@author: chris
"""

from typing import Optional

import numpy as np
from astropy.cosmology import Planck15
from numpy.typing import NDArray
from yt.fields.derived_field import DerivedField
from yt.fields.field_detector import FieldDetector
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.planets import PlanetModel, PlanetOccurenceModel
from gallifrey.stars import ChabrierIMF, StellarModel
from gallifrey.utilities.logging import logger

# create Logger
logger = logger(__name__)


class Fields:
    """Filter class to effective add new filters to yt data source."""

    def __init__(self, ds: ArepoHDF5Dataset | YTDataContainerDataset):
        """
        Initialize.

        Parameters
        ----------
        ds : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        """
        self.ds = ds

        self.star_properties_flag = (
            False  # flag if star_properties method was executed.
        )

    def convert_star_properties(self) -> None:
        """
        Replaces stellar_age field (which contains the formation scale parameter) to
        stellar age = (current time - formation time) in Gyr.

        """
        if "stars" not in dir(self.ds.fields):
            raise AttributeError(
                "'Stars' field does not exist. Needs to be created "
                "to calculate stellar ages using filters.add_stars()."
            )

        logger.info(
            "FIELDS: Overriding ('stars', 'stellar_age') field with " "ages in Gyr."
        )

        self.ds.add_field(
            ("stars", "stellar_age"),
            function=self._stellar_age,
            sampling_type="local",
            units="Gyr",
            force_override=True,
        )

        logger.info(
            "FIELDS: Adding field ('stars', 'InitialMass'), which is identical to "
            "('stars', 'GFM_InitialMass') but with units changed from 'dimensionless'"
            " to 'code_mass'."
        )

        def _get_stellar_mass(
            field: DerivedField,
            data: FieldDetector,
        ) -> NDArray:
            return self.ds.arr(data["stars", "GFM_InitialMass"].value, "code_mass")

        self.ds.add_field(
            ("stars", "InitialMass"),
            function=_get_stellar_mass,
            sampling_type="local",
            units="code_mass",
            force_override=True,
        )

        self.star_properties_flag = True

    def add_planets(
        self,
        stellar_model: StellarModel,
        planet_model: PlanetModel,
        imf: ChabrierIMF,
        lower_bound: float = 0.08,
    ) -> None:
        """
        Add (number of) planet and field to star particles. Also adds planet_effect
        field that encodes the dominant effect on the planet number.

        Parameters
        ----------
        stellar_model : StellarModel
            Stellar model that connects mass to other stellar parameter.
        planet_model : PlanetModel
            Planet model that contains relevant planet parameter.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.
        lower_bound : float, optional
            Lower bound for the integration of the Chabrier IMF. The default is 0.08.

        """
        self.check_star_properties()

        planets_occ_model = PlanetOccurenceModel(stellar_model, planet_model, imf)

        def _planets(field: DerivedField, data: FieldDetector) -> NDArray:
            return planets_occ_model.number_of_planets(data, lower_bound=lower_bound)

        self.ds.add_field(
            ("stars", "planets"),
            function=_planets,
            sampling_type="local",
            units="auto",
            dimensions=1,
        )

        def _planets_effect(field: DerivedField, data: FieldDetector) -> NDArray:
            return planets_occ_model.dominant_effect(data)

        self.ds.add_field(
            ("stars", "planet_effects"),
            function=_planets_effect,
            sampling_type="local",
            units="auto",
            dimensions=1,
        )

    def add_main_sequence_stars(
        self,
        stellar_model: StellarModel,
        imf: ChabrierIMF,
        lower_bound: float = 0.08,
        temperature_limit: Optional[int] = None,
    ) -> None:
        """
        Add (number of) main sequence star field to star particles.

        Parameters
        ----------
        stellar_model : StellarModel
            Stellar model that connects mass to other stellar parameter.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.
        lower_bound : float, optional
            Lower bound for the integration of the Chabrier IMF. The default is 0.08.
        temperature_limit: Optional[int], optional
            Upper integration bound from maximum planet temperature (in K). The default
            is None, ignoring this effect.
        """

        self.check_star_properties()

        def _star_number(field: DerivedField, data: FieldDetector) -> NDArray:
            masses = data["stars", "InitialMass"].to("Msun").value
            upper_bound = stellar_model.mass_from_lifetime(data["stars", "stellar_age"])
            if temperature_limit is not None:
                m_temperature = stellar_model.mass_from_temperature(temperature_limit)
                upper_bound[upper_bound > m_temperature] = m_temperature
            return imf.number_of_stars(masses, lower_bound, upper_bound)

        self.ds.add_field(
            ("stars", "main_sequence_stars"),
            function=_star_number,
            sampling_type="local",
            units="auto",
            dimensions=1,
        )

    def add_iron_abundance(self, log_solar_fe_fraction: float = -2.7) -> None:
        """
        Add iron abundanace [Fe/H].

        Parameters
        ----------
        log_solar_fe_fraction : float
            Solar iron fraction,  m_Fe/m_H.

        """
        self.check_star_properties()

        def _iron_abundance(field: DerivedField, data: FieldDetector) -> NDArray:
            # calculate iron abundance for star particles
            fe_fraction = (
                data["stars", "Fe_fraction"].value / data["stars", "H_fraction"].value
            )
            fe_fraction[fe_fraction < 0] = 0  # some values are < 0
            log_fe_fraction = np.where(
                fe_fraction > 0, np.ma.log10(fe_fraction), -3
            )  # set those values to -100

            # normalise to stellar fraction
            fe_abundance = log_fe_fraction - log_solar_fe_fraction
            return fe_abundance

        self.ds.add_field(
            ("stars", "[Fe/H]"),
            function=_iron_abundance,
            sampling_type="local",
            units="auto",
            dimensions=1,
        )

    @staticmethod
    def _stellar_age(
        field: DerivedField,
        data: FieldDetector,
        interpolation_num: int = 500,
    ) -> None:
        """
        Calculate stellar ages from formation scale factor.

        Parameters
        ----------
        field : DerivedField
            Field parameter used for adding field to yt Dataset.
        data : FieldDetector
            Data parameter used for adding field to yt Dataset.
        interpolation_num : int, optional
            Number of data points for redshift-formation time interpolation. The
            default is 500.

        """
        # get current simulation time, and formation redshifts of star particles from
        # scale factor
        current_time = data.ds.current_time.to("Gyr")
        formation_redshift = (
            1 / np.array(data["stars", "GFM_StellarFormationTime"])
        ) - 1

        if len(formation_redshift) == 0:
            return data.ds.arr(np.array([]), "Gyr")

        # make redshift space and calculate corresponding cosmic time
        max_redshift = np.amax(formation_redshift)
        redshift_grid = np.linspace(
            data.ds.current_redshift, max_redshift, interpolation_num
        )
        time_grid = Planck15.age(redshift_grid).value

        # calculate formation times from redshift by interpolating redshift grid
        current_time = data.ds.quan(Planck15.age(data.ds.current_redshift).value, "Gyr")
        formation_time = data.ds.arr(
            np.interp(formation_redshift, redshift_grid, time_grid), "Gyr"
        )

        return current_time - formation_time

    def check_star_properties(self) -> None:
        """
        Sanity check if correct functions have been run before adding new fields.

        """
        stars_filter_exits = "stars" in dir(self.ds.fields)

        if stars_filter_exits and self.star_properties_flag:
            pass
        else:
            raise AttributeError(
                "'stars' field has not properly been set. Run "
                "'add_stars' method from Filter() and "
                "'convert_star_properties' method from Fields() first."
            )
