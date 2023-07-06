#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:10:14 2023

@author: chris
"""
from typing import Optional

import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from yt.fields.derived_field import DerivedField
from yt.fields.field_detector import FieldDetector
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.planets import PlanetModel
from gallifrey.stars import ChabrierIMF
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

    def convert_PartType4_properties(self) -> None:
        """
        Replaces stellar_age field (which contains the formation scale parameter) to
        stellar age = (current time - formation time) in Gyr, and adds InitialMass
        field which has the correct units.

        """
        logger.info(
            "FIELDS: Adding field ('PartType4', 'stellar_age') field with "
            "ages in Gyr."
        )

        self.ds.add_field(
            ("PartType4", "stellar_age"),
            function=self._stellar_age,
            sampling_type="local",
            units="Gyr",
            force_override=True,
        )

        logger.info(
            "FIELDS: Adding field ('PartType4', 'InitialMass'), with masses in "
            " 'code_mass'."
        )

        def _get_stellar_mass(
            field: DerivedField,
            data: FieldDetector,
        ) -> NDArray:
            return self.ds.arr(data["PartType4", "GFM_InitialMass"].value, "code_mass")

        self.ds.add_field(
            ("PartType4", "InitialMass"),
            function=_get_stellar_mass,
            sampling_type="local",
            units="code_mass",
            force_override=True,
        )

        self.star_properties_flag = True

    def add_planets(
        self,
        category: str,
        host_star_masses: float | tuple[float, ...],
        planet_model: PlanetModel,
        imf: ChabrierIMF,
        imf_bounds: tuple[float, float],
        reference_age: Optional[int] = 100000000,
        num_integral_points: int = 50,
    ) -> None:
        """
        Add number of planets of a given category associated with the star particle.
        This is done by calculating the number of planets per star using the
        NGPPS population model and then multiplying by the number of stars in the
        considered range in the case that host_star_masses is scalar. If
        host_star_masses is a tuple, calculate values for different masses and the
        corresponding IMF values and integrate numerically. (Values between masses
        in list are interpolated.)

        Parameters
        ----------
        category : str
            The category of planets to consider, e.g. "Earth", "Giant", etc. Find
            list of available categories in planet_model.population class.
        host_star_masses : float | tuple[float, ...]
            Masses of host stars used for calculation. Can be a scalar of tuple of
            values, values must be in [0.1, 0.3, 0.5, 1].
        planet_model : PlanetModel
            The planet model that associates a stellar particle properties
            with number of planets (of a given class).
        imf : ChabrierIMF
            Stellar initial mass function of the star particles..
        imf_bounds : tuple[float, float]
            The range over with to integrate the imf. Corresponds to the mass range
            of stars considered.
        reference_age : Optional[int], optional
            The age at which to evaluate the planet population model. The default is
            int(1e+8), i.e. 100Myr. If the value is None, the age of the star particle
            is used. (This is much slower and memory intensive.)
        num_integral_points : int, optional
            Number of points to evaluate the numerical integral on, in case
            host_star_masses is a list. The default is 50.

        """
        # check if star properties are correctly set
        self.check_star_properties()

        def _planets(field: DerivedField, data: FieldDetector) -> NDArray:
            stellar_ages = data["stars", "stellar_age"].value
            metallicities = data["stars", "[Fe/H]"]

            particle_masses = data["stars", "InitialMass"].to("Msun").value

            # choose what age to associate with star particles
            if reference_age is None:
                ages = stellar_ages * 1e9
            elif isinstance(reference_age, int):
                ages = np.repeat(reference_age, len(stellar_ages))
            else:
                raise ValueError("reference_age must be int or None.")

            # create dataframe from relevant quantities
            variables_dataframe = pd.DataFrame(
                np.array([ages, metallicities]).T,
                columns=["age", "[Fe/H]"],
            )

            # if host star mass is scalar, calculate planets for that mass and
            # multiply by imf around that region
            if isinstance(host_star_masses, (int, float)):
                # calculate planets per star using KNN interpolation of NGPPS results
                planets_per_star = planet_model.prediction(
                    category, host_star_masses, variables_dataframe
                )
                # calculate number of stars
                number_of_stars = imf.number_of_stars(particle_masses, *imf_bounds)
                # calculate total number of planets
                planets = planets_per_star.to_numpy()[:, 0] * number_of_stars

            # if host star mass is list, numerically integrate imf*number of stars
            elif isinstance(host_star_masses, tuple):
                # sort masses for interpolation
                sorted_masses = sorted(host_star_masses)

                # create integration space
                m_space = np.geomspace(*imf_bounds, num_integral_points)

                # linear interpolation of number of planets
                planets_per_star_function = interp1d(
                    sorted_masses,
                    np.array(
                        [
                            planet_model.prediction(
                                category, mass, variables_dataframe
                            ).to_numpy()[:, 0]
                            for mass in sorted_masses
                        ]
                    ),
                    axis=0,
                )
                planets_per_star = planets_per_star_function(m_space)

                # IMF njmber density contribution, value of the pdf rescaled for the
                # mass of the star particle
                imf_contribution = imf.number_density(particle_masses, m_space)

                # numerical integration
                planets = trapezoid(planets_per_star.T * imf_contribution, m_space)

            else:
                raise ValueError(
                    "ngpps_star_masses must either be number or a list of " "numbers."
                )

            return self.ds.arr(planets, "1")

        self.ds.add_field(
            ("stars", category),
            function=_planets,
            sampling_type="local",
            units="auto",
            dimensions=1,
        )

    def add_number_of_stars(
        self,
        imf: ChabrierIMF,
        imf_bounds: tuple[float, float] = (1, 1.04),
    ) -> None:
        """
        Add (number of) stars field to star particles.

        Parameters
        ----------
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.
        imf_bounds : tuple[float, float], optional
            The range over with to integrate the imf. Corresponds to the mass range
            of stars considered. The default is (1, 1.04).
        """

        self.check_star_properties()

        def _star_number(field: DerivedField, data: FieldDetector) -> NDArray:
            particle_masses = data["stars", "InitialMass"].to("Msun").value
            number_of_stars = imf.number_of_stars(particle_masses, *imf_bounds)
            return self.ds.arr(number_of_stars, "1")

        self.ds.add_field(
            ("stars", "number"),
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
            )  # set those values to -3

            # normalise to stellar fraction
            fe_abundance = log_fe_fraction - log_solar_fe_fraction
            return self.ds.arr(fe_abundance, "1")

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
            1 / np.array(data["PartType4", "GFM_StellarFormationTime"])
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

        # add interpolated redshift if z>0, otherwise add negative number which
        # makes formation time negative (indicating wind particles)
        formation_time = np.where(
            formation_redshift >= 0,
            np.interp(formation_redshift, redshift_grid, time_grid),
            2 * current_time,
        )
        return current_time - data.ds.arr(formation_time, "Gyr")

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
                "convert_PartType4_properties' method from Fields() first "
                "and then 'add_stars' method from Filter()."
            )
