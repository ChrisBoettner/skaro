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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from unyt.array import unyt_array
from yt.fields.derived_field import DerivedField
from yt.fields.field_detector import FieldDetector
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.planets import PlanetModel
from gallifrey.stars import ChabrierIMF, StellarModel
from gallifrey.utilities.logging import logger
from gallifrey.utilities.structures import find_closest

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
        stellar_model: StellarModel,
        imf: ChabrierIMF,
        planet_hosting_number: tuple[float, float],
        reference_age: int | None = 100000000,
        num_integral_points: int = 50,
        only_mainsequence: bool = True,
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
        stellar_model: StellarModel
            Stellar Model, used to calculate lifetime of stars for upper integration
            bound.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles..
        planet_hosting_number : tuple[float, float]
            The range over with to integrate the imf. Corresponds to the mass range
            of stars considered.
        reference_age : int | None, optional
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
            try:
                number_of_stars = data["stars", "planet_hosting_number"]
            except KeyError:
                raise KeyError(
                    "['stars', 'number'] field does not exist. Create first "
                    "using add_number_of_stars method."
                )

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
                # calculate total number of planets
                planets = planets_per_star.to_numpy()[:, 0] * number_of_stars

            # if host star mass is list, numerically integrate imf*number of stars
            elif isinstance(host_star_masses, tuple):
                # sort masses for interpolation
                sorted_masses = sorted(host_star_masses)
                # calculate upper bound fro integration, maximum is upper limit of
                # imf_bound, but if particle is old enough cut might need to be earler
                # due to stars already going off main sequence
                upper_bound = stellar_model.mass_from_lifetime(stellar_ages)
                upper_bound[
                    upper_bound > planet_hosting_number[1]
                ] = planet_hosting_number[1]

                # create integration space
                m_space = np.geomspace(*planet_hosting_number, num_integral_points)

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

                # numerical integration:
                # respect upper integration bound by integration to maximum, saving
                # all steps and then taking the one clostest to the upper_limit
                planets = cumulative_trapezoid(
                    planets_per_star.T * imf_contribution, m_space, initial=0
                )
                closest_index_to_limit = find_closest(
                    upper_bound, m_space, return_index=True
                )
                planets = planets[np.arange(planets.shape[0]), closest_index_to_limit]

            else:
                raise ValueError(
                    "ngpps_star_masses must either be number or a tuple of numbers."
                )

            return self.ds.arr(planets, "1")

        self.ds.add_field(
            ("stars", category),
            function=_planets,
            sampling_type="local",
            units="auto",
            dimensions=1,
            force_override=True,
        )

    def add_total_star_number(
        self,
        stellar_model: StellarModel,
        imf: ChabrierIMF,
        imf_bounds: Optional[tuple[float, float]] = None,
        only_mainsequence: bool = True,
    ) -> None:
        """
        Add total number of stars field to star particles, corresponding to
        number of stars between imf_bounds. If no value is given,
        defaults to bounds of imf object.

        Parameters
        ----------
        stellar_model: StellarModel
            Stellar Model, used to calculate lifetime of stars for upper integration
            bound.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.
        imf_bounds : Optional[tuple[float, float]], optional
            The range over with to integrate the imf for total_number field. The default
            is None, which uses imf object bounds.
        only_mainsequence: bool, optional
            If True, only integrate IMF up to the mass where the stellar lifetime
            matches the star particle age, i.e. only include main sequence stars. If
            False, integrate up to the given upper IMF bound in all cases. The default
            is True.

        """
        self.check_star_properties()

        # add total number of stars
        def _total_star_number(field: DerivedField, data: FieldDetector) -> NDArray:
            # if no IMF bounds are given, use bounds of imf object
            if imf_bounds is None:
                bounds = (imf.a, imf.b)
            else:
                bounds = imf_bounds
            return self._star_number(
                field,
                data,
                imf=imf,
                bounds=bounds,
                stellar_model=stellar_model,
                only_mainsequence=only_mainsequence,
            )

        self.ds.add_field(
            ("stars", "total_number"),
            function=_total_star_number,
            sampling_type="local",
            units="auto",
            dimensions=1,
            force_override=True,
        )

    def add_planet_hosting_star_number(
        self,
        stellar_model: StellarModel,
        imf: ChabrierIMF,
        planet_hosting_imf_bounds: Optional[tuple[float, float]] = None,
        only_mainsequence: bool = True,
    ) -> None:
        """
        Add number of planet hosting stars field to star particles.
        planet_hosting_number corresponds to stars between planet_hosting_imf_bounds.

        Parameters
        ----------
        stellar_model: StellarModel
            Stellar Model, used to calculate lifetime of stars for upper integration
            bound.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.
        planet_hosting_imf_bounds : Optional[tuple[float, float]], optional
            The range over with to integrate the imf for planet_hosting_number. The
            default is None. In that case, planet_hosting_number will not be created.
        only_mainsequence: bool, optional
            If True, only integrate IMF up to the mass where the stellar lifetime
            matches the star particle age, i.e. only include main sequence stars. If
            False, integrate up to the given upper IMF bound in all cases. The default
            is True.

        """
        self.check_star_properties()

        if planet_hosting_imf_bounds is not None:

            def _planet_star_number(
                field: DerivedField, data: FieldDetector
            ) -> NDArray:
                return self._star_number(
                    field,
                    data,
                    imf=imf,
                    bounds=planet_hosting_imf_bounds,
                    stellar_model=stellar_model,
                    only_mainsequence=only_mainsequence,
                )

            self.ds.add_field(
                ("stars", "planet_hosting_number"),
                function=_planet_star_number,
                sampling_type="local",
                units="auto",
                dimensions=1,
                force_override=True,
            )
        else:
            logger.warn(
                "FIELDS: No planet_hosting_imf_bounds bounds given to "
                "add_planet_hosting_star_number. planet_hosting_number field not "
                "created."
            )

    def add_iron_abundance(self, log_solar_fe_fraction: float = -2.7) -> None:
        """
        Add iron abundanace [Fe/H].

        Parameters
        ----------
        log_solar_fe_fraction : float
            Solar iron fraction, m_Fe/m_H.

        """
        self.check_star_properties()

        def _iron_abundance(field: DerivedField, data: FieldDetector) -> NDArray:
            # calculate iron abundance for star particles
            fe_fraction = (
                data["stars", "Fe_fraction"].value / data["stars", "H_fraction"].value
            )
            log_fe_fraction = np.where(
                fe_fraction > 0,
                np.ma.log10(fe_fraction),
                -10,
            )  # set values<0 values to -3

            # normalise to stellar fraction
            fe_abundance = log_fe_fraction - log_solar_fe_fraction
            # deal with outliers
            return self.ds.arr(fe_abundance, "1")

        self.ds.add_field(
            ("stars", "[Fe/H]"),
            function=_iron_abundance,
            sampling_type="local",
            units="auto",
            dimensions=1,
            force_override=True,
        )

    def add_alpha_abundance(self, log_solar_alpha_fe_fraction: float = 1.09) -> None:
        """
        Add alpha element abundance [alpha/Fe].

        Parameters
        ----------
        log_solar_alpha_fe_fraction : float
            Solar alpha element fraction, m_alpha/m_Fe.

        """
        self.check_star_properties()

        def _alpha_abundance(field: DerivedField, data: FieldDetector) -> NDArray:
            # calculate alpha element masses for star particles
            alpha_elements = np.sum(
                np.array(
                    [
                        data["stars", fraction].value
                        for fraction in [
                            "O_fraction",
                            "C_fraction",
                            "N_fraction",
                            "Mg_fraction",
                            "Si_fraction",
                            "Ne_fraction",
                        ]
                    ]
                ),
                axis=0,
            )

            # calculate alpha/Fe for valid inputs
            valid_condition = (data["stars", "Fe_fraction"].value > 0) & (
                alpha_elements > 0
            )
            alpha_fraction = np.where(
                valid_condition,
                np.ma.divide(alpha_elements, data["stars", "Fe_fraction"].value),
                1,
            )

            log_alpha_fraction = np.ma.log10(alpha_fraction)

            # normalise to stellar fraction
            alpha_abundance = log_alpha_fraction - log_solar_alpha_fe_fraction
            return self.ds.arr(alpha_abundance, "1")

        self.ds.add_field(
            ("stars", "[alpha/Fe]"),
            function=_alpha_abundance,
            sampling_type="local",
            units="auto",
            dimensions=1,
            force_override=True,
        )

    def add_angular_momentum_alignment(self, normal_vector: np.ndarray) -> None:
        """
        Add alignment of angular momentum of star particles, defined by ratio between
        angular momentum component normal to galactic plane divided by magnitude of
        total angular momentum.

        Parameters
        ----------
        normal_vector : np.ndarray
            Normal vector to galactic plane.

        """
        self.check_star_properties()

        # scale vector to magnitude = 1
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        def _angular_momentum_alignment(
            field: DerivedField,
            data: FieldDetector,
        ) -> NDArray:
            # calculate relative velocity within dataset
            mean_velocity = np.mean(data["stars", "particle_velocity"], axis=0)
            relative_velocity = data["stars", "particle_velocity"] - mean_velocity

            # calculate specific angular momentum
            specific_angular_momentum = self.ds.arr(
                np.cross(
                    data["stars", "relative_particle_position"].to("km"),
                    relative_velocity.to("km/s"),
                ),
                "km**2/s",
            )

            # calculate between normal component and magnitude
            normal_component = np.dot(specific_angular_momentum, normal_vector)
            magnitude = np.sqrt(np.sum(specific_angular_momentum**2, axis=1))

            # define positive normal direction so that most stars are co-rotation
            # (meaning if more stars are counter rotating, switch sign around)
            num_co_rotating = np.count_nonzero(normal_component > 0)
            num_anti_rotating = np.count_nonzero(normal_component < 0)
            if num_anti_rotating > num_co_rotating:
                normal_component = -normal_component

            # calculate circularity as ratio between normal component and magnitude
            circularity = np.where(
                magnitude > 0,
                np.ma.divide(normal_component, magnitude),
                0,
            )
            return self.ds.arr(circularity, "1")

        self.ds.add_field(
            ("stars", "angular_momentum_alignment"),
            function=_angular_momentum_alignment,
            sampling_type="local",
            units="auto",
            dimensions=1,
            force_override=True,
        )

    def add_height(self, normal_vector: np.ndarray) -> None:
        """
        Add height over galactic plane.

        Parameters
        ----------
        normal_vector : np.ndarray
            Normal vector to galactic plane.

        """
        self.check_star_properties()

        # scale vector to magnitude = 1
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        def _height(field: DerivedField, data: FieldDetector) -> NDArray:
            coordinates = data["stars", "relative_particle_position"]
            height = np.dot(coordinates, normal_vector).to("kpc").value
            return self.ds.arr(height, "kpc")

        self.ds.add_field(
            ("stars", "height"),
            function=_height,
            sampling_type="local",
            units="kpc",
            force_override=True,
        )

    def add_planar_radius(self, normal_vector: np.ndarray) -> None:
        """
        Add radial distance projected onto the galactic plane.

        Parameters
        ----------
        normal_vector : np.ndarray
            Normal vector to galactic plane.

        """
        self.check_star_properties()

        # scale vector to magnitude = 1
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        def _planar_radius(field: DerivedField, data: FieldDetector) -> NDArray:
            coordinates = data["stars", "relative_particle_position"]
            height = np.dot(coordinates, normal_vector).to("kpc").value
            distances = data["stars", "particle_radius"].to("kpc").value

            planar_radius = np.sqrt(distances**2 - height**2)
            return self.ds.arr(planar_radius, "kpc")

        self.ds.add_field(
            ("stars", "planar_radius"),
            function=_planar_radius,
            sampling_type="local",
            units="kpc",
            force_override=True,
        )

    @staticmethod
    def _stellar_age(
        field: DerivedField,
        data: FieldDetector,
        interpolation_num: int = 500,
    ) -> unyt_array:
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

        Returns
        -------
        unyt_array
           Array containing the ages of stars in Gyr.
        """
        # get current simulation time, and formation redshifts of star particles from
        # scale factor
        current_time = data.ds.current_time.to("Gyr")
        formation_redshift = (
            1 / np.array(data["PartType4", "GFM_StellarFormationTime"])
        ) - 1

        if len(formation_redshift) == 0:
            return data.ds.arr(np.array([]), "Gyr")

        # fudging away numerical issues with formation time calculation,
        # this should only affect a small number particles
        formation_redshift[formation_redshift < 0] = 10

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

    @staticmethod
    def _star_number(
        field: DerivedField,
        data: FieldDetector,
        imf: ChabrierIMF,
        bounds: tuple[float, float],
        stellar_model: StellarModel,
        only_mainsequence: bool,
    ) -> unyt_array:
        """
        Calculate number of stars from IMF between two bounds.

        Parameters
        ----------
        field : DerivedField
            Field parameter used for adding field to yt Dataset.
        data : FieldDetector
            Data parameter used for adding field to yt Dataset.
        imf: ChabrierIMF
            The IMF used for integration.
        bounds:
            The bounds for the IMF integration.
        stellar_model : StellarModel
            The stellar model, used for calculating main sequence ages
        only_mainsequence : bool
            Choose if only main sequence stars should be included or not.

        Returns
        -------
        unyt_array
           Array containing the number of stars.
        """
        if only_mainsequence:
            # calculate upper bound based on star particle age and stellar
            # lifetime
            stellar_ages = data["stars", "stellar_age"].value
            upper_bound = stellar_model.mass_from_lifetime(stellar_ages)
            upper_bound[upper_bound > bounds[1]] = bounds[1]
        else:
            upper_bound = bounds[1]

        particle_masses = data["stars", "InitialMass"].to("Msun").value
        number_of_stars = imf.number_of_stars(particle_masses, bounds[0], upper_bound)
        return data.ds.arr(number_of_stars, "1")

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
