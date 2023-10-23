#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:29:47 2023

@author: chris
"""
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset

from gallifrey.data.load import load_snapshot
from gallifrey.decomposition.mordor import galaxy_components
from gallifrey.fields import Fields
from gallifrey.filter import Filter
from gallifrey.halo import MainHalo
from gallifrey.planets import PlanetModel
from gallifrey.stars import ChabrierIMF, StellarModel
from gallifrey.utilities.logging import logger
from gallifrey.utilities.time import Timer

# create Logger
logger = logger(__name__)


class Model:
    """
    Object that contains simulation snapshot and model components.

    """

    def __init__(
        self,
        snapshot: int = 127,
        resolution: int = 4096,
        sim_id: str = "09_18",
        halo_name: str = "MW",
        star_age_bounds: tuple[float, float] = (0.02, np.inf),
        ngpps_num_embryos: int = 50,
        ngpps_star_masses: float | tuple[float, ...] = 1,
        ngpps_hard_bounds: str = "none",
        planet_hosting_imf_delta: float = 0.05,
        planet_params: Optional[dict[str, Any]] = None,
        calculate_components: bool = True,
        force_decomposition_calculation: bool = False,
    ) -> None:
        """
        Load Hestia simulation snapshot, create Milky way halo object and
        add stellar/imf/planet properties and fields.

        Parameters
        ----------
        snapshot : int, optional
            Snapshot number of the Hestia snapshot to load. The default is 127, which is
            the at z=0.
        resolution : int, optional
            Resolution of the Hestia run to load. The default is 4096.
        sim_id : str, optional
            ID of the specific Hestia run. The default is "09_18".
        halo_name: str, optional
            The name of the main Halo. The default is "MW".
        star_age_bounds : tuple[float, float], optional
            The age range for star particles to be considered in the add_stars
            command. The default is (0.02, np.inf).
        ngpps_num_embryos : tuple[int, float], optional
            Parameter describing the NGPPS population run, number of embryos used for
            the run. The default is 50.
        ngpps_star_masses: float | tuple[float, ...]
            Host star masses considered in the planet population calculation. Can be
            scalar (in which case the imf_delta is invoked for the IMF integration) or
            a tuple. The values must be in the NGPPS runs (0.1, 0.3, 0.5, 0.7, 1).
            Masses < 1 only available in ngpps_num_embryos = 50. The default is 1.
        ngpps_hard_bounds: str, optional
            Star particles with parameter outside of reference range will be assigned
            zero planets, if bounds are "lower", "upper" or "both". No bounds condition
            can be set using "none". This is especially relevant for the metallicity
            considerations, where star particles with metallicities below the miminimum
            value of the NGPPS sample ([Fe/H] = -0.6) are assigned zero planets. The
            default is "none". For more details, see
            gallifrey.utilities.dataframe.within_bounds.
        planet_hosting_imf_delta : float, optional
            If ngpps_star_masses is a single number, the IMF is integrated in the range
            ((1-imf_delta_mass)*ngpps_star_mass, (1+imf_delta)*ngpps_star_masses). No
            effect if the ngpps_star_masses span a range. The default is 0.05.
        lower_stellar_age_bound: float, optional
            Lower age of star particles considered. The upper end is inferred based on
            the stellar model and upper IMF bound. The default is 0.02, i.e. 20Myr, the
            planet formation time in the NGPPS model.
        planet_params: dict[str, Any], optional
            Additional parameter passed to fields.add_planets
        calculate_components: bool, optional
            If True, calculates galaxy components and adds correspoding fields. The
            default is True.
        force_decomposition_calculation: bool, optional
            If True, galaxy components are recalculated and not loaded from external
            file, even if that's possible. The default is False.

        """
        # config attributes
        self.config_list = [
            "snapshot",
            "resolution",
            "sim_id",
            "halo_name",
            "star_age_bounds",
            "ngpps_num_embryos",
            "ngpps_star_masses",
            "ngpps_hard_bounds",
            "planet_hosting_imf_delta",
            "planet_params",
        ]

        # initialize attributes attributes
        self.snapshot = snapshot
        self.resolution = resolution
        self.sim_id = sim_id
        self.halo_name = halo_name
        self.star_age_bounds = star_age_bounds
        self.ngpps_num_embryos = ngpps_num_embryos
        self.ngpps_star_masses = ngpps_star_masses
        self.ngpps_hard_bounds = ngpps_hard_bounds
        self.planet_hosting_imf_delta = planet_hosting_imf_delta
        if planet_params is None:
            planet_params = {}
        self.planet_params = planet_params

        # check if local testsnapshot should be loaded instead
        self.test_flag = self.check_for_local_machine()

        # load snapshot, snapshot_path, fields, filters and halo
        (
            self.ds,
            self.filters,
            self.fields,
            self.snapshot_path,
        ) = self.load_galaxy_simulation()
        self.halo = self.load_halo()

        # add star and planet fields
        self.fields.convert_PartType4_properties()
        self.stellar_model, self.imf = self.add_star_fields()
        self.planet_model = self.add_planet_fields()

        # calculate decomposition
        self.halo_ids = None
        if calculate_components:
            self.halo_ids = self.load_particle_ids()
            self.calculate_galaxy_decomposition(force_decomposition_calculation)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Modification of setattr, so that information is printed if one of the config
        parameters is changed.

        Parameters
        ----------
        name : str
            Name of attribute.
        value : Any
            Value of attribute.

        """
        if (not hasattr(self, "config_list")) and name == "config_list":
            # pass to add config_list to attributes
            pass
        elif (name in self.config_list) and hasattr(self, name):
            # print change if config value is changed (after it has already been set)
            if not getattr(self, name) == value:
                logger.info(f"CONFIG: Setting {name} to {value}.")
        super().__setattr__(name, value)

    def get_config(self) -> dict[str, Any]:
        """
        Get all config variables and their values.

        Returns
        -------
        dict[str, Any]
            Dict with config variables and values.

        """
        return {attr: getattr(self, attr) for attr in self.config_list}

    def check_for_local_machine(
        self,
        default_resolution: int = 4096,
        default_snapshot: int = 127,
        default_sim_id: str = "09_18",
    ) -> bool:
        """
        Check if local machine is used. If so, change resolution, snapshot and sim_id
        to values corresponding to local test snapshot.

        Parameters
        ----------
        default_resolution: int, optional
            The default resolution value. The default is 4096.
        default_snapshot: int, optional
            The default snapshot value. The default is 127.
        default_sim_id : str, optional
            The default simulation run ID. The default is "09_18".

        Returns
        -------
        test_flag: bool
            Flag if test snapshot is used or not.

        """
        if os.environ.get("USER") == "chris":  # if local system, load the test file
            logger.info("DETECTED LOCAL MACHINE: Test snapshot loaded.")
            self.resolution = default_resolution
            self.snapshot = default_snapshot
            self.sim_id = default_sim_id
            test_flag = True
        else:
            test_flag = False
        return test_flag

    def load_galaxy_simulation(
        self,
    ) -> tuple[ArepoHDF5Dataset, Filter, Fields, str]:
        """
        Load galaxy simulation snapshot, and the filter and fields objects based on that
        snapshot.

        Returns
        -------
        tuple[ArepoHDF5Dataset, Filter, Fields, str]
            The snapshot, filters, fields and snapshot_path.

        """
        with Timer("Loading Hestia Snapshot..."):
            ds, snapshot_path = load_snapshot(
                self.snapshot,
                self.resolution,
                sim_id=self.sim_id,
                test_flag=self.test_flag,
            )

            filters = Filter(ds)
            fields = Fields(ds)
        return ds, filters, fields, snapshot_path

    def load_halo(self) -> MainHalo:
        """
        Load (main) halo object.

        Returns
        -------
        MainHalo
            The MainHalo yt object.

        """
        return MainHalo(
            self.halo_name,
            self.resolution,
            self.ds,
            sim_id=self.sim_id,
            test_flag=self.test_flag,
        )

    def add_star_fields(self) -> tuple[StellarModel, ChabrierIMF]:
        """
        Create stellar model object and imf model object, create star field and add
        total number of stars, as well as iron abundance, alpha abundance.
        Additional adds height over galactic plane and planar radius fields for stars.

        Returns
        -------
        tuple[StellarModel, ChabrierIMF]
            The stellar model and imf objects.

        """
        with Timer("Adding Stars..."):
            stellar_model = StellarModel()
            imf = ChabrierIMF()

            logger.info(
                "STARS: 'stars' field derives from PartType4 field in age range: "
                f"{[np.round(bound, 2) for bound in self.star_age_bounds]} Gyr."
            )

            self.filters.add_stars(age_limits=self.star_age_bounds)

            self.fields.add_total_star_number(
                stellar_model,
                imf,
            )

            self.fields.add_iron_abundance()
            self.fields.add_alpha_abundance()

            normal_vector = self.halo.normal_vector(
                "stars", data=self.halo.sphere(radius=(10, "kpc"))
            )
            self.fields.add_height(normal_vector)
            self.fields.add_planar_radius(normal_vector)
        return stellar_model, imf

    def planet_hosting_imf_bounds(self) -> tuple[float, float]:
        """
        The bounds for the IMF integration for the stars considered planet hosting,
        based on the star_masses and

        Returns
        -------
        tuple[float, float]
            The bounds (lower bound, upper bound).

        """
        if isinstance(self.ngpps_star_masses, (int, float)):
            planet_hosting_imf_bounds = (
                (1 - self.planet_hosting_imf_delta) * self.ngpps_star_masses,
                (1 + self.planet_hosting_imf_delta) * self.ngpps_star_masses,
            )
        elif isinstance(self.ngpps_star_masses, tuple):
            planet_hosting_imf_bounds = (
                np.amin(self.ngpps_star_masses),
                np.amax(self.ngpps_star_masses),
            )
        else:
            raise ValueError(
                "ngpps_star_masses must either be number (int, float) or a list of "
                "numbers."
            )
        return planet_hosting_imf_bounds

    def add_planet_fields(
        self,
    ) -> PlanetModel:
        """
        Create planet_model, add number of planet hosting stars and planet fields.

        Returns
        -------
        planet_model : PlanetModel
            The planet model.

        """
        with Timer("Adding Planets..."):
            if self.planet_params is None:
                self.planet_params = {}

            self.fields.add_planet_hosting_star_number(
                self.stellar_model,
                self.imf,
                planet_hosting_imf_bounds=self.planet_hosting_imf_bounds(),
            )

            planet_model = PlanetModel(self.ngpps_num_embryos)
            for category in planet_model.categories:
                self.fields.add_planets(
                    category,
                    host_star_masses=self.ngpps_star_masses,
                    planet_model=planet_model,
                    stellar_model=self.stellar_model,
                    imf=self.imf,
                    planet_hosting_imf_bounds=self.planet_hosting_imf_bounds(),
                    hard_bounds=self.ngpps_hard_bounds,
                    **self.planet_params,
                )
        return planet_model

    def update_fields(self, **config: Any) -> None:
        """
        Update the planet and stellar field with new config values.

        Parameters
        ----------
        **config : Any
            The new values to be used in the config for the field creation.

        """
        for attr_name, new_value in config.items():
            if attr_name not in self.config_list:
                logger.warn(
                    f"WARNING: {attr_name} of config attribute not in " "config_list."
                )

            if new_value is not None:
                setattr(self, attr_name, new_value)

        # update star fields if star parameters are changed
        if "star_age_bounds" in config.keys():
            self.stellar_model, self.imf = self.add_star_fields()

        # update planet fields if any of the planet related parameter are changed
        if not set(
            (
                "ngpps_num_embryos",
                "ngpps_star_masses",
                "ngpps_hard_bounds",
                "planet_hosting_imf_delta",
            )
        ).isdisjoint(set(config.keys())):
            self.planet_model = self.add_planet_fields()
        return

    def load_particle_ids(
        self,
    ) -> pd.DataFrame:
        """
        Load particle IDs coneccted to halo

        Returns
        -------
        pd.DataFrame
            The particle IDs and corresponding particle types associated with the halo.

        """
        with Timer("Loading Particle IDs..."):
            halo_ids = self.halo.particle_IDs()
        return halo_ids

    def calculate_galaxy_decomposition(self, force_calculation: bool = False) -> None:
        """
        Calculate galaxy decomposition and add corresponding fields.

        Parameters
        ----------
        force_calculation : bool, optional
            Choose if decomposition calculation should be done, even if values could be
            loaded from file. The default is False

        """
        with Timer("Galaxy Decomposition..."):
            component_dataframe = galaxy_components(
                halo=self.halo,
                snapshot_path=self.snapshot_path + f"/snapshot_{self.snapshot}",
                mode="ID",
                id_list=self.halo_ids,
                force_calculation=force_calculation,
            )
            self.filters.add_galaxy_components(component_dataframe)
        return
