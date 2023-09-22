#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:29:47 2023

@author: chris
"""
import os
from typing import Any, Optional

import numpy as np
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset

from gallifrey.data.load import load_snapshot
from gallifrey.decomposition.mordor import galaxy_components

# from gallifrey.visualization.manager import DefaultFigureManager as fm
from gallifrey.fields import Fields
from gallifrey.filter import Filter
from gallifrey.halo import MainHalo
from gallifrey.planets import PlanetModel
from gallifrey.stars import ChabrierIMF, StellarModel
from gallifrey.utilities.logging import logger
from gallifrey.utilities.time import Timer

# create Logger
logger = logger(__name__)


def data_setup(
    snapshot: int = 127,
    resolution: int = 4096,
    sim_id: str = "09_18",
    ngpps_num_embryos: int = 50,
    ngpps_star_masses: float | tuple[float, ...] = 1,
    star_age_bounds: tuple[float, float] = (0.02, np.inf),
    planet_hosting_imf_delta: float = 0.05,
    planet_params: Optional[dict[str, Any]] = None,
    calculate_components: bool = True,
) -> tuple[ArepoHDF5Dataset, MainHalo, StellarModel, ChabrierIMF, PlanetModel, str]:
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
    ngpps_num_embryos : tuple[int, float], optional
        Parameter describing the NGPPS population run, number of embryos used for the
        run. The default is 50.
    ngpps_star_masses: float | tuple[float, ...]
        Host star masses considered in the planet population calculation. Can be scalar
        (in which case the imf_delta is invoked for the IMF integration) or a tuple. The
        values must be in the NGPPS runs (0.1, 0.3, 0.5, 0.7, 1). Masses < 1 only
        available in ngpps_num_embryos = 50. The default is 1.
    star_age_bounds : tuple[float, float], optional
        The age range for star particles to be considered in the add_stars
        command. The default is (0.02, np.inf).
    planet_hosting_imf_delta : float, optional
        If ngpps_star_masses is a single number, the IMF is integrated in the range
        ((1-imf_delta_mass)*ngpps_star_mass, (1+imf_delta)*ngpps_star_masses). No effect
        if the ngpps_star_masses span a range. The default is 0.05.
    lower_stellar_age_bound: float, optional
        Lower age of star particles considered. The upper end is inferred based on the
        stellar model and upper IMF bound. The default is 0.02, i.e. 20Myr, the planet
        formation time in the NGPPS model.
    planet_params: dict[str, Any], optional
        Additional parameter passed to fields.add_planets
    calculate_components: bool, optional
        If True, calculates galaxy components and adds correspoding fields. The default
        is True.

    Returns
    -------
    ds : ArepoHDF5Dataset
        The full Hestia simulation dataset for the snapshot.
    mw : MainHalo
        Halo object containing information about the Milky Way Halo.
    stellar_model : StellarModel
        The model containing the stellar properties and prescriptions.
    imf : ChabrierIMF
        The Chabier IMF object.
    planet_model : PlanetModel
        The planet model.
    snapshot_path : str
        Path to the snapshot.

    """
    # %%
    if os.environ.get("USER") == "chris":  # if local system, load the test file
        logger.info("DETECTED LOCAL MACHINE: Test snapshot loaded.")
        resolution = 4096
        snapshot = 127
        test_flag = True  # load test snapshot on local machine
    else:
        test_flag = False

    with Timer("Loading Hestia Snapshot..."):
        ds, snapshot_path = load_snapshot(snapshot, resolution, test_flag=test_flag)
        mw = MainHalo("MW", resolution, ds, sim_id=sim_id, test_flag=test_flag)

        filters = Filter(ds)
        fields = Fields(ds)

    with Timer("Loading ParticleIDs..."):
        mw_IDs = mw.particle_IDs()

    # %%
    with Timer("Adding Stars..."):
        stellar_model = StellarModel()
        imf = ChabrierIMF()

        if isinstance(ngpps_star_masses, (int, float)):
            planet_hosting_imf_bounds = (
                (1 - planet_hosting_imf_delta) * ngpps_star_masses,
                (1 + planet_hosting_imf_delta) * ngpps_star_masses,
            )
        elif isinstance(ngpps_star_masses, tuple):
            planet_hosting_imf_bounds = (
                np.amin(ngpps_star_masses),
                np.amax(ngpps_star_masses),
            )
        else:
            raise ValueError(
                "ngpps_star_masses must either be number (int, float) or a list of "
                "numbers."
            )

        logger.info(
            "STARS: 'stars' field derives from PartType4 field in age range: "
            f"{[np.round(bound, 2) for bound in star_age_bounds]} Gyr."
        )

        fields.convert_PartType4_properties()
        filters.add_stars(age_limits=star_age_bounds)

        fields.add_number_of_stars(
            stellar_model,
            imf,
            planet_hosting_imf_bounds=planet_hosting_imf_bounds,
        )
        fields.add_iron_abundance()
        fields.add_alpha_abundance()

        normal_vector = mw.normal_vector("stars", data=mw.sphere(radius=(10, "kpc")))
        fields.add_height(normal_vector)
        fields.add_planar_radius(normal_vector)

        if calculate_components:
            component_dataframe = galaxy_components(
                halo=mw,
                snapshot_path=snapshot_path + f"/snapshot_{snapshot}",
                mode="ID",
                id_list=mw_IDs["ParticleIDs"],
            )
            filters.add_galaxy_components(component_dataframe)

    # %%
    with Timer("Adding Planets..."):
        if planet_params is None:
            planet_params = {}

        planet_model = PlanetModel(ngpps_num_embryos)
        for category in planet_model.categories:
            fields.add_planets(
                category,
                ngpps_star_masses,
                planet_model,
                stellar_model,
                imf,
                planet_hosting_imf_bounds,
                **planet_params,
            )

    return ds, mw, stellar_model, imf, planet_model, snapshot_path
