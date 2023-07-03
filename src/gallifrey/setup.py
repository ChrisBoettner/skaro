#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:29:47 2023

@author: chris
"""

from typing import Any, Optional

from yt.frontends.arepo.data_structures import ArepoHDF5Dataset

from gallifrey.data.load import load_snapshot

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
    ngpps_id: str = "ng75",
    planet_params: Optional[dict[str, Any]] = None,
) -> tuple[ArepoHDF5Dataset, MainHalo, StellarModel, ChabrierIMF, PlanetModel]:
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
    ngpps_id : str, optional
        ID of the NGPPS population synthesis run for the planet model. The default is
        "ng75", which is a solar-like star with 50 embryos.
    planet_params: dict[str, Any], optional
        Additional parameter passed to fields.add_planets

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

    """
    # %%
    with Timer("Loading Hestia Snapshot.."):
        ds = load_snapshot(snapshot, resolution)
        mw = MainHalo("MW", resolution, ds, sim_id=sim_id)

        filters = Filter(ds)
        fields = Fields(ds)

    # %%
    with Timer("Adding Stars.."):
        stellar_model = StellarModel()
        imf = ChabrierIMF()

        filters.add_stars()
        fields.convert_star_properties()

        fields.add_main_sequence_stars(stellar_model, imf)
        fields.add_iron_abundance()

    # %%
    with Timer("Adding Planets.."):
        if planet_params is None:
            planet_params = {}

        planet_model = PlanetModel(ngpps_id)
        for category in planet_model.categories:
            fields.add_planets(category, planet_model, imf, **planet_params)

    with Timer("Other calculations.."):
        mw.insert(
            "BULGE_END",
            5,
            "in kpc. Rough estimate for run 09_18",
        )
        mw.insert(
            "DISK_END",
            18,
            "in kpc. Rough estimate for run 09_18",
        )

    return ds, mw, stellar_model, imf, planet_model