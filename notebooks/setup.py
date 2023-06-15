#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:58:51 2023

@author: chris
"""


def data_setup(snapshot: int = 127, resolution: int = 4096, sim_id: str = "09_18"):
    # %%
    # load local version before pip installed version, for debugging
    import pathlib
    import sys

    sys.path.append(pathlib.Path(__file__).parent.parent.joinpath("src"))

    # %%
    from gallifrey.data.load import load_snapshot

    # from gallifrey.visualization.manager import DefaultFigureManager as fm
    from gallifrey.fields import Fields
    from gallifrey.filter import Filter
    from gallifrey.halo import MainHalo
    from gallifrey.planets import PlanetModel
    from gallifrey.stars import ChabrierIMF, StellarModel
    from gallifrey.utilities.time import Timer

    # %%
    with Timer("load data"):
        ds = load_snapshot(snapshot, resolution)
        mw = MainHalo("MW", resolution, ds, sim_id=sim_id)

        filters = Filter(ds)
        fields = Fields(ds)

    # %%
    with Timer("stars"):
        stellar_model = StellarModel()
        imf = ChabrierIMF()

        filters.add_stars()
        fields.convert_star_properties()

        fields.add_main_sequence_stars(stellar_model, imf)
        fields.add_iron_abundance()

    # %%
    with Timer("planets"):
        planet_model = PlanetModel()
        fields.add_planets(stellar_model, planet_model, imf)

    with Timer("other"):
        filters.add_old_stars()
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
