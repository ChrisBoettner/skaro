#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:10:14 2023

@author: chris
"""

import numpy as np
from astropy.cosmology import Planck15
from yt.fields.derived_field import DerivedField
from yt.fields.field_detector import FieldDetector
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset


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

    def add_stellar_age(self) -> None:
        """
        Add filter to ds that selects PartType4 particles with
        GFM_StellarFormationTime>0 (meaning stars rather than wind particles).
        """

        def _stellar_age(
            field: DerivedField,
            data: FieldDetector,
            interpolation_num: int = 500,
        ) -> None:
            current_time = data.ds.current_time.to("Gyr")
            formation_redshift = (
                1 / np.array(data["stars", "GFM_StellarFormationTime"]) - 1
            )
            max_redshift = np.amax(formation_redshift)

            redshift_grid = np.linspace(
                data.ds.current_redshift, max_redshift, interpolation_num
            )
            time_grid = Planck15.age(redshift_grid).value

            formation_time = data.ds.arr(
                np.interp(formation_redshift, redshift_grid, time_grid),
                "Gyr",
            )

            return current_time - formation_time

        self.ds.add_field(
            ("stars", "stellar_age"),
            function=_stellar_age,
            sampling_type="local",
            units="Gyr",
            force_override=True,
        )
