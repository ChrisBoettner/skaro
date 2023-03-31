#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:51:50 2023

@author: chris
"""

from typing import Any

import numpy as np
import yt
from numpy.typing import ArrayLike
from yt.data_objects.particle_filters import ParticleFilter
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

class Filter:
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

    def add_stars(self) -> None:
        """
        Add filter to ds that selects PartType4 particles with
        GFM_StellarFormationTime > 0 (meaning stars rather than wind particles).
        """

        @yt.particle_filter(
            requires=["GFM_StellarFormationTime"],
            filtered_type="PartType4",
        )
        def stars(
            pfilter: ParticleFilter,
            data: Any,
        ) -> ArrayLike:
            age_filter = data[(pfilter.filtered_type, "GFM_StellarFormationTime")] > 0
            return age_filter

        self.ds.add_particle_filter("stars")

    def add_halo_stars(self, ParticleIDs: ArrayLike) -> None:
        """
        Add filter to ds that selects PartType4 particles based on ID.

        Parameters
        ----------
        ParticleIDs : ArrayLike
            List of IDs to be filtered for.
        """

        @yt.particle_filter(
            requires=["ParticleIDs"],
            filtered_type="PartType4",
        )
        def halo_stars(
            pfilter: ParticleFilter,
            data: Any,
        ) -> ArrayLike:
            id_filter = np.in1d(
                data[(pfilter.filtered_type, "ParticleIDs")].value,
                ParticleIDs,
                assume_unique=True,
            )
            return id_filter

        self.ds.add_particle_filter("halo_stars")

    def add_halo_gas(self, ParticleIDs: ArrayLike) -> None:
        """
        Add filter to ds that selects PartType0 particles based on ID.

        Parameters
        ----------
        ParticleIDs : ArrayLike
            List of IDs to be filtered for.
        """

        @yt.particle_filter(
            requires=["ParticleIDs"],
            filtered_type="PartType0",
        )
        def halo_gas(
            pfilter: ParticleFilter,
            data: Any,
        ) -> ArrayLike:
            id_filter = np.in1d(
                data[(pfilter.filtered_type, "ParticleIDs")].value,
                ParticleIDs,
                assume_unique=True,
            )
            return id_filter

        self.ds.add_particle_filter("halo_gas")
