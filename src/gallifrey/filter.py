#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:51:50 2023

@author: chris
"""

from typing import Any, Optional

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

    def add_stars(self, age_limits: Optional[tuple[float, float]] = None) -> None:
        """
        Add filter to ds that selects PartType4 particles with stellar_age > 0
        (meaning stars rather than wind particles).
        The particles can further be filtered to only fall into a certain age_range
        using age_limits.

        Parameters
        ----------
        age_limits : Optional[tuple[float,float]], optional
            If given, only include particles that have an age between the age_limits.

        """

        @yt.particle_filter(
            requires=["stellar_age"],
            filtered_type="PartType4",
        )
        def stars(
            pfilter: ParticleFilter,
            data: Any,
        ) -> ArrayLike:
            if age_limits:
                # if age limits are given, get stars between these limits
                age_filter = (
                    age_limits[0] <= data[(pfilter.filtered_type, "stellar_age")]
                ) & (data[(pfilter.filtered_type, "stellar_age")] <= age_limits[1])
            else:
                # otherwise only filter out wind particles
                age_filter = data[(pfilter.filtered_type, "stellar_age")] >= 0
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

    def add_galaxy_components(
        self,
        spheroid_circularity_cut: float = 0.2,
        thin_disk_circularity_cut: float = 0.6,
        thin_disk_height_cut: float = 1,
        spheroid_halo_seperation: float = 2.88,
    ) -> None:
        """
        Add stars in different components of galaxy (thin disk, thick disk, spheroid)
        based on their circularity and height over galactic plane.

        Parameters
        ----------
        spheroid_circularity_cut : float, optional
            Circularity below which values are categorized as spheroid. The default
            is 0.2.
        thin_disk_circularity_cut : float, optional
            Circularity above which values are categorized as thin_disk. The default is
            0.6.
        thin_disk_height_cut : float, optional
            Maximum height over galactic plane (in kpc) for classification into
            thin disk. The default is 1.
        bulge_halo_seperation: float, optional
            Radius from galactic center seperating bulge and halo (in kpc). The default
            value is 3, based on the Sérsic profile effective radius given in
            Libeskind2020.

        """

        @yt.particle_filter(
            requires=["circularity"],
            filtered_type="stars",
        )
        def spheroid_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            spheroid_filter = (
                data[(pfilter.filtered_type, "circularity")] <= spheroid_circularity_cut
            )
            return spheroid_filter

        @yt.particle_filter(
            requires=["particle_radius"],
            filtered_type="spheroid_stars",
        )
        def bulge_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            spheroid_filter = (
                data[(pfilter.filtered_type, "particle_radius")].to("kpc")
                <= spheroid_halo_seperation
            )
            return spheroid_filter

        @yt.particle_filter(
            requires=["particle_radius"],
            filtered_type="spheroid_stars",
        )
        def halo_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            spheroid_filter = (
                data[(pfilter.filtered_type, "particle_radius")].to("kpc")
                > spheroid_halo_seperation
            )
            return spheroid_filter

        @yt.particle_filter(
            requires=["circularity"],
            filtered_type="stars",
        )
        def thin_disk_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            circularity_filter = (
                data[(pfilter.filtered_type, "circularity")]
                >= thin_disk_circularity_cut
            )
            height_filter = (
                np.abs(data[(pfilter.filtered_type, "height")].to("kpc"))
                <= thin_disk_height_cut
            )

            return np.logical_and(circularity_filter, height_filter)

        @yt.particle_filter(
            requires=["circularity"],
            filtered_type="stars",
        )
        def thick_disk_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            # recreate spheroid filter
            spheroid_filter = (
                data[(pfilter.filtered_type, "circularity")] <= spheroid_circularity_cut
            )

            # recreate thin disk filter
            circularity_filter = (
                data[(pfilter.filtered_type, "circularity")]
                >= thin_disk_circularity_cut
            )
            height_filter = (
                np.abs(data[(pfilter.filtered_type, "height")].to("kpc"))
                <= thin_disk_height_cut
            )
            thin_disk_filter = np.logical_and(circularity_filter, height_filter)

            # in thick disk, if neither in thin disk nor spheroid
            return np.logical_not(np.logical_or(spheroid_filter, thin_disk_filter))

        self.ds.add_particle_filter("thin_disk_stars")
        self.ds.add_particle_filter("thick_disk_stars")
        self.ds.add_particle_filter("spheroid_stars")
        self.ds.add_particle_filter("halo_stars")
        self.ds.add_particle_filter("bulge_stars")
