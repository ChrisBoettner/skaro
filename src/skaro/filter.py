#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:51:50 2023

@author: chris
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
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

    def add_stars_by_ID(self, ParticleIDs: ArrayLike) -> None:
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
        def stars_by_ID(
            pfilter: ParticleFilter,
            data: Any,
        ) -> ArrayLike:
            id_filter = np.in1d(
                data[(pfilter.filtered_type, "ParticleIDs")].value,
                ParticleIDs,
                assume_unique=True,
            )
            return id_filter

        self.ds.add_particle_filter("stars_by_ID")

    def add_gas_by_ID(self, ParticleIDs: ArrayLike) -> None:
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
        def gas_by_ID(
            pfilter: ParticleFilter,
            data: Any,
        ) -> ArrayLike:
            id_filter = np.in1d(
                data[(pfilter.filtered_type, "ParticleIDs")].value,
                ParticleIDs,
                assume_unique=True,
            )
            return id_filter

        self.ds.add_particle_filter("gas_by_ID")

    def add_galaxy_components(
        self,
        component_dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Add stars in different components of galaxy (thin disk, thick disk, bulge,
        halo). The decomposition is performed by the mordor code (Zana2022).
        For our purposes pseudo-bulge stars are classified as bulge stars.
        Also adds field 'galaxy_stars' with all star particles.

        Parameters
        ----------
        compotent_dataframe : pd.DataFrame
            Dataframe with galaxy components, must contain columns 'Component' and
            'ParticleIDs'. Usually created using
            skaro.decomposition.mordor.galaxy_components
        **kwargs: Any
            Further arguments passed to create_component_mask.

        """

        @yt.particle_filter(requires=["ParticleIDs"], filtered_type="stars")
        def unbound_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            return _create_component_mask(
                component_dataframe,
                data["stars", "ParticleIDs"].astype(int).value,
                component="unbound",
                **kwargs,
            )

        @yt.particle_filter(requires=["ParticleIDs"], filtered_type="stars")
        def bulge_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            return _create_component_mask(
                component_dataframe,
                data["stars", "ParticleIDs"].astype(int).value,
                component="bulge",
                **kwargs,
            )

        @yt.particle_filter(requires=["ParticleIDs"], filtered_type="stars")
        def thin_disk_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            return _create_component_mask(
                component_dataframe,
                data["stars", "ParticleIDs"].astype(int).value,
                component="thin_disk",
                **kwargs,
            )

        @yt.particle_filter(requires=["ParticleIDs"], filtered_type="stars")
        def thick_disk_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            return _create_component_mask(
                component_dataframe,
                data["stars", "ParticleIDs"].astype(int).value,
                component="thick_disk",
                **kwargs,
            )

        @yt.particle_filter(requires=["ParticleIDs"], filtered_type="stars")
        def halo_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            return _create_component_mask(
                component_dataframe,
                data["stars", "ParticleIDs"].astype(int).value,
                component="halo",
                **kwargs,
            )

        @yt.particle_filter(requires=["ParticleIDs"], filtered_type="stars")
        def galaxy_stars(pfilter: ParticleFilter, data: Any) -> ArrayLike:
            return _create_component_mask(
                component_dataframe,
                data["stars", "ParticleIDs"].astype(int).value,
                component="all",
                **kwargs,
            )

        self.ds.add_particle_filter("unbound_stars")
        self.ds.add_particle_filter("bulge_stars")
        self.ds.add_particle_filter("thin_disk_stars")
        self.ds.add_particle_filter("thick_disk_stars")
        self.ds.add_particle_filter("halo_stars")
        self.ds.add_particle_filter("galaxy_stars")


def _create_component_mask(
    compotent_dataframe: pd.DataFrame,
    star_particle_IDs: np.ndarray,
    component: str,
    component_dict: Optional[dict[str, list]] = None,
) -> np.ndarray:
    """
    Creates mask on star_particle_IDs categorising if they belong to a given
    galaxy component.
    By default, pseudo-bulge stars are classified as bulge stars.

    The mordor classification is
        0 - unbound/excluded
        1 - thin/cold disc
        2 - thick/warm disc
        3 - pseudo-bulge
        4 - bulge
        5 - stellar halo

    Parameters
    ----------
    compotent_dataframe : pd.DataFrame
        Dataframe with components, must contain columns 'Component' and 'ParticleIDs'.
        Usually created using skaro.decomposition.mordor.galaxy_components
    star_particle_IDs : np.ndarray
        List of all star particles IDs in yt selection.
    component : str
        Name of component to filter for.
    component_dict : Optional[dict[str, list]], optional
        Mapping between component name and integer used by mordor. The default is None,
        which creates the default mapping.

    Returns
    -------
    np.ndarray
        The mask filtering out the galaxy components.

    """
    # default component mapping, puts pseudo-bulge into bulge category
    if component_dict is None:
        component_dict = {
            "unbound": [0],
            "thin_disk": [1],
            "thick_disk": [2],
            "bulge": [3, 4],
            "halo": [5],
            "all": [0, 1, 2, 3, 4, 5],
        }

    if component not in component_dict.keys():
        raise ValueError("component not found in component_dict.")

    # filter out Particle IDs for component
    component_ids = compotent_dataframe["ParticleIDs"][
        compotent_dataframe["Component"].isin(component_dict[component])
    ].to_numpy()

    # return mask
    return np.isin(star_particle_IDs, component_ids, assume_unique=True)
