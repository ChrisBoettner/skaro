#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:10:14 2023

@author: chris
"""

import numpy as np
from astropy.cosmology import Planck15
from numpy.typing import NDArray
from yt.fields.derived_field import DerivedField
from yt.fields.field_detector import FieldDetector
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.planets import PlanetModel, PlanetOccurenceModel
from gallifrey.stars import ChabrierIMF, StellarModel
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

    def convert_stellar_age(self) -> None:
        """
        Replaces stellar_age field (which contains the formation scale parameter) to
        stellar age = (current time - formation time) in Gyr.

        """
        if "stars" not in dir(self.ds.fields):
            raise AttributeError(
                "'Stars' field does not exist. Needs to be created "
                "to calculate stellar ages using filters.add_stars()."
            )

        logger.info(
            "FIELDS: Overriding ('stars', 'stellar_age') field with " "ages in Gyr."
        )

        self.ds.add_field(
            ("stars", "stellar_age"),
            function=self._stellar_age,
            sampling_type="local",
            units="Gyr",
            force_override=True,
        )

    def add_planets(
        self,
        stellar_model: StellarModel,
        planet_model: PlanetModel,
        imf: ChabrierIMF,
        lower_bound: float = 0.08,
    ) -> None:
        """
        Add (number of) planet and field to star particles. Also adds planet_effect
        field that encodes the dominant effect on the planet number.

        Parameters
        ----------
        stellar_model : StellarModel
            Stellar model that connects mass to other stellar parameter.
        planet_model : PlanetModel
            Planet model that contains relevant planet parameter.
        imf : ChabrierIMF
            Stellar initial mass function of the star particles.
        lower_bound : float, optional
            Lower bound for the integration of the Chabrier IMF. The default is 0.08.

        """

        if "stars" not in dir(self.ds.fields):
            raise AttributeError(
                "'Stars' field does not exist. Needs to be created "
                "to calculate stellar ages using filters.add_stars()."
            )

        if str(self.ds.r["stars", "stellar_age"].units) != "Gyr":
            raise AttributeError(
                "('stars', 'stellar_age') field needs to be given in "
                "'Gyr'. Convert first using convert_stellar_age() "
                "argument."
            )

        planets_occ_model = PlanetOccurenceModel(stellar_model, planet_model, imf)

        def _planets(field: DerivedField, data: FieldDetector) -> NDArray:
            return planets_occ_model.number_of_planets(data, lower_bound=lower_bound)

        self.ds.add_field(
            ("stars", "planets"),
            function=_planets,
            sampling_type="local",
            units="auto",
            dimensions="dimensionless",
        )

        def _planets_effect(field: DerivedField, data: FieldDetector) -> NDArray:
            return planets_occ_model.dominant_effect(data)

        self.ds.add_field(
            ("stars", "planet_effects"),
            function=_planets_effect,
            sampling_type="local",
            units="auto",
            dimensions="dimensionless",
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
            1 / np.array(data["stars", "GFM_StellarFormationTime"])
        ) - 1

        # make redshift space and calculate corresponding cosmic time
        max_redshift = np.amax(formation_redshift)
        redshift_grid = np.linspace(
            data.ds.current_redshift, max_redshift, interpolation_num
        )
        time_grid = Planck15.age(redshift_grid).value

        # calculate formation times from redshift by interpolating redshift grid
        current_time = data.ds.quan(Planck15.age(data.ds.current_redshift).value, "Gyr")
        formation_time = data.ds.arr(
            np.interp(formation_redshift, redshift_grid, time_grid), "Gyr"
        )

        return current_time - formation_time
