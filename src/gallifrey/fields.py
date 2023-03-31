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

from gallifrey.utilities.logging import logger

from gallifrey.stars import StellarModel
from gallifrey.planets import PlanetModel

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
        .
        """
        if "stars" not in dir(self.ds.fields):
            raise AttributeError("'Stars' field does not exist. Needs to be created "
                                 "to calculate stellar ages using filters.add_stars().")
        
        logger.info("FIELDS: Overriding ('stars', 'stellar_age') field with "
                    "ages in Gyr.")
        
        self.ds.add_field(
            ("stars", "stellar_age"),
            function=self._stellar_age,
            sampling_type="local",
            units="Gyr",
            force_override=True,
        )
    
    @staticmethod
    def _stellar_age(
        field: DerivedField,
        data: FieldDetector,
        interpolation_num: int = 500,
    ) -> None:
        '''
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

        '''   
        # get current simulation time, and formation redshifts of star particles from
        # scale factor
        current_time = data.ds.current_time.to("Gyr")
        formation_redshift = (
            (1 / np.array(data['stars', 'GFM_StellarFormationTime'])) - 1
        )
        
        # make redshift space and calculate corresponding cosmic time      
        max_redshift = np.amax(formation_redshift)
        redshift_grid = np.linspace(
            data.ds.current_redshift, max_redshift, interpolation_num
        )
        time_grid = Planck15.age(redshift_grid).value

        # calculate formation times from redshift by interpolating redshift grid
        current_time   = data.ds.quan(Planck15.age(data.ds.current_redshift).value, 
                                      "Gyr")
        formation_time = data.ds.arr(
            np.interp(formation_redshift, redshift_grid, time_grid),
            "Gyr",
        )

        return current_time - formation_time
    
    def add_planets(self):
        if "stars" not in dir(self.ds.fields):
            raise AttributeError("'Stars' field does not exist. Needs to be created "
                                 "to calculate stellar ages using filters.add_stars().")
            
        if str(self.ds.r['stars','stellar_age'].units) != "Gyr":
            raise AttributeError("('stars', 'stellar_age') field needs to be given in "
                                 "'Gyr'. Convert first using convert_stellar_age() "
                                 "argument.")            
        
        
        self.ds.add_field(
            ("stars", "planets"),
            function=self._planets,
            sampling_type="local",
            units="dimensionless",
        )
    
    # @staticmethod
    # def _planets(
    #     field: DerivedField,
    #     data: FieldDetector,
    #     stellar_model : StellarModel,
    #     planet_model: PlanetModel,
    #     lower_stellar_mass:float = 0.08,
    # ) -> None:
        
    #     stellar_ages = data["stars", "stellar_age"].value # stellar ages in Gyr
    #     planets = np.empty_like(stellar_ages)
        
    #     planet_formation_time_mask = stellar_ages < planet_model.planet_formation_time
    #     inv_mask = ~planet_formation_time_mask
        
    #     # no planets if stellar age < planet_formation_time_mask
    #     planets[planet_formation_time_mask] = 0
        
    #     # calculate upper stellar mass limit for planets
    #     # =============================================================================
        
    #     # max mass from lifetime
    #     max_mass_from_lifetime = stellar_model.mass_from_lifetime(stellar_ages)
        
    #     # max mass from metallicity
    #     fe_fraction = data['stars','Fe_fraction'].value/data['stars','H_fraction'].value
        
        
    #     # =============================================================================
        
        
        




# make _stellar_age seperate function
# add planet field

# planets pseudocode:

# check particle age
# if particle age < PlanetModel.planet_formation_time:
#     return 0
# else:
#     calculate maximum mass from particle age -> StellarModel.mass_from_lifetime()
#     calculate maximum mass from occurence rate (PlanetModel.cutoff_temperature) -> StellarModel.mass_from_temperature
#     calculate maximum mass from metallicity:
#         calculate maximum planet formation distance -> PlanetModel.critical_formation_distance()
#         calculate maximum mass from this distance -> StellarModel.inner_HZ_inverse()


#     get minimum of these three masses
#     integrate IMF up to this value -> Chabrier.number_of_stars()
#     multiply number of stars by planet occurence rate -> PlanetModel.occurence_rate
