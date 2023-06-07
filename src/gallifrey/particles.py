#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:02:04 2023

@author: chris
"""
from typing import Optional

import numpy as np
import yt
import unyt as u
from numpy.typing import ArrayLike, NDArray
from yt.frontends.stream.data_structures import StreamParticlesDataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.utilities.math import calculate_rotation_matrix


def rotated_dataset(
    ds: YTDataContainerDataset,
    centre: ArrayLike,
    vector: ArrayLike,
    fields: tuple | list,
    target_vector: ArrayLike = (0, 0, 1),
    length_unit: str = "kpc",
    mass_unit: str = "Msun",
    time_unit: str = "Gyr",
    bounding_box: Optional[tuple] = None,
) -> StreamParticlesDataset:
    """
    Create a yt dataset that contains particle positions, masses and additional fields
    from original ds, rotated in space.

    Parameters
    ----------
    ds : ArepoHDF5Dataset | YTDataContainerDataset
        The yt Dataset for the simulation.
    centre : ArrayLike
        Center of dataset.
    vector : ArrayLike
        Vector that is targeted to be transformed. After transformation, this vector
        will be aligned with target_vector.
    fields : tuple|list
        The fields in question, of form [('PartType','property')] or
        ('PartType','property').
    target_vector : ArrayLike, optional
        The target vector of the transformation. The default is np.array([0,0,1]), which
        is the z-axis.
    length_unit : str, optional
        Length unit of coordinate data. The default is "kpc".
    mass_unit : str, optional
        Mass unit of mass data. The default is "Msun".
    time_unit : str
        Time unit of ages. The default is "Gyr".
    bounding_box : Optional[tuple], optional
        The bounding box (maximum coordinate values) of the data. The default is None,
        which uses minimum/maximum values in dataset.

    Returns
    -------
    StreamParticlesDataset
        Rotated dataset created using yt.

    """

    if type(fields) is tuple:
        if len(fields) == 2:
            field_list = [fields]
        else:
            raise ValueError(
                "If fields is tuple, must be 2-tuple of form (part_type, property)."
            )
    else:
        assert type(fields) is list
        field_list = fields

    # get particle type
    particle_type = field_list[0][0]

    # get original coordinates and masses
    coordinates = ds[(particle_type, "particle_position")].to(length_unit)
    masses = ds[(particle_type, "particle_mass")].to(mass_unit)

    # calculate rotation matrix
    rotation_matrix = calculate_rotation_matrix(vector, target_vector)

    # center and rotate coordimates
    coordinates = coordinates - centre
    rotated_coordinates = np.dot(rotation_matrix, coordinates.T).T

    # calculate bounding box if none is given
    if bounding_box is None:
        bounding_box = (
            (np.amin(rotated_coordinates), np.amax(rotated_coordinates)),
        ) * 3

    # create dataset
    dataset = make_dataset(
        ds,
        field_list,
        length_unit=length_unit,
        mass_unit=mass_unit,
        time_unit=time_unit,
        bounding_box=bounding_box,
        coordinates=rotated_coordinates,
        masses=masses,
    )

    return dataset


def make_dataset(
    ds: YTDataContainerDataset,
    fields: list,
    length_unit: str,
    mass_unit: str,
    time_unit: str,
    bounding_box: tuple,
    coordinates: Optional[NDArray] = None,
    masses: Optional[NDArray] = None,
) -> StreamParticlesDataset:
    """
    Create StreamParticlesDataset from data source.

    Parameters
    ----------
    ds : ArepoHDF5Dataset | YTDataContainerDataset
        The yt Dataset for the simulation.
    fields : list
        The fields in question, of form [('PartType','property')].
    length_unit : str
        Length unit of coordinates.
    mass_unit : str
        Mass unit of masses.
    time_unit : str
        Time unit of ages.
    bounding_box : tuple
        The bounding box (maximum coordinate values) of the data.
    coordinates : Optional[NDArray], optional
        Optional new coordinates. The default is None.
    masses : Optional[NDArray], optional
        Optional new masses. The default is None.

    Returns
    -------
    StreamParticlesDataset
        New dataset created using yt.

    """
    particle_type = fields[0][0]

    # create fields as dict, particle_mass and particle_position are needed internally
    fields += [(particle_type, "particle_mass"), (particle_type, "particle_position")]
    data = {field[1]: ds[field] for field in fields}

    # override coordinates or masses, if required
    if coordinates is not None:
        if (
            len(data["particle_position"]) == len(coordinates)
            and coordinates.shape[1] == 3
        ):
            data["particle_position"] = coordinates
        else:
            raise ValueError("Shape of coordinate array does not match.")

    if masses is not None:
        if len(data["particle_position"]) == len(masses):
            data["particle_masses"] = masses
        else:
            raise ValueError("Shape of mass array does not match.")

    # create dataset
    particle_ds = yt.load_particles(
        data,
        length_unit=length_unit,
        mass_unit=mass_unit,
        time_unit=time_unit,
        bbox=bounding_box,
    )
    
    # create fields with original name
    def create_function(field_name, units):
        def func(field, data):
            return particle_ds.arr(data["io", field_name[1]].value, units)
        return func
    
    fields += [(particle_type, "particle_position_x"),
               (particle_type, "particle_position_y"),
               (particle_type, "particle_position_z"),
               (particle_type, "particle_ones")]
    
    for field in fields:
        current_field = particle_ds.r["io", field[1]]
        if current_field.units == u.dimensionless:
            units = "1" # needed so that units are properly understood
        else:
            units = current_field.units  
            
        particle_ds.add_field(field,
                              function = create_function(field, units),
                              sampling_type="local",
                              units=units)
    return particle_ds