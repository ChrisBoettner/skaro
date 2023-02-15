#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:37:46 2023

@author: chris
"""

import pathlib
from typing import Any, Optional

import numpy as np
from ruamel.yaml import YAML
from unyt.array import unyt_array, unyt_quantity
from yt.data_objects.selection_objects.spheroids import YTSphere
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.data.paths import Path


class HaloContainer:
    """General Halo object."""

    def __init__(self, property_dict: dict[str, float]) -> None:
        """
        For HaloContainer object, the general properties are supplied directly
        via a dictonary.

        Parameters
        ----------
        property_dict : dict
            Halo properties as dictonary. Will be set to attributes by
            self.key = value binding.
        """
        self.X: float | unyt_quantity
        self.Y: float | unyt_quantity
        self.Z: float | unyt_quantity
        self.M: float | unyt_quantity

        for key, value in property_dict.items():
            setattr(self, key, value)

    def centre(self, ds: ArepoHDF5Dataset | YTDataContainerDataset) -> unyt_array:
        """
        Returns center of halo as unyt array.

        Parameters
        ----------
        ds : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        Raises
        ------
        AttributeError
            Raised if coordinated X,Y and Z are not all defined.

        Returns
        -------
        unyt_array
            Centre of halo in kpc.

        """
        if not all([hasattr(self, attr) for attr in ["X", "Y", "Z"]]):
            raise AttributeError("X, Y and Z must be defined for centre.")

        return ds.arr(
            np.array([self.X, self.Y, self.Z]) * getattr(ds, "hubble_constant"), "kpc"
        )

    def virial_radius(
        self,
        ds: ArepoHDF5Dataset | YTDataContainerDataset,
        overdensity_constant: float = 200,
    ) -> unyt_quantity:
        """
        Calculate virial radius of halo from mass.

        Parameters
        ----------
        ds : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        overdensity_constant : float, optional
            Overdensity threshold for halo. The default is 200.

        Raises
        ------
        AttributeError
            Raised if coordinated M is not all defined.

        Returns
        -------
        TYPE
            Virial radius in kpc.

        """
        if not hasattr(self, "M"):
            raise AttributeError("M must be defined for virial radius.")

        critical_density = ds.critical_density.to("Msun/kpc**3")
        scale = ds.quan(self.M, "Msun") / (overdensity_constant * critical_density)

        return (0.75 / np.pi * scale) ** (1 / 3)

    def sphere(
        self,
        ds: ArepoHDF5Dataset | YTDataContainerDataset,
        centre: Optional[unyt_array] = None,
        radius: Optional[unyt_quantity] = None,
        overdensity_constant: float = 200,
        **kwargs: Any,
    ) -> YTSphere:
        """
        Return YTSphere Data Object. If centre and radius are not given, uses
        centre() and virial_radius() methods.

        Parameters
        ----------
        ds : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        centre : Optional[unyt_array], optional
            Centre of sphere in kpc. The default is None.
        radius : Optional[unyt_quantity], optional
            Radius of sphere in kpc. The default is None.
        overdensity_constant : float, optional
            Overdensity threshold if virial_radius() is used for radius. The
            default is 200.
        **kwargs : Any
            Kwargs passed to YTSphere.

        Returns
        -------
        YTSphere
            YTSphere Data Object.

        """
        if centre is None:
            centre = self.centre(ds)

        if radius is None:
            radius = self.virial_radius(ds, overdensity_constant)

        return getattr(ds, "sphere")(centre, radius, **kwargs)


class Halo(HaloContainer):
    """
    Halo object derived from HaloContainer.
    For this class, it is assumed the halo properties are saved in
    /data/processed/ as a YAML file.
    """

    def __init__(
        self,
        halo_id: str,
        resolution: int,
        sim_id: str = "09_18",
        snapshot: int = 127,
        path: Optional[str] = None,
    ) -> None:
        """
        Add details to attributes, construct path and load halo information

        Parameters
        ----------
        halo_id : str
            ID or name of halo in question.
        resolution : int
            Resolution of simulation used.
        sim_id : str
            ID of simulation run.
        snapshot : TYPE, optional
            Index of snapshot. The default is 127.
        """
        self.halo_id = halo_id
        self.resolution = resolution
        self.sim_id = sim_id
        self.snapshot = f"00{snapshot}"[-3:]

        self.set_path(path)
        self.load()

    def set_path(self, path: Optional[str]) -> None:
        """
        Path to YAML source file.

        Parameters
        ----------
        path : str
            Absolute path to file.
        """
        self.path = path

    def load(self) -> None:
        """
        Read values from YAML file and write to attributes.


        Raises
        ------
        AttributeError
            Raised if path is not a string.
        """
        if isinstance(self.path, str):
            self.file = YAML().load(pathlib.Path(self.path))
            data = dict(self.file[self.halo_id])
            for key, value in data.items():
                setattr(self, key, value)
        else:
            raise AttributeError("Path must be str with absolute path to file.")

    def insert(
        self,
        key: str,
        value: float,
        comment: str,
        position: int = 0,
        save: bool = True,
    ) -> None:
        """
        Insert new quantity into halo data.

        Parameters
        ----------
        key : str
            Key of the quantity.
        value : float
            Value of the quantity.
        comment : str
            Comment, should at least include the unit.
        position : int, optional
            Position where new information should be included in file. The default is 0.
        save : bool, optional
            If true, new quantity is directly saved to file. The default is True.

        Raises
        ------
        AttributeError
            Raised if path is not a string.

        """
        self.file[self.halo_id].insert(position, key, value, comment)

        if save:
            if isinstance(self.path, str):
                with open(self.path, "w") as file:
                    YAML().dump(self.file, file)
        else:
            raise AttributeError("Path must be str with absolute path to file.")


class MainHalo(Halo):
    """Halo object specifically for main halos (M31 and MW)."""

    def set_path(self, path: Optional[str]) -> None:
        """
        If no path is given, use the default path to the main halo files.

        Parameters
        ----------
        path : str
            Absolute path to file.
        """
        if path is None:
            self.path = Path().processed_data(
                f"{self.resolution}/{self.sim_id}/"
                f"snapshot_{self.snapshot}_main_halos.yaml"
            )
        else:
            self.path = path
