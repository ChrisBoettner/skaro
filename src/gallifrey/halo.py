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
from yt.data_objects.selection_objects.disk import YTDisk
from yt.data_objects.selection_objects.region import YTRegion
from yt.data_objects.selection_objects.spheroids import YTSphere
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.data.paths import Path
from gallifrey.filter import Filter


class HaloContainer:
    """
    General Halo object.
    """

    def __init__(
        self,
        property_dict: dict[str, float],
        ds: ArepoHDF5Dataset | YTDataContainerDataset,
    ) -> None:
        """
        For HaloContainer object, the general properties are supplied directly
        via a dictonary.

        Parameters
        ----------
        property_dict : dict
            Halo properties as dictonary. Will be set to attributes by
            self.key = value binding.
        ds : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        """
        self.X: float | unyt_quantity
        self.Y: float | unyt_quantity
        self.Z: float | unyt_quantity
        self.M: float | unyt_quantity

        for key, value in property_dict.items():
            setattr(self, key, value)

        self.ds = ds
        self.filter = Filter(self.ds)

    def centre(self) -> unyt_array:
        """
        Returns center of halo as unyt array.

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

        return self.ds.arr(
            np.array([self.X, self.Y, self.Z]) / getattr(self.ds, "hubble_constant"),
            "kpc",
        )

    def virial_radius(
        self,
        overdensity_constant: float = 200,
    ) -> unyt_quantity:
        """
        Calculate virial radius of halo from mass.

        Parameters
        ----------
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

        critical_density = self.ds.critical_density.to("Msun/kpc**3")
        scale = self.ds.quan(self.M, "Msun") / (overdensity_constant * critical_density)

        return (0.75 / np.pi * scale) ** (1 / 3)

    def sphere(
        self,
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
            centre = self.centre()

        if radius is None:
            radius = self.virial_radius(overdensity_constant)

        return getattr(self.ds, "sphere")(centre, radius, **kwargs)

    def box(
        self,
        centre: Optional[unyt_array] = None,
        radius: Optional[unyt_quantity] = None,
        overdensity_constant: float = 200,
        **kwargs: Any,
    ) -> YTRegion:
        """
        Return YTRegion Data Object. If centre and radius are not given, uses
        centre() and virial_radius() methods.

        Parameters
        ----------
        centre : Optional[unyt_array], optional
            Centre of sphere in kpc. The default is None.
        radius : Optional[unyt_quantity], optional
            Box side length is 2 * radius. The default is None.
        overdensity_constant : float, optional
            Overdensity threshold if virial_radius() is used for radius. The
            default is 200.
        **kwargs : Any
            Kwargs passed to YTRegion.

        Returns
        -------
        YTRegion
            YTRegion Data Object.

        """
        if centre is None:
            centre = self.centre()

        if radius is None:
            radius = self.virial_radius(overdensity_constant)
        return getattr(self.ds, "box")(centre - radius, centre + radius, **kwargs)

    def disk(
        self,
        centre: Optional[unyt_array] = None,
        normal: Optional[unyt_array] = None,
        radius: Optional[unyt_quantity] = None,
        height: Optional[unyt_quantity] = None,
        overdensity_constant: float = 200,
        **kwargs: Any,
    ) -> YTDisk:
        """
        Return YTRegion Data Object. If centre and radius are not given, uses
        centre() and virial_radius() methods. If normal is not given, calculates
        angular momentum vector within sphere of virial radius and uses that. Height
        defaults to 0.5 kpc.

        Parameters
        ----------
        centre : Optional[unyt_array], optional
            Centre of sphere in kpc. The default is None.
        normal : Optional[unyt_array], optional
            Normal vector of disk. The default is None.
        radius : Optional[unyt_quantity], optional
            Radius of disk. The default is None.
        height : Optional[unyt_quantity], optional
            Height of disk. The default is None.
        overdensity_constant : float, optional
            Overdensity threshold if virial_radius() is used for radius. The
            default is 200.
        **kwargs : Any
            Kwargs passed to YTDisk.

        Returns
        -------
        YTDisk
            YTDisk Data Object.

        """
        if centre is None:
            centre = self.centre()

        if normal is None:
            if hasattr(self.sphere().quantities, "angular_momentum_vector"):
                normal = getattr(self.sphere().quantities, "angular_momentum_vector")()
            else:
                raise AttributeError(
                    "'sphere().quantities' has no attribute 'angular_momentum_vector'"
                )

        if radius is None:
            radius = self.virial_radius(overdensity_constant)

        if height is None:
            height = self.ds.quan(0.5, "kpc")

        return getattr(self.ds, "disk")(centre, normal, radius, height, **kwargs)


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
        ds: ArepoHDF5Dataset | YTDataContainerDataset,
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
        ds : ArepoHDF5Dataset | YTDataContainerDataset
            The yt Dataset for the simulation.
        sim_id : str
            ID of simulation run.
        snapshot : TYPE, optional
            Index of snapshot. The default is 127.
        path : Optional[str]
            Absolute path to file.
        """
        self.halo_id = halo_id
        self.resolution = resolution
        self.sim_id = sim_id
        self.snapshot = f"00{snapshot}"[-3:]

        self.ds = ds
        self.filter = Filter(self.ds)

        self.set_path(path)
        self.load()

    def set_path(self, path: Optional[str]) -> None:
        """
        Path to YAML source file.

        Parameters
        ----------
        path : Optional[str]
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
        if isinstance(self.path, str | pathlib.Path):
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
        self.key = value
        if key in self.file[self.halo_id]:
            self.file[self.halo_id][key] = value
        else:
            self.file[self.halo_id].insert(position, key, value, comment)

        if save:
            if isinstance(self.path, str | pathlib.Path):
                with open(self.path, "w") as file:
                    YAML().dump(self.file, file)
                self.load()  # reload file
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
