#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:37:46 2023

@author: chris
"""

import pathlib
from typing import Any, Optional

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from unyt.array import unyt_array, unyt_quantity
from yt.data_objects.data_containers import YTDataContainer
from yt.data_objects.selection_objects.disk import YTDisk
from yt.data_objects.selection_objects.region import YTRegion
from yt.data_objects.selection_objects.spheroids import YTSphere
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset
from yt.frontends.ytdata.data_structures import YTDataContainerDataset

from gallifrey.data.load import load_AHF_particles
from gallifrey.data.paths import Path
from gallifrey.filter import Filter
from gallifrey.utilities.math import calculate_pca


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
        self.ID: int

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

    def normal_vector(
        self,
        particle_type: str,
        data: Optional[YTDataContainerDataset | YTDataContainer] = None,
    ) -> np.ndarray:
        """
        Calculate normal vector to structure of particles using PCA. Assumes a disk-like
        structure and return Principle Component with smallest explained variance.

        Parameters
        ----------
        particle_type : str
            Type of particles for which normal vector should be calculated.
        data : Optional[YTDataContainerDataset | YTDataContainer], optional
            The data source. If None, calculate use sphere around origin with a radius
            of 10% the virial radius. The default is None.

        Returns
        -------
        normal_vector : np.array
            Normal vector (normalised to 1).

        """

        if data is None:
            data = self.sphere(radius=0.1 * self.virial_radius())

        normal_vector = calculate_pca(data[particle_type, "Coordinates"]).components_[
            -1
        ]

        return normal_vector

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

    @staticmethod
    def extract_particle_ids(
        halo_ID: int,
        snapshot: int | str,
        resolution: int,
        sim_id: str = "09_18",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Return dataframe with all particle IDs associated with this halo for a given
        halo ID and snapshot properties.
        Parameters
        ----------
        halo_ID : int
            The halo ID associated with the particles.
        snapshot : int | str
            Snapshot number, should be between 0 and 127.
        resolution : int, optional
            Particle resolution of the simulation, should be 2048, 4096 or 8192.
        sim_id : str, optional
            ID of the concrete simulation run. The default is "09_18".
        **kwargs : Any
            Further arguments passed to load_AHF_particles.

        Raises
        ------
        AttributeError
            Raised if the halo ID is not found in the AHF output.

        Returns
        -------
        pd.DataFrame
            The particle IDs and corresponding particle types associated with the halo.

        """
        # Load AHF data
        ahf_particles = load_AHF_particles(snapshot, resolution, sim_id, **kwargs)

        # Check where halo data begins and ends
        start_ID = str(halo_ID)
        end_ID = str(halo_ID + 1)
        start_idx = end_idx = None
        for idx, line in enumerate(ahf_particles):
            if start_ID in line:
                start_idx = idx
            elif end_ID in line:
                end_idx = idx
                break

        if start_idx is None:
            raise AttributeError("Halo ID not in AHF particle file.")

        if end_idx is None:
            end_idx = -1  # assume its the last halo and all data up to end is used

        # convert and save in DataFrame format
        halo_particles = np.array(
            [line.split("\t") for line in ahf_particles[start_idx + 1 : end_idx]]
        )
        halo_particle_df = pd.DataFrame(
            halo_particles, columns=("ParticleIDs", "PartType")
        ).astype(int)

        return halo_particle_df.sort_values(by=["PartType", "ParticleIDs"])


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
        test_flag: bool = False,
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
        path : Optional[str], optional
            Absolute path to file.
        test_flag: bool, optional
            If true, assume halo belongs to local test. Passes corresponding information
            to particle_IDs to identify correct file path. The default is False.
        """
        self.halo_id = halo_id
        self.resolution = resolution
        self.sim_id = sim_id
        self.snapshot = f"00{snapshot}"[-3:]
        self.test_flag = test_flag

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

    def particle_IDs(
        self,
        halo_ID: Optional[int] = None,
        id_path: Optional[str] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Return dataframe with all particle IDs associated with this halo (based on
        halo ID) using the AHF output.

        Parameters
        ----------
        halo_ID : Optional[int], optional
            The halo ID associated with the particles. The default is None, which
            defaults to the ID attribute of the Halo object.
        id_path : Optional[str], optional
            Path to ID file, in order to load/save file. The default is None,
            which looks in the data/processed directory.
        save: bool, optional
            Decide if ID file should be saved, if it was just created.
        force_calculation: bool, optional

        Raises
        ------
        AttributeError
            Raised if the halo ID is not found in the AHF output.

        Returns
        -------
        pd.DataFrame
            The particle IDs and corresponding particle types associated with the halo.

        """
        if halo_ID is None:
            if not hasattr(self, "ID"):
                raise AttributeError(
                    "Halo needs ID attribute to load corresponding AHF particles."
                )
            halo_ID = self.ID

        # choose path
        if id_path is None:
            path = Path().processed_data(
                f"{self.resolution}/{self.sim_id}/"
                f"snapshot_{self.snapshot}_{self.halo_id}_particle_IDs.csv"
            )
        else:
            path = id_path

        # load or create particle IDs
        try:
            particle_id_dataframe = pd.read_csv(path)

        except FileNotFoundError:
            particle_id_dataframe = self.extract_particle_ids(
                halo_ID,
                self.snapshot,
                self.resolution,
                self.sim_id,
                test_flag=self.test_flag,
            )
            if save:
                particle_id_dataframe.to_csv(path, index=None)

        return particle_id_dataframe


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
