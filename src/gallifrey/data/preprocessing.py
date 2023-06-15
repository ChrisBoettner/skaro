#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 20:23:23 2023

@author: chris
"""

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any

from ruamel.yaml import YAML, CommentedMap

from gallifrey.data.paths import Path
from gallifrey.utilities.logging import logger

# create Logger
logger = logger(__name__)


class HaloPreprocessingAbstract(ABC):
    """Abstract Class for preprocessing information about halos."""

    def __init__(
        self,
        resolution: int,
        snapshot: int = 127,
    ) -> None:
        """
        Initialize object with information about simulation.

        Parameters
        ----------
        resolution : int
            Resolution of simulation used.
            Likely in [512, 1024, 2048, 4096, 8192, 16384].
        snapshot : int, optional
            Index of snapshot. The default is 127.
        """
        self.resolution = resolution
        self.snapshot = f"00{snapshot}"[-3:]  # always 3 digits

    def create(self, overwrite: bool = False) -> None:
        """
        Load, process and save halo information.
        Parameters
        ----------
        overwrite : bool, optional
            Passed to save_yaml() method. If false, raises an error if the file already
            exists. The default is False.
        """
        halo_information = self.load_source()
        processed_halo_information = self.fill_yaml(halo_information)
        self.save_yaml(processed_halo_information, overwrite=overwrite)

    @abstractmethod
    def load_source(self) -> list[list] | list[Any]:
        """Load and preprocess halo information."""
        pass

    @abstractmethod
    def fill_yaml(
        self,
        halo_information: Any,
    ) -> CommentedMap | dict[str, CommentedMap]:
        """
        Turn preprocessed halo information into dict with CommentedMap YAML
        object. Keys of dict should be ID of simulation run values should be
        CommentedMap object that can be coverted into yaml. If only one run is
        considered, can also return CommentedMap object directly.
        """
        pass

    @abstractmethod
    def save_yaml(
        self,
        processed_halo_information: CommentedMap | dict[str, CommentedMap],
        overwrite: bool,
    ) -> None:
        """Save halo information to yaml file."""
        pass

    @staticmethod
    def commented_yaml_string(
        ID: str,
        X: float,
        Y: float,
        Z: float,
        M: float,
    ) -> str:
        """
        Template string that data is filled in to be turned into yaml.

        Parameters
        ----------
        ID : str
            ID to identify halo.
        X : float
            X coordinate of halo in kpc.
        Y : float
            Y coordinate of halo in kpc.
        Z : float
            Z coordinate of halo in kpc.
        M : float
            Mass in solar masses.

        Returns
        -------
        str
            Template string.

        """
        yaml_string = f"""
        {ID}: # ID
            X: {X} # in kpc
            Y: {Y} # in kpc
            Z: {Z} # in kpc
            M: {M} # in solar masses
        """
        return yaml_string


class MainHaloPreprocessing(HaloPreprocessingAbstract):
    """Halo preprocessing for MW and M31 based on Noam's precreated files."""

    def load_source(self) -> list[list]:
        """Load Noam's main halo information files."""

        file_path = Path().raw_data(
            rf"LGs_{self.resolution}_GAL_FOR.txt",
            remote_abspath=r"/z/nil/codes/HESTIA/FIND_LG",
        )

        # read file line by line and preprocess
        halo_information = []
        with open(file_path) as file:
            for i, line in enumerate(file):
                if i == 0:
                    pass  # header line
                else:
                    split_line = line.split()  # split at whitespace
                    split_line = split_line[:-1]  # last entry is useless
                    # re-insert simulation ID in correct format
                    sim_id = f"0{split_line[0]}"[-2:] + f"_{split_line[1]}"
                    split_line = [sim_id] + split_line[2:]
                    halo_information.append(split_line)
        return halo_information

    def fill_yaml(
        self,
        halo_information: list[list],
    ) -> dict[str, CommentedMap]:
        """
        Fill yaml CommentedMap for MW and M31 (Mass and
        x,y,z coordinate).

        Parameters
        ----------
        halo_information : list[list]
            Halo Information created from load_source.

        Returns
        -------
        dict[str, CommentedMap]
            Dict with Simulation ID as keys and CommentedMap object as values.

        """
        processed_halo_information = {}

        for simulation_run in halo_information:
            sim_id = simulation_run[0]

            # fill MW information
            information_string = self.commented_yaml_string(
                "MW",
                simulation_run[29],
                simulation_run[30],
                simulation_run[31],
                simulation_run[5],
            )
            # fill M31 information
            information_string += self.commented_yaml_string(
                "M31",
                simulation_run[26],
                simulation_run[27],
                simulation_run[28],
                simulation_run[4],
            )

            # turn to yaml
            yaml_object = YAML().load(information_string)
            processed_halo_information[sim_id] = yaml_object

        return processed_halo_information

    def save_yaml(
        self,
        processed_halo_information: dict[str, CommentedMap],
        overwrite: bool = False,
    ) -> None:
        """
        Save halo information as yaml.

        Parameters
        ----------
        processed_halo_information : dict[str, CommentedMap]
            Dictonary of halo information per run, created by fill_yaml.
        overwrite : bool, optional
            If false, raises an error if the file already exists. The default is False.

        Raises
        ------
        FileExistsError
            Raised if file already exists and overwrite=False.

        """
        # iterate over all runs
        for sim_id, yaml_object in processed_halo_information.items():
            path = Path().processed_data(f"{self.resolution}/{sim_id}")

            # create dir if it does not exist
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            file = path.joinpath(f"snapshot_{self.snapshot}_main_halos.yaml")

            if os.path.isfile(file) and (not overwrite):
                raise FileExistsError("File already exists, needs overwrite=True.")

            with open(file, "w") as savefile:
                YAML().dump(yaml_object, savefile)
