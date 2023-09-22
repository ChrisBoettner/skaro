#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:52:43 2023

@author: chris
"""
import os

import yt
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset

from gallifrey.data.paths import Path
from gallifrey.utilities.logging import logger

# create Logger
logger = logger(__name__)


def load_snapshot(
    snapshot: int,
    resolution: int,
    sim_id: str = "09_18",
    test_flag: bool = False,
) -> tuple[ArepoHDF5Dataset, str]:
    """
    Load HESTIA snapshot.

    Parameters
    ----------
    snapshot : int
        Snapshot number, should be between 0 and 127.
    resolution : int, optional
        Particle resolution of the simulation, should be 2048, 4096 or 8192.
    sim_id : str, optional
        ID of the concrete simulation run. The default is "09_18".
    test_flag : bool, optional
        If True, load local test snapshot. The default is False.

    Raises
    ------
    ValueError
        Raised when snapshot or resolution has value outside of expected set.
    FileNotFoundError
        Raised when no ValueError occured but the file still could not be
        loaded.

    Returns
    -------
    ArepoHDF5Dataset
        yt dataset object.
    Path
        Path to snapshot directory.

    """
    if test_flag:  # if local system, load the test file
        path = Path().raw_data("snapdir_127/snapshot_127.0.hdf5")
        index_name = "test_snapshot.index5_7.ewah"

    else:
        # check if snapshot is available
        if not ((snapshot >= 0) and (snapshot <= 127)):
            raise ValueError("Snapshot should be between 0 and 127.")
        # add leading zeros to snapshot if necessary
        snapshot_string = f"00{snapshot}"[-3:]

        path = Path().raw_data(rf"{resolution}/GAL_FOR/{sim_id}")
        match resolution:
            case 8192:
                path = path.joinpath(
                    rf"output_2x2.5Mpc/snapdir_{snapshot_string}"
                    rf"/snapshot_{snapshot_string}.0.hdf5",
                )

            case 2048 | 4096:
                path = path.joinpath(
                    rf"output/snapdir_{snapshot_string}"
                    rf"/snapshot_{snapshot_string}.0.hdf5",
                )

            case _:
                raise ValueError("Resolution should be 2048, 4096 or 8192.")

        index_name = f"{sim_id}_{snapshot}_{resolution}.index5_7.ewah"

    # location and name of cache file for indexing created by yt
    index_cache_path = Path().interim_data(f"index_cache/{index_name}")
    # create directory if it doesn't exist already
    if not os.path.exists(os.path.dirname(index_cache_path)):
        os.makedirs(os.path.dirname(index_cache_path))

    try:
        dataset = yt.load(path, index_filename=index_cache_path)
    except FileNotFoundError:
        raise FileNotFoundError("Snapshot not found.")

    return dataset, str(path.parent)


def load_AHF_particles(
    snapshot: int,
    resolution: int,
    sim_id: str = "09_18",
    test_flag: bool = False,
) -> list[str]:
    """
    Load particle file that associates each halo with its particles. In the file
    the first number is usually the ParticleID and the second numver is the PartType.
    If the second number is longer than one digit, it is the halo ID instead and all
    the following particles belong to that halo.

    Parameters
    ----------
    snapshot : int
        Snapshot number, currently only implemented for 127.
    resolution : int, optional
        Particle resolution of the simulation, should be 2048, 4096 or 8192.
    sim_id : str, optional
        ID of the concrete simulation run. The default is "09_18".
    test_flag : bool, optional
        If True, load local test snapshot. The default is False.

    Raises
    ------
    NotImplementedError
        Raised when snapshot is not 127.
    FileNotFoundError
        Raised when no ValueError occured but the file still could not be
        loaded.

    Returns
    -------
    ArepoHDF5Dataset
        yt dataset object.
    Path
        Path to snapshot directory.

    """
    if test_flag:  # if local system, load the test file
        path = Path().raw_data("HESTIA_100Mpc_4096_09_18.127.z0.000.AHF_particles")

    else:
        # check if snapshot is available
        if snapshot not in (127, "127"):
            raise NotImplementedError(
                "load_AHF_particles currently only implemented for snapshot 127. "
                "For other particles, need to adjust the redshift-snapshot conversion."
            )
        # add leading zeros to snapshot if necessary
        snapshot_string = f"00{snapshot}"[-3:]

        path = Path().raw_data(rf"{resolution}/GAL_FOR/{sim_id}")
        match resolution:
            case 8192:
                path = path.joinpath(
                    rf"AHF_output_2x2.5Mpc/HESTIA_100Mpc_{resolution}_{sim_id}."
                    rf"{snapshot_string}.z0.000.AHF_particles"
                )

            case 2048 | 4096:
                path = path.joinpath(
                    rf"AHF_output/HESTIA_100Mpc_{resolution}_{sim_id}."
                    rf"{snapshot_string}.z0.000.AHF_particles"
                )

            case _:
                raise ValueError("Resolution should be 2048, 4096 or 8192.")
    try:
        with open(path) as file:
            ahf_particles = [line.rstrip() for line in file]
    except FileNotFoundError:
        raise FileNotFoundError("AHF particles file not found.")

    return ahf_particles
