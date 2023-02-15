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


def load_snapshot(
    snapshot: int,
    resolution: int,
    sim_id: str = "09_18",
) -> ArepoHDF5Dataset:
    """


    Parameters
    ----------
    snapshot : int
        Snapshot number, should be between 0 and 127.
    resolution : int, optional
        Particle resolution of the simulation, should be 2048, 4096 or 8192.
    sim_id : str, optional
        ID of the concrete simulation run. The default is "09_18".

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

    """
    if os.environ.get("USER") == "chris":  # if local system, load the test file
        print("\n      DETECTED LOCAL MACHINE: Test snapshot loaded.\n")
        path = Path().raw_data(r"snapdir_127/snapshot_127.0.hdf5")

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

    try:
        dataset = yt.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("Snapshot not found.")

    return dataset
