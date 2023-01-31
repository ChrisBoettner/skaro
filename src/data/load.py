#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:52:43 2023

@author: chris
"""

import os

import yt
from yt.frontends.arepo.data_structures import ArepoHDF5Dataset


def load_snapshot(
    snapshot: int,
    resolution: int = 8192,
    run: str = "09_18",
) -> ArepoHDF5Dataset:
    """


    Parameters
    ----------
    snapshot : int
        Snapshot number, should be between 0 and 127.
    resolution : int, optional
        Particle resolution of the simulation, should be 2048, 4096 or 8192.
        The default is 8192.
    run : str, optional
        Name of the concrete simulation run. The default is "09_18".

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
        path = r"data/raw/snapdir_002/snapshot_002.0.hdf5"

    else:

        # check if snapshot is available
        if not ((snapshot >= 0) and (snapshot <= 127)):
            raise ValueError("Snapshot should be between 0 and 127.")
        # add leading zeros to snapshot if necessary
        snapshot_string = f"00{snapshot}"[-3:]

        path = rf"/store/clues/HESTIA/RE_SIMS/{resolution}/GAL_FOR/{run}"
        match resolution:
            case 8192:
                path += (
                    rf"/output_2x2.5Mpc/snapdir_{snapshot_string}"
                    rf"/snapshot_{snapshot_string}.0.hdf5"
                )

            case 2048 | 4096:
                path += (
                    rf"/output/snapdir_{snapshot_string}"
                    rf"/snapshot_{snapshot_string}.0.hdf5"
                )

            case _:
                raise ValueError("Resolution should be 2048, 4096 or 8192.")

    try:
        dataset = yt.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("Snapshot not found.")

    return dataset
