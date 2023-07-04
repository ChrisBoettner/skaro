#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:48:05 2023

@author: chris
"""

import pathlib
from typing import Optional

from dace_query.population import Population

from gallifrey.data.paths import Path


def query_DACE_populations(
    population_ids: Optional[list] = None,
    snapshot_ages: Optional[list] = None,
    overwrite: bool = False,
) -> None:
    """
    Query DACE database for NGPPS populations, and save them to raw_data directory.

    Parameters
    ----------
    population_ids : Optional[list], optional
        List of the IDs of the populations to query. The default is None, which defaults
        to the list of Generation III solar-like populations (except 1 embryo run).
    snapshot_ages : Optional[list], optional
        List of the snapshot ages. The default is None, which queries all ages.
    overwrite : bool, optional
        Overwrite file if it already exists. The default is False.

    """
    if population_ids is None:
        population_ids = ["ng96", "ng74", "ng75", "ng76"]

    if snapshot_ages is None:
        snapshot_ages = Population.get_snapshot_ages()

    base_path = Path().external_data("NGPPS")
    for population_id in population_ids:
        # create directory if it does not exist
        population_path = pathlib.Path(base_path, population_id)
        population_path.mkdir(parents=True, exist_ok=True)

        # query snapshots and save dataframes as csv files
        for snapshot_age in snapshot_ages:
            file_path = population_path.joinpath(f"snapshot_{snapshot_age}.csv")

            # skip if file already exists and overwrite=False
            if file_path.exists() and (not overwrite):
                continue

            columns = Population.get_columns(population_id, output_format="pandas")

            snapshot = Population.get_snapshots(
                population_id,
                snapshot_age,
                columns=columns["name"],
                output_format="pandas",
            )

            print(f"Population: {population_id}, age: {snapshot_age}")
            snapshot.to_csv(file_path, index=False)

    print("Done")
