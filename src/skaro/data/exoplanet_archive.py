#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:06:53 2023

@author: chris
"""
import requests

from skaro.data.paths import Path


def query_exoplanet_data(query: str, file_name: str) -> None:
    """
    Download exoplanet data from NASA Exoplanet Archive based on the provided query and
    save to file.

    Parameters
    ----------
    query : str
        SQL query.
    file_name : str
        Name of savefile.

    """
    # API endpoint
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"

    # Full URL, return in csv format
    full_url = url + "query=" + getattr(requests.utils, "quote")(query) + "&format=csv"

    # Make the GET request
    response = requests.get(full_url)

    # Check that the request was successful
    if response.status_code == 200:
        # Save the result to a csv file
        with open(Path().external_data(file_name), "w") as f:
            f.write(response.text)
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == "__main__":
    # Define the SQL for orital distance -- stellar metallicity plot
    query = """
    select
        pl_name,
        pl_refname,
        pl_controv_flag,
        pl_orbsmax,
        pl_orbsmaxerr1,
        pl_orbsmaxerr2,
        pl_orbsmaxlim,
        st_met,
        st_meterr1,
        st_meterr2,
        st_metlim,
        st_metratio
    from
        ps
    where
        pl_controv_flag = 0
        and pl_orbsmax is not null
        and st_met is not null
        and st_metratio like '%Fe%'
    """

    # Call the function to download exoplanet data
    query_exoplanet_data(query, "observed_exoplanet_data.csv")
