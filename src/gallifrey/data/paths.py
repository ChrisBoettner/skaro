#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 20:55:44 2023

@author: chris
"""
import os


def get_load_path() -> str:
    """
    Load path on local or remote system.

    Returns
    -------
    str
        Load path.

    """
    if os.environ.get("USER") == "chris":  # check for local system
        path = r"/home/chris/Documents/Projects/gallifrey/data/raw/"
    else:
        path = r"/store/clues/HESTIA/RE_SIMS/"

    return path


def get_save_path(mode: str) -> str:
    """
    Save path on local or remote system.

    Parameters
    ----------
    mode : str
        If 'data' save to 'data/processed'.
        If 'figures' save to 'figures'.

    Raises
    ------
    ValueError
        Raised in mode is not 'data' or 'figures'.

    Returns
    -------
    str
        Save path.

    """
    if os.environ.get("USER") == "chris":  # check for local system
        path = r"/home/chris/Documents/Projects/gallifrey/"
    else:
        path = r"/z/boettner/gallifrey/"

    if mode == "data":
        path += "data/processed/"
    elif mode == "figures":
        path += "figures/"
    else:
        raise ValueError("Mode not known. Must be 'data' or 'figures'.")
    return path
