#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 20:55:44 2023

@author: chris
"""
import os


class Path:
    def __init__(self) -> None:
        pass

    @staticmethod
    def raw_data(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/data/raw",
        remote_abspath: str = r"/store/clues/HESTIA/RE_SIMS",
    ) -> str:
        """
        Path to raw data.

        Parameters
        ----------
        relative_path : str
            Relative file path.
        local_abspath : str, optional
            Local absolute path.
            The default is r"/home/chris/Documents/Projects/gallifrey/data/raw".
        remote_abspath : str, optional
            Remote ansolute path.
            The default is r"/store/clues/HESTIA/RE_SIMS".

        Returns
        -------
        str
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return os.path.join(abspath, relative_path)

    @staticmethod
    def processed_data(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/data/processed",
        remote_abspath: str = r"/z/boettner/gallifrey/data/processed",
    ) -> str:
        """
        Path to processed data.

        Parameters
        ----------
        relative_path : str
            Relative file path.
        local_abspath : str, optional
            Local absolute path.
            The default is r"/home/chris/Documents/Projects/gallifrey/data/processed".
        remote_abspath : str, optional
            Remote ansolute path.
            The default is r"/z/boettner/gallifrey/data/processed".

        Returns
        -------
        str
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return os.path.join(abspath, relative_path)

    @staticmethod
    def figures(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/figures",
        remote_abspath: str = r"/z/boettner/gallifrey/figures",
    ) -> str:
        """
        Path to figures.

        Parameters
        ----------
        relative_path : str
            Relative file path.
        local_abspath : str, optional
            Local absolute path.
            The default is r"/home/chris/Documents/Projects/gallifrey/figures".
        remote_abspath : str, optional
            Remote ansolute path.
            The default is r"/z/boettner/gallifrey/figures".

        Returns
        -------
        str
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return os.path.join(abspath, relative_path)

    @staticmethod
    def choose_path(local_abspath: str, remote_abspath: str) -> str:
        if os.environ.get("USER") == "chris":  # check for local system
            abspath = local_abspath
        else:
            abspath = remote_abspath
        return abspath