#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 20:55:44 2023

@author: chris
"""
import os
import pathlib


class Path:
    @staticmethod
    def raw_data(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/data/raw",
        remote_abspath: str = r"/store/clues/HESTIA/RE_SIMS",
    ) -> pathlib.Path:
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
        pathlib.Path
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return abspath.joinpath(relative_path)

    @staticmethod
    def processed_data(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/data/processed",
        remote_abspath: str = r"/z/boettner/gallifrey/data/processed",
    ) -> pathlib.Path:
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
        pathlib.Path
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return abspath.joinpath(relative_path)

    @staticmethod
    def external_data(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/data/external",
        remote_abspath: str = r"/z/boettner/gallifrey/data/external",
    ) -> pathlib.Path:
        """
        Path to external data.

        Parameters
        ----------
        relative_path : str
            Relative file path.
        local_abspath : str, optional
            Local absolute path.
            The default is r"/home/chris/Documents/Projects/gallifrey/data/external".
        remote_abspath : str, optional
            Remote ansolute path.
            The default is r"/z/boettner/gallifrey/data/external".

        Returns
        -------
        pathlib.Path
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return abspath.joinpath(relative_path)

    @staticmethod
    def interim_data(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/data/interim",
        remote_abspath: str = r"/z/boettner/gallifrey/data/interim",
    ) -> pathlib.Path:
        """
        Path to interim data.

        Parameters
        ----------
        relative_path : str
            Relative file path.
        local_abspath : str, optional
            Local absolute path.
            The default is r"/home/chris/Documents/Projects/gallifrey/data/interim".
        remote_abspath : str, optional
            Remote ansolute path.
            The default is r"/z/boettner/gallifrey/data/interim".

        Returns
        -------
        pathlib.Path
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return abspath.joinpath(relative_path)

    @staticmethod
    def figures(
        relative_path: str,
        local_abspath: str = r"/home/chris/Documents/Projects/gallifrey/figures",
        remote_abspath: str = r"/z/boettner/gallifrey/figures",
    ) -> pathlib.Path:
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
        pathlib.Path
            System-dependent absolute path to file.

        """
        abspath = Path.choose_path(local_abspath, remote_abspath)
        return abspath.joinpath(relative_path)

    @staticmethod
    def choose_path(
        local_abspath: str,
        remote_abspath: str,
        local_name: str = "chris",
    ) -> pathlib.Path:
        """
        Choose path depending on system, determined by os environment name.

        Parameters
        ----------
        local_abspath : str
            Local absolute path.
        remote_abspath : str
            Remote ansolute path.
        local_name : str
            Environment name of local machine.

        Returns
        -------
        pathlib.Path
            Correct absolute path for system.

        """
        if os.environ.get("USER") == local_name:  # check for local system
            abspath = local_abspath
        else:
            abspath = remote_abspath
        return pathlib.Path(abspath)
