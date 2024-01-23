#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:44:05 2023

@author: chris
"""

import logging

# set basic logging config, and create convenience shortcut
logging.basicConfig(level=logging.INFO, format="\n      %(message)s\n")
logger = logging.getLogger

# create default logger, parent to all other logger in module
defaultlogger = logging.getLogger("skaro")
