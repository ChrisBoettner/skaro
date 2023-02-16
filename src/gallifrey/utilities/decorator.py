#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:35:33 2023

@author: chris
"""

import functools
from typing import Any, Callable

from gallifrey.utilities.logging import logger

logger = logger()


def debug(func: Callable) -> Callable:
    """Print the function signature and return value."""

    @functools.wraps(func)
    def wrapper_debug(*args: Any, **kwargs: Any) -> Any:
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"DEBUG DECORATOR: Calling {func.__name__}({signature}).")
        value = func(*args, **kwargs)
        logger.debug(f"DEBUG DECORATOR: {func.__name__!r} returned {value!r}.")
        return value

    return wrapper_debug


class CountCalls:
    """Add and increment 'calls' attribute to function."""

    def __init__(self, func: Callable) -> None:
        functools.update_wrapper(self, func)
        self.func = func
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.func(*args, **kwargs)
