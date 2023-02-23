#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:52:54 2023

@author: chris
"""

import functools
import time
from builtins import BaseException
from types import TracebackType
from typing import Any, Callable, Optional, Type

from gallifrey.utilities.logging import logger

# create logger
logger = logger(__name__)


def timing(decimals: int = 2) -> Callable:
    """Timing function usable as decorator."""

    # outer function to accept argument for number of decimal points for timer
    def decorator(func: Callable) -> Callable:
        # decorator function
        @functools.wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> Any:
            # wrapper for timing
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = round(end_time - start_time, decimals)
            logger.info(
                f"TIMING DECORATOR: Function {func.__name__!r} "
                f"took {run_time} seconds."
            )
            return value

        return wrap

    return decorator


class Timer:
    """
    Context manager for timing a block of code.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        decimals: int = 2,
        print_to_console: bool = True,
    ) -> None:
        """
        Initilize timer.

        Parameters
        ----------
        name : Optional[str], optional
            Optional name of timer block. The default is None.
        decimals : int, optional
            Number of decimals in output. The default is 2.
        print_to_console : bool, optinal
            If True print result, else do not. The default is True.
        """
        self.decimals = decimals
        self.name = name
        self.print_to_console = print_to_console

    def __enter__(self) -> None:
        """Start timer."""
        self.start_time = time.time()

    def __exit__(
        self,
        exception_type: Type[BaseException],
        exception_value: Type[BaseException],
        exception_traceback: TracebackType,
    ) -> None:
        """
        End timer and print result. Arguments are needed by default in case error
        occurs within code block.
        """
        end_time = time.time()
        run_time = round(end_time - self.start_time, self.decimals)
        if self.print_to_console:
            if self.name:
                logger.info(f"Timer: Block {self.name!r} took {run_time} seconds.")
            else:
                logger.info(f"Timer: Block took {run_time} seconds.")
