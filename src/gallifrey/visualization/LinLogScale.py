#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:45:48 2023

@author: chris
"""
from typing import Any, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.scale import LogScale, register_scale
from matplotlib.ticker import (
    Locator,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
from matplotlib.transforms import Transform

# ###################### TRANSFORM #####################################################


class LinLogTransform(Transform):
    """
    Symmetrical linear log transform class.
    """

    input_dims: int = 1
    output_dims: int = 1

    def __init__(
        self,
        base: float,
        linthresh: float,
        linscale: float,
        clip_value: float | str,
    ) -> None:
        """
        Initialize the log transformation.

        Parameters
        ----------
        base : float
            Base of the logarithm.
        linthresh : float
            The range within which the plot is linear. This avoids having the plot go
            to infinity around zero.
        linscale : float
            This allows the linear range (-linthresh to linthresh) to be stretched
            relative to the logarithmic range.
        clip_value: float | "mask"
            Inputs that are <=0 (and therefore make a problem for the log scale) are set
            to this value in LinLogTransform. If the clip_value is "mask", values will
            be set to -np.inf and not plotted.
        Raises
        ------
        ValueError
            Raised if 'base' is not larger than 1, 'linthresh' is not positive or
            'linscale' is not positive.
        """
        super().__init__()

        if base <= 1.0:
            raise ValueError("'base' must be larger than 1")
        if linthresh <= 0.0:
            raise ValueError("'linthresh' must be positive")
        if linscale <= 0.0:
            raise ValueError("'linscale' must be positive")

        self.base: float = base
        self.linthresh: float = linthresh
        self.linscale: float = linscale
        self.clip_value: float | str = clip_value
        self._linscale_adj: float = linscale / (1.0 - self.base**-1)
        self._log_base: float = np.log(base)

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:
        """
        Perform the custom log transformation on values.

        Parameters
        ----------
        values : np.ndarray
            The input values to be transformed.

        Returns
        -------
        np.ndarray
            The transformed values.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                np.abs(values) <= self.linthresh,
                np.sign(values)
                * self.linthresh
                * (1.0 + np.log(np.abs(values) / self.linthresh) / self._log_base),
                values,
            )

            out = np.where(
                np.abs(values) > self.linthresh,
                self.linthresh
                + np.sign(values)
                * self._linscale_adj
                * (np.abs(values) - self.linthresh),
                out,
            )

            # Clipping the values similar to LogScale
            if self.clip_value == "mask":
                out[values <= 0] = -np.inf
            elif isinstance(self.clip_value, (int, float)) and self.clip_value > 0:
                out[values <= 0] = self.linthresh * (
                    1.0
                    + np.log(np.abs(self.clip_value) / self.linthresh) / self._log_base
                )
            else:
                raise ValueError("clip_value must either be 'mask' or a number>0.")
            return out

    def inverted(self) -> "InvertedLinLogTransform":
        """
        Get the inverse transformation of this transformation.

        Returns
        -------
        InvertedLinLogTransform
            The inverse transformation object.
        """
        return InvertedLinLogTransform(
            self.base, self.linthresh, self.linscale, self.clip_value
        )


class InvertedLinLogTransform(Transform):
    """
    Inverted version of the custom log transform.
    This transformation class provides an inverse to the logarithmic
    transformation. It allows for the mapping back of values that underwent the
    LinLogTransform.
    """

    input_dims: int = 1
    output_dims: int = 1

    def __init__(
        self,
        base: float,
        linthresh: float,
        linscale: float,
        clip_value: float | str,
    ) -> None:
        """
        Initialize the inverted log transformation.

        Parameters
        ----------
        base : float
            Base of the logarithm.
        linthresh : float
            The range within which the plot is linear in the original transformation.
        linscale : float
            Used to stretch the linear range in the original transformation.
        clip_value: float | "mask"
            Inputs that are <=0 (and therefore make a problem for the log scale) are set
            to this value in LinLogTransform. If the clip_value is "mask", values will
            be set to -np.inf and not plotted. The default is "mask".
        """
        super().__init__()

        linlog = LinLogTransform(base, linthresh, linscale, clip_value)
        self.base: float = base
        self.linthresh: float = linthresh
        self.invlinthresh: float = linlog.transform(linthresh)
        self.linscale: float = linscale
        self.clip_value: float | str = clip_value
        self._linscale_adj: float = linscale / (1.0 - self.base**-1)

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:
        """
        Perform the inverted custom log transformation on values.

        Parameters
        ----------
        values : np.ndarray
            The input values to be transformed.

        Returns
        -------
        np.ndarray
            The transformed values.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                np.abs(values) <= self.invlinthresh,
                np.sign(values)
                * self.linthresh
                * np.exp((np.abs(values) / self.linthresh) - 1.0),
                values,
            )

            out = np.where(
                np.abs(values) > self.invlinthresh,
                np.sign(values)
                * (
                    self.linthresh
                    + (np.abs(values) - self.invlinthresh) / self._linscale_adj
                ),
                out,
            )
        return out

    def inverted(self) -> "LinLogTransform":
        """
        Get the inverse transformation of this transformation, which is the original.

        Returns
        -------
        LinLogTransform
            The original transformation object.
        """
        return LinLogTransform(
            self.base, self.linthresh, self.linscale, self.clip_value
        )


# ###################### FORMATTER #####################################################


class LinLogFormatter(ScalarFormatter):
    """
    Lin-log formatter for axis labels.

    This formatter is tailored for logarithmic scales that also have a linear region.
    For values within the linear threshold, it formats with precision that aligns with
    the scale of the number. For values outside this threshold, use the default
    ScalarFormatter, unless their integer in which case they are returned as integers.
    """

    def __init__(self, linthresh: float):
        """
        Initialize the formatter

        Parameters
        ----------
        linthresh : float
            The threshold between linear and logarithmic regime.
        """
        super().__init__()
        self.linthresh = linthresh

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """
        Format a value according to the custom log formatting rules.

        Parameters
        ----------
        x : float
            The value to be formatted.
        pos : Optional[int], optional
            The position of the tick (can be ignored for this custom formatter).

        Returns
        -------
        str
            The formatted value as a string.
        """
        if abs(x) < self.linthresh:
            # Calculate the number of decimal places based on the magnitude of x
            decimal_places: int = abs(int(np.log10(abs(x)))) if abs(x) > 0 else 0
            format_string: str = "{:." + str(decimal_places) + "f}"
            return format_string.format(x)

        elif x.is_integer():
            return f"{int(x)}"

        else:
            # Default ScalarFormatter formatting
            label = super().__call__(x, pos)

            # Remove trailing zeros, and decimal point if it's the last character
            if "." in label:
                label = label.rstrip("0").rstrip(".")
            return label


# ###################### LOCATOR #######################################################


class CombinedLogLinearLocator(Locator):
    """
    A custom locator for axes that combines both logarithmic and linear scales.

    This locator creates tick locations suitable for log-linear plots, where there's
    a transition from a logarithmic scale to a linear scale at a certain threshold.
    This is useful for visualizing datasets that span several orders of magnitude.

    """

    def __init__(
        self,
        base: float = 10.0,
        subs: tuple = (1.0,),
        linthresh: float = 2,
        numticks_log: Optional[int] = None,
        numbins: Optional[int | str] = "auto",
    ) -> None:
        """
        Initialize the combined log-linear locator.

        Parameters
        ----------
        base : float, optional
            Base of the logarithm. The default is 10.0.
        subs : tuple, optional
            The sequence of the location of the minor ticks. For example, in a logarithm
            base 10 scale, you might want minor ticks at 1, 2, ..., 9. The default is
            (1.0,).
        linthresh : float, optional
            The range within which the numbers are considered to be in the linear scale.
            The default is 2.
        numticks_log : Optional[int], optional
            The number of ticks intended for the logarithmic scale. The default is None.
        numbins : Optional[int | str], optional
            The number of bins intended for the linear scale. The default is auto.
        """
        super().__init__()
        self.base: float = base
        self.subs: tuple = subs
        self.linthresh: float = linthresh
        self.numticks_log: Optional[int] = numticks_log
        self.numbins: Optional[int | str] = numbins
        # Separate locators for log and linear regions
        self.log_locator = LogLocator(base=base, subs=subs, numticks=numticks_log)
        self.maxnlocator = MaxNLocator(numbins, symmetric=True)

    def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
        """
        Calculate tick values given the range of the data.

        Parameters
        ----------
        vmin : float
            Minimum value of the data.
        vmax : float
            Maximum value of the data.

        Returns
        -------
        np.ndarray
            Array of tick values.
        """
        if vmin <= 0.0 and self.axis:
            vmin = self.axis.get_minpos()

        # Define range limits for log and linear parts
        log_vmin, log_vmax = min(vmin, self.linthresh), min(vmax, self.linthresh)
        linear_vmin, linear_vmax = max(vmin, self.linthresh), max(vmax, self.linthresh)

        # Get ticks for log and linear regions
        log_ticks = self.log_locator.tick_values(log_vmin, log_vmax)
        log_ticks = log_ticks[log_ticks <= self.linthresh]

        linear_ticks = self.maxnlocator.tick_values(linear_vmin, linear_vmax)
        linear_ticks = linear_ticks[linear_ticks > linear_vmin]

        # Combine and return the ticks
        return np.concatenate([log_ticks, linear_ticks])

    def __call__(self) -> np.ndarray:
        """
        Return tick values for the current axis view interval.

        Returns
        -------
        np.ndarray
            Array of tick values.
        """
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)


class CustomLogLocator(LogLocator):
    """
    A custom locator for axes that combines both logarithmic and linear scales, used for
    the minor ticks.

    This locator creates tick locations suitable for log plots, but only below the
    threshold value linthresh.
    """

    def __init__(self, linthresh: float = 2, *args: Any, **kwargs: Any):
        """
        Initialize the combined log-linear locator.

        Parameters
        ----------
        linthresh : float, optional
            The range within which the numbers are considered to be in the linear scale.
            The default is 2.
        args, kwargs : Any
            Further parameter passed to LogLocator.

        """
        super().__init__(*args, **kwargs)
        self.linthresh = linthresh

    def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
        """
        Calculate tick values given the range of the data, cut off at threshold.

        Parameters
        ----------
        vmin : float
            Minimum value of the data.
        vmax : float
            Maximum value of the data.

        Returns
        -------
        np.ndarray
            Array of tick values.
        """
        tick_values = super().tick_values(vmin, vmax)
        return np.array([val for val in tick_values if val <= self.linthresh])

    def __call__(self) -> np.ndarray:
        """
        Return tick values for the current axis view interval.

        Returns
        -------
        np.ndarray
            Array of tick values.
        """
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)


# ###################### SCALE #########################################################


class LinLogScale(LogScale):
    """
    A custom linear-logarithmic scale.

    This scale provides a logarithmic transformation for data that may uses a log
    scale below a certain threshold and a linear scale above.
    """

    name: str = "linlog"

    def __init__(
        self,
        axis: Axes,
        base: float = 10,
        linthresh: float = 1,
        linscale: float = 1,
        clip_value: float | str = "mask",
        subs: Optional[tuple] = None,
    ) -> None:
        """
        Initialize the custom symmetrical logarithmic scale.

        Parameters
        ----------
        axis : Axis
            The axis object to which this scale is attached.
        base : float, optional
            Base of the logarithm. The default is 10.
        linthresh : float, optional
            The range within which the numbers are linearly scaled. The default is 2.
        linscale : float, optional
            Factor by which data within linthresh is linearly scaled. The default is 1.
        clip_value: float | "mask"
            Inputs that are <=0 (and therefore make a problem for the log scale) are set
            to this value in LinLogTransform. If the clip_value is "mask", values will
            be set to -np.inf and not plotted. The default is "mask".
        subs : Optional[tuple], optional
            The sequence of the location of the minor ticks.
        """
        super().__init__(axis)
        self._transform = LinLogTransform(base, linthresh, linscale, clip_value)
        self.subs = subs

    @property
    def base(self) -> float:
        """Base of the logarithm used by this scale."""
        return self._transform.base

    @property
    def linthresh(self) -> float:
        """Range within which numbers are linearly scaled."""
        return self._transform.linthresh

    @property
    def linscale(self) -> float:
        """Factor by which data within linthresh is linearly scaled."""
        return self._transform.linscale

    def set_default_locators_and_formatters(self, axis: Axes) -> None:
        """
        Set the default locators and formatters for this scale.

        Parameters
        ----------
        axis : Axes
            The axis object to which this scale is attached.
        """
        formatter = LinLogFormatter(self.linthresh)

        # Set major and minor locators and formatters
        axis.set_major_locator(
            CombinedLogLinearLocator(base=self.base, linthresh=self.linthresh)
        )
        axis.set_major_formatter(formatter)

        axis.set_minor_locator(
            CustomLogLocator(
                base=self.base, linthresh=self.linthresh, subs=np.arange(2, 10)
            ),
        )
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self) -> LinLogTransform:
        """
        Return the transformation associated with this scale.

        Returns
        -------
        LinLogTransform
            The transformation object.
        """
        return self._transform


register_scale(LinLogScale)
