#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:34:58 2023

@author: chris
"""
from typing import Any, Callable, Optional

import methodtools
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rv_continuous, truncnorm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from gallifrey.data.paths import Path
from gallifrey.utilities.structures import find_closest, make_meshgrid


class Population:
    """
    NGPPS planet population.

    """

    def __init__(
        self, population_id: str, age: int, category_dict: dict[str, Callable]
    ) -> None:
        """
        Initialize object, load dataframe with population data
        and add planet category flags based on category_dict.

        Parameters
        ----------
        population_id : str
            Name of the population run.
        age : int
            Age of system at time of snapshot.
        category_dict: dict[str, Callable]
            Dictonary that contains the names of the categories, and a function
            how to assign a given category to a row in the population dataframe based
            on the values in the row.

        """
        # load populations
        self.population = pd.read_csv(
            Path().external_data(f"NGPPS/{population_id}/snapshot_{age}.csv")
        )

        # add planet flags and number of planets dataframe
        self.add_category_flags(category_dict)
        self.planet_number = self.count_planets(list(category_dict.keys()))

    def match_dataframes(
        self,
        category: str,
        system_dataframe: pd.DataFrame,
        population_dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Match the dataframes (e.g. the monte carlo variables used to run the
        simulations) to a column in the population_dataframe.

        Parameters
        ----------
        category: str
            Category that is matched to system variables.
        system_dataframe: pd.DataFrame
            Dataframe containing system_ids and system variables (monte carlo
            varliables).
        population_dataframe : pd.DataFrame, optional
            Dataframe that has the "system_id" column and category columns, used for
            matching. The default is None, which defaults to the planet number
            dataframe.

        Returns
        -------
        matched_systems : pd.DataFrame
            Dataframe with system variables and matched category, merged on system_id.

        """
        if population_dataframe is None:
            try:
                population_dataframe = self.planet_number
            except AttributeError:
                raise AttributeError(
                    "If no match_dataframe is given, default to "
                    "'planet_number' dataframe, which needs to be "
                    "created first using 'count_planets'."
                )
        # match on system_id
        matched_dataframe = system_dataframe.merge(
            population_dataframe[category], on="system_id"
        )
        return matched_dataframe

    def add_category_flags(self, category_dict: dict[str, Callable]) -> None:
        """
        Adds planet categories to columns.

        Parameters
        ----------
        category_dict: dict[str, Callable]
            Dictonary that contains the names of the categories, and a function
            how to assign a given category to a row in the population dataframe based
            on the values in the row.

        """
        # Apply each function to the DataFrame to create new columns
        for category, condition in category_dict.items():
            self.population[category] = self.population.apply(condition, axis=1)

    def count_planets(self, categories: list[str]) -> pd.DataFrame:
        """
        Count the number of planets for each planet category by grouping the dataframe
        based on the system_id and then summing over the number of True values for a
        given category.

        Parameters
        ----------
        categories: list[str]
            List of considered categories for which to count the planet number.

        Returns
        -------
        planet_number : DataFrame
            Dataframe containing the system_id and number of planets per category for
            that system.

        """
        planet_number = self.population.groupby("system_id")[categories].sum()
        return planet_number.astype(int)


class Systems:
    """
    Population system properties.

    """

    def __init__(
        self, population_id: str, distributions: Optional[dict[str, Callable]] = None
    ) -> None:
        """
        Load system variables connected to a specific simulation run given by the
        population_id. If the host star mass is smaller than 1, rescale according
        to prescriptions given in paper IV (Burn2021).
        Also create distributions of these variables according to the description in
        paper II (Emsenhuber2021). The default distributions are truncated Gaussians,
        custom distributions can be passed using the distributions parameter.

        Parameters
        ----------
        population_id : str
            Name of population id to retrieve system data for.
        distributions: Optional[dict[str,Callable]], optional
            Dictonary of probability distributions for variable. Of form
            {variable_name: Callable}.
        """
        self.host_star_mass_dict = {
            "ng96": 1,
            "ng74": 1,
            "ng75": 1,
            "ng76": 1,
            "ngm12": 0.7,
            "ngm11": 0.5,
            "ngm14": 0.3,
            "ngm10": 0.1,
        }

        try:
            self.host_star_mass = self.host_star_mass_dict[population_id]
        except KeyError:
            raise ValueError("population_id not known.")

        # create variable dataframe
        self.variables = self.load_system_variables(population_id)
        self.variable_names = tuple(self.variables.columns)

        # get variable bounds (upper and lower value in sample)
        self.bounds = self.variable_bounds()

        # Create Gaussian parameter distributions according to paper description
        self.variable_gaussian_parameter = {
            "log_initial_mass": (-1.49, 0.35),
            "[Fe/H]": (-0.02, 0.22),
            "log_inner_edge": (-1.26, 0.206),
            "log_photoevaporation": (-6, 0.5),
        }
        self.distributions = {}
        for variable_name in self.variable_names:
            self.distributions[variable_name] = self._truncated_gaussian(variable_name)

        # overwrite distributions with custom distributions if given
        if isinstance(distributions, dict):
            for variable_name, function in distributions.items():
                if variable_name not in self.distributions.keys():
                    raise ValueError(f"{variable_name!r} is not a valid variable name.")
                self.distributions[variable_name] = function

    def load_system_variables(self, population_id: str) -> pd.DataFrame:
        """
        Loads system monte carlo variables as provided by Emsenhuber. Rescale according
        to host star mass following paper IV (Burn2021).

        Parameters
        ----------
        population_id : str
            Name of population id to retrieve system data for.

        Raises
        ------
        NotImplementedError
            Raised if population_id does not match one of the solar-like runs, since
            currently only those have the variables available.

        Returns
        -------
        system_variables : DataFrame
            Dataframe containing the system variables.

        """
        raw_variables = self._load_raw_system_variables()

        system_variables = pd.DataFrame()
        system_variables["system_id"] = raw_variables["system_id"]
        # calculate Monte Carlo variables as described in paper II (Emsenhuber2021):
        # mass of gas disk
        system_variables["log_initial_mass"] = np.log10(
            (raw_variables["aout"] / 10) ** 1.6 * 2e-3
        )  # paper Eq. 1
        # metallicity
        system_variables["[Fe/H]"] = np.log10(
            raw_variables["fpg"] / 0.0149  # solar dust-to-gas ratio
        )  # paper Eq. 2
        # inner edge
        system_variables["log_inner_edge"] = np.log10(raw_variables["ain"])
        # photo evaporation
        system_variables["log_photoevaporation"] = np.log10(raw_variables["mwind"])

        # scale system variables according to host star mass
        system_variables = self.scale_variables(system_variables, self.host_star_mass)

        # # make system_id the index of the dataframe
        system_variables = system_variables.set_index("system_id")
        return system_variables

    @staticmethod
    def scale_variables(
        system_variables: pd.DataFrame,
        host_star_mass: float,
    ) -> pd.DataFrame:
        """
        Scaling the variables with the mass of the host star according to paper IV
        (Burn2021).

        Parameters
        ----------
        system_variables : pd.DataFrame
            Dataframe containing the system variables.
        host_star_mass : float
            Mass of host star, must be in [0.1, 0.3, 0.5, 1]..

        Returns
        -------
        system_variables : pd.DataFrame
            Dataframe containing the rescaled system variables.

        """
        # gas mass scales linearly with star mass
        if "log_initial_mass" in system_variables:
            system_variables["log_initial_mass"] -= np.log10(host_star_mass)
        # inner edge scales with (star mass)^1/3
        if "log_inner_edge" in system_variables:
            system_variables["log_inner_edge"] -= 1 / 3 * np.log10(host_star_mass)
        return system_variables

    def variable_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Calculate paramter bounds for the system variables (monte carlo variables).

        Raises
        ------
        AttributeError
            Raised if variables attribute does not exist.

        Returns
        -------
        dict[str,list[float,float]]
            Dictonaries with bound, of form {parameter_name : (min, max)}.

        """
        if not hasattr(self, "variables"):
            raise AttributeError(
                "No 'variables' dataframe found to calculate bounds from. "
                "Create first using 'load_system_variables'."
            )

        bounds = pd.DataFrame(
            {"min": self.variables.min(), "max": self.variables.max()}
        )
        bounds_dict = {
            index: (row["min"], row["max"]) for index, row in bounds.iterrows()
        }
        return bounds_dict

    def variable_grid(
        self,
        num_bins: int,
        included_variables: Optional[tuple[str, ...]] = None,
        custom_bounds: Optional[dict[str, tuple]] = None,
        as_dataframe: bool = False,
    ) -> list[np.ndarray] | pd.DataFrame:
        """
        Create a equally spaced meshgrid of the variables between the variable bounds.
        The parameter included_variables can be used to choose the variables.
        If custom bounds are needed, they can be passed directly as a list of 2-tuples,
        which make up the lower and upper bounds, respectively.

        Parameters
        ----------
        num_bins : int
            Number of bins for all dimensions.
        included_variables : Optional[tuple[str, ...]], optional
            Names of variables included in the calculation. The default is None, which
            includes all variables.
         custom_bounds : Optional[dict[str, tuple]],  optional
             A dictonary of custom bound to create the meshgrid from. The key must be
             the variable name and the value a 2-tuple of (lower_bound, upper_bound).
             If this parameter is passed included_variables is ignored. The default is
             None, which then uses the variable bounds stored as attributes.
        as_dataframe: bool, optional
            If True, return a dataframe of coordinate pairs, rather than meshgrid. The
            default is False.

        Returns
        -------
        list[np.ndarray]
            The list of arrays that make up the meshgrid if as_list is False. If
            as_dataframe is True, returns dataframe of coordinate pairs instead.

        """
        if included_variables is None:
            included_variables = self.variable_names

        # get bounds of included variable
        if custom_bounds:
            bounds = custom_bounds
        else:
            bounds = {
                variable: self.bounds[variable] for variable in included_variables
            }

        # create meshgrid
        meshgrid = make_meshgrid(
            bounds.values(), num_bins=num_bins, as_list=as_dataframe
        )

        if as_dataframe:
            # turn coordinate pairs into dataframe
            meshgrid = pd.DataFrame(meshgrid, columns=bounds.keys())
        return meshgrid

    def sample_distribution(
        self,
        num: int,
        included_variables: Optional[tuple[str, ...]] = None,
    ) -> pd.DataFrame:
        """
        Sample variable distribtuions. The parameter included_variables can be used to
        choose the variables.

        Parameters
        ----------
        num : int
            Number if samples.
        included_variables : Optional[tuple[str, ...]], optional
            Names of variables included in the calculation. The default is None, which
            includes all variables.

        Returns
        -------
        result : pd.DataFrame
            Dataframe of samples.

        """
        if included_variables is None:
            included_variables = self.variable_names

        # choose distributions to include
        distributions = [self.distributions[var] for var in included_variables]

        # sample distributions
        result = []
        for distribution in distributions:
            result.append(distribution.rvs(num))

        return pd.DataFrame(np.array(result).T, columns=included_variables)

    def pdf(
        self,
        variable_values: ArrayLike,
        included_variables: Optional[tuple[str, ...]] = None,
        multiply: bool = True,
    ) -> pd.DataFrame | NDArray:
        """
        Calculate the value of the pdfs. for some variable values by evaluating the
        individual variable distributions. If multiply is True, the results are
        multiplied to form the multivariate pdf (assuming independence). The parameter
        included_variables can be used to choose the variables.

        Parameters
        ----------
        variable_values : ArrayLike
            Input values for the distributions (must have same length as distributions
            dict).
        included_variables : Optional[tuple[str, ...]], optional
            Names of variables included in the calculation. The default is None, which
            includes all variables.
        multiply : bool, optional
            Choose if individual pdf values should be multiplied to obtain the value
            of the multivariate pdf or not.

        Returns
        -------
        result : pd.DataFrame | NDArray
            Value of pdf. If multiply is False, returns Dataframe with value for every
            distribution. If True, return Array with product of values.

        """
        if included_variables is None:
            included_variables = self.variable_names

        variable_values = np.asarray(variable_values)
        if variable_values.shape[-1] != len(included_variables):
            raise ValueError(
                "Shape of variable_values must match number of "
                "distributions in distribution dict (or length of "
                "included_variables if given)."
            )

        distributions = [self.distributions[var] for var in included_variables]

        result = []
        for distribution, value in zip(distributions, variable_values):
            result.append(distribution.pdf(value))

        result_array = np.array(result)
        if result_array.ndim == 1:
            result_array = result_array.reshape(1, -1)

        if multiply:
            return np.prod(result_array, axis=1)
        else:
            return pd.DataFrame(result_array, columns=included_variables)

    def _truncated_gaussian(self, variable_name: str) -> rv_continuous:
        """
        Create truncated Gaussian distribution for variable according to stored
        parameter.

        Parameters
        ----------
        variable_name : str
            Name of the variable connected to the distribution.

        Returns
        -------
        distribution : rv_continuous
            Frozen truncated Gaussian scipy distribution.

        """
        # get parameter from stored values
        lower_bound, upper_bound = self.bounds[variable_name]
        mu, sigma = self.variable_gaussian_parameter[variable_name]

        # create distributions
        distribution = truncnorm(
            loc=mu,
            scale=sigma,
            a=(lower_bound - mu) / sigma,  # a and b are given in
            b=(upper_bound - mu) / sigma,  # terms of sigma
        )
        return distribution

    def _load_raw_system_variables(self) -> pd.DataFrame:
        """
        Loads file with system variables provided by Emsenhuber and preprocess.

        Returns
        -------
        variables : DataFrame
            Dataframe containing the raw system variables.
        """
        # column names
        columns = [
            "system_id",
            "mstar",
            "sigma",
            "expo",
            "ain",
            "aout",
            "fpg",
            "mwind",
        ]
        # read variables data file
        raw_variables = pd.read_csv(
            Path().external_data("NGPPS/NGPPS_variables.txt"),
            delimiter=r"\s+",
            names=columns,
        )

        # modify the 'system_id' column to remove 'system_id' prefix
        raw_variables["system_id"] = raw_variables["system_id"].str[3:].astype(int)

        # convert columns to float
        for col in columns[1:]:
            raw_variables[col] = raw_variables[col].map(
                lambda x: float(x.split("=")[1])
            )
        return raw_variables


class PlanetModel:
    """
    Planet model.

    """

    def __init__(self, num_embryos: int) -> None:
        """
        Initialize a planet model based on specific population, based on number
        of embryos. Populations with a host star mass < 1 only have 50 embryo run.

        Parameters
        ----------
        num_embryos : int
            Number of embryos used for population run, must be in [10, 20, 50, 100].

        """
        self.num_embryos = num_embryos

        # available populations dict with keys [number of embryos, mass of star]
        self.populations_dict = {
            (10, 1): "ng96",
            (20, 1): "ng74",
            (50, 1): "ng75",
            (100, 1): "ng76",
            (50, 0.7): "ngm12",
            (50, 0.5): "ngm11",
            (50, 0.3): "ngm14",
            (50, 0.1): "ngm10",
        }

        # available snapshot ages (code to generate them so that get_snapshot_ages()
        # from dace module doesn't have to be run, and there is no huge list here. Might
        # lead to errors if snapshot ages change for a different population run)
        self.available_ages = tuple(
            [
                int(j * (10**i))
                for i in range(5, 11)
                for j in range(1, 10 if i != 10 else 2)
            ]
        )

        # define planet categories
        self.category_dict = {
            "Dwarf": lambda row: row["total_mass"] < 0.5,
            "Earth": lambda row: 0.5 <= row["total_mass"] < 2,
            "Super-Earth": lambda row: 2 <= row["total_mass"] < 10,
            "Neptunian": lambda row: 10 <= row["total_mass"] < 30,
            "Sub-Giant": lambda row: 30 <= row["total_mass"] < 100,
            "Giant": lambda row: 100 <= row["total_mass"],
            "D-Burner": lambda row: 4322 <= row["total_mass"],
        }
        self.categories = list(self.category_dict.keys())

    def get_population_id(self, num_embryos: int, host_star_mass: float) -> str:
        """
        Get population ID for run with specified number of embryos and host_star_mass

        Parameters
        ----------
        num_embryos : int
            Number of embryos used for population run, must be in [10, 20, 50, 100].
        host_star_mass : float
            Mass of host star, must be in [0.1, 0.3, 0.5, 1]..

        Returns
        -------
        str
            Population ID for run.

        """
        return self.populations_dict[(num_embryos, host_star_mass)]

    @methodtools.lru_cache(maxsize=256)
    def get_population(self, population_id: str, age: int) -> Population:
        """
        Retrieve population for the given population ID and age using the lru_cache
        for efficiency.

        Parameters
        ----------
        population_id : str
            Name of the population run.
        age : int
            Age of the population to retrieve.

        Returns
        ----------
        Population
            An instance of the Population class.

        """
        if age not in self.available_ages:
            raise ValueError("Age does not match any snapshot.")
        return Population(population_id, age, self.category_dict)

    @methodtools.lru_cache(maxsize=256)
    def get_systems(self, population_id: str) -> Systems:
        """
        Retrieve system information for the given population ID using the lru_cache
        for efficiency.

        Parameters
        ----------
        population_id : str
            Name of the population run.

        Returns
        ----------
        Systems
            An instance of the Systems class.

        """
        return Systems(population_id)

    @methodtools.lru_cache(maxsize=512)
    def get_planet_function(
        self,
        category: str,
        population_id: str,
        included_variables: Optional[tuple[str]] = None,
        ages: Optional[tuple[int, ...] | int] = None,
        neighbors: int = 30,
        weights: str = "uniform",
        **kwargs: Any,
    ) -> tuple[KNeighborsRegressor, StandardScaler]:
        """
        Calculate KNN interpolation for a population snapshot for a given category.

        Parameters
        ----------
        category: str
            Category that is matched to system variables.
        population_id : str
            Name of the population run.
        included_variables : Optional[tuple[str]], optional
            Names of variables included in the calculation. The default is None, which
            includes all variables.
        ages : Optional[tuple[int, ...]] | int], optional
            A list of snapshot ages to include in the interpolation. The default is
            None, which includes every age found in available_ages.
        neighbors : int, optional
            Number of neighbors to use in the KNN regression. The default is 30.
        weights : str, optional
            Weight function to use in prediction for KNN regression. he default is
            'uniform'.
        kwargs : dict
            Additional arguments to pass to the KNeighborsRegressor.

        Returns
        ----------
        KNeighborsRegressor
            KNeighborsRegressor model fitted on the population data.
        StandardScaler
            Gaussian scaler used to scale the data before fitting, scaler needs to
            be applied to any dataset before calling knn.predict.

        """
        match ages:
            case None:
                ages = self.available_ages
            case int():
                ages = (ages,)
            case tuple():
                ages = ages
            case _:
                raise ValueError("ages must be None, int or tuple[int]")

        # gather data for considered snapshots
        datasets = []
        for age in ages:
            population = self.get_population(population_id, age)
            systems = self.get_systems(population_id)
            data = population.match_dataframes(
                category, system_dataframe=systems.variables
            )
            data["age"] = age
            datasets.append(data)
        data = pd.concat(datasets)

        # define KNN regressor
        knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, **kwargs)

        # prepare input data
        input_data = data.drop(columns=category)
        if included_variables:
            input_data = input_data[list(included_variables) + ["age"]]

        # scale input data before passing it to the regressor
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(input_data),
            columns=input_data.columns,
        )

        # save the features seen during training
        self.features = input_data.columns

        # fit and return the KNN regressor with the data
        fitted_knn = knn.fit(data_scaled, data[category])
        return fitted_knn, scaler

    def prediction(
        self,
        categories: str | list,
        host_star_mass: float,
        variables: Optional[pd.DataFrame] = None,
        included_variables: Optional[tuple[str]] = None,
        ages: Optional[tuple[int, ...] | int] = None,
        return_full: bool = False,
        num_samples: int = 5000,
        default_age: int = 100000000,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Predict a number of planets in a given category given some input system
        variable using the KNN regressor. Variables that are not directly passed are
        sampled from systems variable distributions, meaning the output is stochastic.

        IMPORTANT: The KNN regressor considers monte carlo variables as well as
        snapshot/system age, but since no distribution for ages is given, the
        'variables' dataframe needs to contain a column 'ages'.

        Parameters
        ----------
        category: str
            Category that is matched to system variables.
        host_mass_star:
            Mass of host star, must be in [0.1, 0.3, 0.5, 1].
        variables : Optional[pd.DataFrame], optional
            DataFrame of variables to be used in the prediction, the remaining variables
            are sampled from variable distributions. The default is None, which means
            all parameter are sampled from distribution.
        included_variables : Optional[tuple[str]], optional
            Names of variables included in the calculation. If variables is not None,
            all variable names in the variables dataframe must be also in
            included_variables. The default is None, which includes all variables.
        ages : Optional[tuple[int, ...]] | int], optional
            A list of snapshot ages to include in the interpolation. The default is
            None, in that case the relevant ages are inferred from the age column
            of the variables dataframe.
        return_full : bool, optional
            If True, return the full DataFrame (variables + prediction). Otherwise,
            return only the category column (i.e. the prediction).
        num_samples: int
            If variables dataframe is None, use this many samples from distribution.
            Otherwise number of samples is inferred from shape of variables dataframe.
            The default is 5000.
        default_age: int
            The default age to assign to the population if the variables dataframe is
            None. Otherwise, age is inferred from age column of variables dataframe.
            The default is 100000000, i.e. 100Myr.
        kwargs : Any
            Additional arguments to pass to the get_planet_function method.

        Returns
        ----------
        pd.DataFrame
            The predicted values as dataframe. If return_full=True, this includes
            the sample of variables used for the calculation.

        """
        if isinstance(categories, str):
            categories = [categories]
        elif isinstance(categories, list):
            pass
        else:
            raise ValueError("categories must be str or list of strings.")

        # identify population run
        try:
            population_id = self.get_population_id(self.num_embryos, host_star_mass)
        except KeyError:
            raise ValueError(
                "Population not found. Are you sure the combinations of "
                "embryo number and host star mass exists?"
            )

        if variables is not None:
            # check if ages column exists and set number of samples
            if "age" not in variables.columns:
                variables["age"] = default_age
            num_samples = len(variables)

        # get systems
        systems = self.get_systems(population_id)
        # sample the system variable distributions amd scale according to host mass
        sample = systems.sample_distribution(
            num_samples, included_variables=included_variables
        )
        sample = systems.scale_variables(sample, host_star_mass)

        # check if included_variables matches requirements
        if included_variables is not None:
            if not isinstance(included_variables, tuple):
                raise ValueError(
                    "included_variables must be tuple " "(needed for hashing)."
                )
            if not set(included_variables).issubset(set(systems.variable_names)):
                raise ValueError(
                    "included_variables must be subset of "
                    f" {systems.variable_names}."
                )

        # if variables dataframe does not exist, set default age
        # otherwise if it exists, overwrite sample columns with columns given by
        # variables dataframe
        if variables is None:
            sample["age"] = default_age
        else:
            for column in variables.columns:
                if included_variables is not None:
                    if (column == "age") or (column in included_variables):
                        pass
                    else:
                        raise ValueError(
                            f"{column!r}-column found in 'variables' "
                            "dataframe but not part of "
                            "'included_variables'."
                        )

                # match on same index, otherwise matching error can occur
                sample[column] = variables.reset_index(drop=True)[column]

        # find relevant snapshot ages to be passed to get_planet_function,
        # saves a lot of time if ages are all the same or similar
        if ages is None:
            ages = tuple(np.unique(find_closest(sample["age"], self.available_ages)))

        # Get the KNN model and predict the category
        prediction_dataframe = sample.copy()
        for category in categories:
            knn, scaler = self.get_planet_function(
                category=category,
                population_id=population_id,
                included_variables=included_variables,
                ages=ages,
                **kwargs,
            )
            # scale data with same scaler used for fitting KNN
            data_scaled = pd.DataFrame(scaler.transform(sample), columns=sample.columns)
            prediction_dataframe[category] = knn.predict(data_scaled)

            # check if all values in category column are whole numbers, and if so
            # convert to int data type
            if prediction_dataframe[category].apply(float.is_integer).all():
                prediction_dataframe[category] = prediction_dataframe[category].astype(
                    int
                )

        if return_full:
            return prediction_dataframe
        else:
            return prediction_dataframe[categories]
