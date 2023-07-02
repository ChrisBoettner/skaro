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

from gallifrey.data.paths import Path
from gallifrey.utilities.structures import find_closest


class Population:
    """
    NGPPS planet population.

    """

    def __init__(self, population_id: str, age: int) -> None:
        """
        Initialize object, load dataframe and add planet categories.

        Parameters
        ----------
        population_id : str
            Name of the population run.
        age : int
            Age of system at time of snapshot.

        """
        # load populations
        self.population = pd.read_csv(
            Path().raw_data(f"NGPPS/{population_id}/snapshot_{age}.csv")
        )

        # add planet categories
        self.category_dict = {
            "Dwarf": lambda row: row["total_mass"] < 0.5,
            "Earth": lambda row: 0.5 <= row["total_mass"] < 2,
            "SuperEarth": lambda row: 2 <= row["total_mass"] < 10,
            "Neptunian": lambda row: 10 <= row["total_mass"] < 30,
            "SubGiant": lambda row: 30 <= row["total_mass"] < 300,
            "Giant": lambda row: 300 <= row["total_mass"],
            "DBurning": lambda row: 4322 <= row["total_mass"],
        }
        self.categories = list(self.category_dict.keys())

        # add planet flags and number of planets dataframe
        self.add_category_flags()
        self.planet_number = self.count_planets()

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

    def add_category_flags(self) -> None:
        """
        Adds planet categories to columns.

        """
        # Apply each function to the DataFrame to create new columns
        for category, condition in self.category_dict.items():
            self.population[category] = self.population.apply(condition, axis=1)

    def count_planets(self) -> pd.DataFrame:
        """
        Count the number of planets for each planet category by grouping the dataframe
        based on the system_id and then summing over the number of True values for a
        given category.

        Returns
        -------
        planet_number : DataFrame
            Dataframe containing the system_id and number of planets per category for
            that system.

        """
        planet_number = self.population.groupby("system_id")[self.categories].sum()
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
        population_id.
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
        # create variable dataframe
        self.variables = self.load_system_variables(population_id)
        self.variable_names = list(self.variables.columns)

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
        Loads system monte carlo variables. Currently only implemented for solar-like
        runs with population_id's ["ng96", "ng74", "ng75", "ng76"].

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
        if population_id in ["ng96", "ng74", "ng75", "ng76"]:
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
                raw_variables["fpg"] / 0.0149
            )  # paper Eq. 2
            # inner edge
            system_variables["log_inner_edge"] = np.log10(raw_variables["ain"])
            # photo evaporation
            system_variables["log_photoevaporation"] = np.log10(raw_variables["mwind"])

            # make system_id the index of the dataframe
            system_variables = system_variables.set_index("system_id")
        else:
            raise NotImplementedError(
                "Population ID does not much any solar-like run. If you "
                "use other runs with other stellar masses, the system "
                "variables won't match."
            )
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
                "Create frist using 'load_system_variables'."
            )

        bounds = pd.DataFrame(
            {"min": self.variables.min(), "max": self.variables.max()}
        )
        bounds_dict = {
            index: (row["min"], row["max"]) for index, row in bounds.iterrows()
        }
        return bounds_dict

    def sample_distribution(
        self,
        num: int,
        included_variables: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Sample variable distribtuions. The parameter included_variables can be used to
        choose the variables.

        Parameters
        ----------
        num : int
            Number if samples.
        included_variables : Optional[list[str]], optional
            Names of variables included in the calucation. The default is None, which
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
        included_variables: Optional[list[str]] = None,
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
        included_variables : Optional[list[str]], optional
            Names of variables included in the calucation. The default is None, which
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
            Path().external_data("NGPPS_variables.txt"),
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

    def __init__(self, population_id: str):
        """
        Initialize a planet model based on specific population.

        Parameters
        population_id : str
            Name of the population run.
        """
        # code to generate all the avaiable snapshot ages
        self.available_ages = np.array(
            [
                int(j * (10**i))
                for i in range(5, 11)
                for j in range(1, 10 if i != 10 else 2)
            ]
        )

        self.population_id = population_id
        # load systems information
        self.systems = Systems(self.population_id)

    @methodtools.lru_cache(maxsize=256)
    def get_population(self, age: int) -> Population:
        """
        Retrieve population for the given age using the lru_cache for efficiency.

        Parameters
        ----------
        age : int
            Age of the population to retrieve.

        Returns
        ----------
        Population
            An instance of the Population class.
        """
        if age not in self.available_ages:
            raise ValueError("Age does not match any snapshot.")
        return Population(self.population_id, age)

    @methodtools.lru_cache(maxsize=512)
    def get_planet_function(
        self,
        category: str,
        ages: Optional[tuple[int] | int] = None,
        neighbors: int = 3,
        weights: str = "uniform",
        **kwargs: Any,
    ) -> KNeighborsRegressor:
        """
        Calculate KNN interpolation for a population snapshot
        for a given age and category.

        Parameters
        ----------
        category: str
            Category that is matched to system variables.
        ages : Optional[list[int] | int], optional
            A list of snapshot ages to include in the interpolation. The default is
            None, which includes every other age found in available_ages attribute that
            are larger than 20Myr.
        neighbors : int, optional
            Number of neighbors to use in the KNN regression. The default is 3.
        weights : str, optional
            Weight function to use in prediction for KNN regression. he default is
            'uniform'.
        kwargs : dict
            Additional arguments to pass to the KNeighborsRegressor.

        Returns
        ----------
        KNeighborsRegressor
            KNeighborsRegressor model fitted on the population data.
        """
        if ages is None:
            ages = self.available_ages
        elif isinstance(ages, int):
            ages = (ages)
        
        # gather data for considered snapshots
        datasets = []
        for age in ages:
            population = self.get_population(age)
            data = population.match_dataframes(
                category, system_dataframe=self.systems.variables
            )
            data["age"] = age
            datasets.append(data)
        data = pd.concat(datasets)
        
        # Fit the KNN regressor with the data
        knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, **kwargs)
        return knn.fit(data.drop(columns=category), data[category])

    def prediction(
        self,
        categories: str | list,
        variables: pd.DataFrame,
        ages: Optional[tuple[int] | int] = None,
        return_full: bool = False,
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
        variables : pd.DataFrame
            DataFrame of variables to be used in the prediction.
        ages : Optional[tuple[int] | int], optional
            A list of snapshot ages to include in the interpolation. The default is
            None, in that case the relevant ages are inferred from the age column
            of the variables dataframe.
        return_full : bool, optional
            If True, return the full DataFrame (variables + prediction). Otherwise,
            return only the category column (i.e. the prediction).
        kwargs : dict
            Additional arguments to pass to the get_planet_function method.

        Returns
        ----------
        pd.DataFrame
            The predicted values as dataframe. If return_full=True, this includes
            the sample of variables used for the calculation.
        """
        if isinstance(categories, str):
            categories = [categories]
            
        if "age" not in variables.columns:
            raise ValueError("variables dataframe needs to contain column 'ages'.")

        # Sample the system variable distributions
        sample = self.systems.sample_distribution(variables.shape[0])

        # Replace sample variables with the passed variables
        for column in variables.columns:
            sample[column] = variables[column]
            
        # find relevant snapshot ages to be passed to get_planet_function,
        # saves a lot of time if ages are all the same or similar
        if ages is None:
            ages = np.unique(find_closest(sample["age"], self.available_ages))

        # Get the KNN model and predict the category
        prediction_dataframe = sample.copy()
        for category in categories:
            knn = self.get_planet_function(category, ages=tuple(ages),
                                           **kwargs)
            prediction_dataframe[category] = knn.predict(sample)

        if return_full:
            return prediction_dataframe
        else:
            return prediction_dataframe[categories]

if __name__=="__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    category = "Earth"
    pop_id = "ng75"
    samples = int(1e5)
    
    model = PlanetModel(pop_id)
    
    variables = pd.DataFrame(np.linspace(*model.systems.bounds["[Fe/H]"], samples),
                             columns=["[Fe/H]"])
    
    variables["age"] = int(1e+9)
    
    result = model.prediction(category, variables, return_full=True)

    result["planets_binned"] = pd.cut(result[category], bins=5)
    
# sns.pairplot(result.drop(columns=["age", category]), hue="planets_binned", kind="hist")
# sns.heatmap(result.drop(columns=["age", "planets_binned"]).corr(), vmax=1,
#             square=True,annot=True)
# and ridge plots

print("TODO:")
# check and understand if and how interpolation works
# clean up planets implementation in field plot
# adjust setup.py in notebooks to work with new function
# !remember that some metallicities are far outside NGPPS bounds

