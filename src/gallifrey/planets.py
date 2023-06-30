#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:34:58 2023

@author: chris
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, truncnorm

from sklearn.neighbors import KNeighborsRegressor
from gallifrey.data.paths import Path

from functools import lru_cache

from numpy.typing import ArrayLike, NDArray

# things to do now (maybe PlanetModel from now on):
#
# interpolate (KNN)
# from sklearn.neighbors import KNeighborsRegressor
# knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
# e = o.match_systems("Earth")
# y = knn.fit(e.drop(columns="Earth"), e["Earth"])

# marginalise over unwanted parameter (using their distributions)
# to get correct distribution

# implement it in such a way that if function is called once, it gets saved to
# cache (lru_cache), for performance
# properties to cache: {age, category : callable distribution}


class Population:
    """
    NGPPS planet population.

    """

    def __init__(self, population_id: str, age: int) -> None:
        """
        Initialize object, load dataframe and add planet categories.

        Parameters
        ----------
        population_id : str | int
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
            self.distributions[variable_name] = self.truncated_gaussian(variable_name)

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
    
    def variable_probabilities(self,
                               included_variables:Optional[list[str]]=None,
                               ) -> pd.DataFrame:
        """
        Calculate probablities for monte carlo variables from multivariate pdf. 
        Additional parameter, include which distributions to include can be passed
        to multivariate pdf using kwargs.
        
        Parameters
        ----------
        included_variables : Optional[list[str]], optional
            Names of variables included in the calucation. The default is None, which
            includes all variables.

        Returns
        -------
        pd.DataFrame
            Dataframe with system ids and probabilities.

        """
        probabilities = pd.DataFrame()
        probabilities.index = self.variables.index.copy()
        
        variables = self.variables[included_variables]
        probabilities["probability"] = self.multivariate_pdf(variables.T,
                                                             included_variables)
        
        return probabilities
        

    def truncated_gaussian(self, variable_name: str) -> rv_continuous:
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
    
    def multivariate_pdf(self, variable_values: ArrayLike, 
                         included_variables:Optional[list[str]]=None) -> NDArray:
        """
        Calculate the value of the multivariate pdf for some variable values by 
        evaluating and multiplying the individual variable distributions (i.e. assume
        they are independent). The parameter included_variables can be used to choose
        the variables.

        Parameters
        ----------
        variable_values : ArrayLike
            Input values for the distributions (must have same length as distributions 
            dict).
        included_variables : Optional[list[str]], optional
            Names of variables included in the calucation. The default is None, which
            includes all variables.
            
        Returns
        -------
        result : NDArray
            Value of the multivariate pdf.

        """
        if included_variables is None:
            included_variables = self.variable_names
        
        variable_values = np.asarray(variable_values)
        if variable_values.shape[0] != len(included_variables):
            raise ValueError("Shape of variable_values must match number of "
                             "distributions in distribution dict (or length of "
                             "included_variables if given).")
        
        result = 1
        distributions = [self.distributions[var] for var in included_variables]
        for name, distribution, value in zip(included_variables,
                                             distributions, 
                                             variable_values):
            if name in included_variables:
                result *= distribution.pdf(value)
        return result

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
    def __init__(self, population_id: str):
        self.population_id = population_id
        self.systems = Systems(self.population_id)
    
    @lru_cache(maxsize=256)
    def get_population(self, age: int) -> Population:
        population = Population(self.population_id, age)
        return population
    
    @lru_cache(maxsize=512)
    def function(self, age, category, neighbors=5, weights="distance", **kwargs):
        knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, **kwargs)
        
        population = self.get_population(age)
        data = population.match_systems(category, 
                                        system_dataframe=self.systems.variables)
        func = knn.fit(data.drop(columns="Earth").to_numpy(), data["Earth"])
        return func
    
    @lru_cache(maxsize=512)
    def create_(self, age, category, **kwargs):
        func = self.function(age, category, **kwargs)
        multi_pdf = self.systems.multivariate_pdf
        
        def prod(variable_values):
            variable_values = np.asarray(variable_values)
            if variable_values.ndim==1:
                variable_values = variable_values.reshape(1, -1)
            
            return func.predict(variable_values) * multi_pdf(variable_values.T)
        
        return prod

import matplotlib.pyplot as plt
import seaborn as sns

category = "Earth"
pop_id = "ng76"

l = Population(pop_id, int(1e+10))
o = Systems(pop_id)
bounds = list(o.variable_bounds().values())      

pdf_vals = o.variable_probabilities(included_variables=['log_initial_mass', 
                                                        'log_inner_edge', 
                                                        'log_photoevaporation'])
func_vals = l.match_dataframes(category, pdf_vals).prod(axis=1) # product of number of planets and pdf

def create_mgrid(bounds, num_points):
    """Create a meshgrid from a list of bounds and calculate volume of a grid cell.
    Each grid point is at the center of the cell. 
    num_points is a list specifying number of points per dimension.
    """
    if isinstance(num_points, int):
        num_points = [num_points] * len(bounds)

    # Calculate the step size for each dimension
    steps = [(stop - start) / n for (start, stop), n in zip(bounds, num_points)]

    # Adjust the bounds so the points are in the center of the cells
    axes = [np.linspace(start + step / 2, stop - step / 2, n) 
            for (start, stop), step, n in zip(bounds, steps, num_points)]
    
    meshgrid = np.meshgrid(*axes, indexing='ij')

    # Calculate volume of a grid cell
    cell_volume = np.prod(steps)

    return meshgrid, cell_volume



mgrid, cell_volume = create_mgrid(bounds, num_points=[50,50,50,50])
df = pd.DataFrame(np.column_stack([grid.ravel() for grid in mgrid]),
                           columns=o.variable_names)

knn = KNeighborsRegressor(n_neighbors=3, weights="distance")


e = l.match_dataframes(category, o.variables)

func = knn.fit(o.variables[o.variables.index.isin(func_vals.index)],
              func_vals)

# Predict the output for each grid point
df[category] = func.predict(df)
df['planets_binned'] = pd.cut(df[category], bins=4)
sns.pairplot(e, hue=category, kind='hist')
sns.pairplot(df.drop(columns=category).sample(5000), hue='planets_binned', kind='hist')


#func = knn.fit(e.drop(columns=category),e[category])
#df[category] = df[category].round().astype(int)
#sns.pairplot(e, hue=category, kind='hist')
#sns.pairplot(df.sample(5000), hue=category, kind='hist')


# Reshape the predictions to have the same shape as the input grid
#predictions_grid = predictions.reshape(mgrid[0].shape)



# class PlanetModel:
#     """
#     Planet Model.

#     """

#     def __init__(
#         self,
#         planet_formation_time: float = 0.1,
#         cutoff_temperature: float = 7200,
#         occurence_rate: float = 0.5,
#     ) -> None:
#         """
#         Initialize.

#         Parameters
#         -------
#         planet_formation_time : float, optional
#             Estimated time scale for rocky (habitable) planet formation in Gyr. The
#             default is 0.1
#         cutoff_temperature : float, optional
#             Maximum stellar effective temperature for which planets are considered in
#             K.
#             We estimate the occurence rate for more massive stars at 0, since little
#             data is available on occurence rates and habitable zones. The default
#             is 7200.
#         occurence_rate : float, optional
#             Occurence rate of planets in habitable zone below temperature cutoff. The
#             default value is 0.5 (for M and FGK spectral types), from Bryson2020
#             and Hsu2020.

#         """
#         self.planet_formation_time = planet_formation_time  # in Gyr
#         self.cutoff_temperature = cutoff_temperature  # in K
#         self.occurence_rate = occurence_rate

#     @staticmethod
#     def critical_formation_distance(iron_abundance: ArrayLike) -> NDArray:
#         """
#         Critical distance for planet formation based on [Fe/H] estimated by
#         Johnson2012.

#         Parameters
#         ----------
#         fe_fraction : Arraylike
#             Iron abundace [Fe/H] as estimator of metallicity.

#         Returns
#         -------
#         NDArray
#             Estimated maximum distance for planet formation.

#         """

#         return np.power(10, 1.5 + np.asarray(iron_abundance))


# class PlanetOccurenceModel:
#     """
#     Model to assign planets to star particles.
#     """

#     def __init__(
#         self,
#         stellar_model: StellarModel,
#         planet_model: PlanetModel,
#         imf: ChabrierIMF,
#     ) -> None:
#         """
#         Initialize.

#         Parameters
#         ----------
#         stellar_model : StellarModel
#             Stellar model that connects mass to other stellar parameter.
#         planet_model : PlanetModel
#             Planet model that contains relevant planet parameter.
#         imf : ChabrierIMF
#             Stellar initial mass function of the star particles.

#         """

#         self.planet_model = planet_model
#         self.stellar_model = stellar_model
#         self.imf = imf

#     def number_of_planets(
#         self,
#         data: ArepoHDF5Dataset | YTDataContainerDataset,
#         lower_bound: float = 0.08,
#         mass_limits: Optional[NDArray] = None,
#     ) -> NDArray:
#         """
#         Calculate the number of planets associated with the star particles based on
#         the mass of the star particle and including the different cut off effects from
#         stellar lifetime, metallicity, temperature and planet formation time.

#         Parameters
#         ----------
#         data : ArepoHDF5Dataset | YTDataContainerDataset
#             The yt Dataset for the simulation..
#         lower_bound : float, optional
#             Lower bound for the integration of the Chabrier IMF. The default is 0.08.
#         mass_limits : NDArray, optional
#             Mass limits used to choose integration limit from. If not provided,
#             calculated using mass_limits method. Primarely implemented as argument to
#             avoid having to calculate values multiple times when calling
#             number_of_planets and dominant_effect methods. The default is None.

#         Returns
#         -------
#         NDArray
#             Number of planets associated with star particles.

#         """
#         stellar_ages = data["stars", "stellar_age"].value  # in Gyr
#         masses = data["stars", "InitialMass"].to("Msun").value

#         # calculate mass limits based on different effects, if mass limits are not
#         # provided
#         if not mass_limits:
#             mass_limits = self.mass_limits(data)
#         mass_limit = np.amin(mass_limits, axis=1)

#         # based on mass limit, calculate number of eligable stars
#         star_number = self.imf.number_of_stars(
#             masses, upper_bound=mass_limit, lower_bound=lower_bound
#         )

#         # calculate number of planets by multiplying number of eligable stars with
#         # planet occurence rate, set number of planets to 0 if stellar age is below
#         # planet formation time
#         planet_number = np.where(
#             stellar_ages >= self.planet_model.planet_formation_time,
#             self.planet_model.occurence_rate * star_number,
#             0,
#         )
#         return planet_number

#     def dominant_effect(
#         self,
#         data: ArepoHDF5Dataset | YTDataContainerDataset,
#         mass_limits: Optional[NDArray] = None,
#     ) -> NDArray:
#         """
#         Calculate the dominant effect on the number of planets based on the different
#         mass limits and planet formation time by calculating all effects and then
#         choosing the relevant one.
#         Returns array with values between 0 and 3, where the number indicates the
#         dominant effect:
#             0: lifetime
#             1: metallicity
#             2: temperature cut
#             3: planet formation time

#         Parameters
#         ----------
#         data : : ArepoHDF5Dataset | YTDataContainerDataset
#             The yt Dataset for the simulation.
#         mass_limits : NDArray, optional
#             Mass limits used to choose integration limit from. If not provided,
#             calculated using mass_limits method. Primarely implemented as argument to
#             avoid having to calculate values multiple times when calling
#             number_of_planets and dominant_effect methods. The default is None.

#         Returns
#         -------
#         dominant_eff : NDArray
#             Array containing the dominant effect on the planet number.

#         """
#         if not mass_limits:
#             mass_limits = self.mass_limits(data)
#         dominant_eff = np.argmin(mass_limits, axis=1)

#         # add planet formation time effect
#         stellar_ages = data["stars", "stellar_age"].value  # in Gyr
#         dominant_eff[stellar_ages < self.planet_model.planet_formation_time] = 3
#         return dominant_eff

#     def mass_limits(
#         self,
#         data: ArepoHDF5Dataset | YTDataContainerDataset,
#     ) -> NDArray:
#         """
#         Calculate maximum considered stellar mass limits based the different modelled
#         effects.

#         Parameters
#         ----------
#         data : : ArepoHDF5Dataset | YTDataContainerDataset
#             The yt Dataset for the simulation.

#         Returns
#         -------
#         mass_limits : NDArray
#             Array of mass limits (size: [number of effects, number of star
#             particles]).

#         """
#         limit_models = [
#             self.mass_limit_from_lifetime,
#             self.mass_limit_from_metallicity,
#             self.mass_limit_from_temperature,
#         ]
#         mass_limits = np.array([func(data) for func in limit_models]).T
#         return mass_limits

#     def mass_limit_from_lifetime(
#         self,
#         data: ArepoHDF5Dataset | YTDataContainerDataset,
#     ) -> NDArray:
#         """
#         Calculate maximum considered stellar mass based on the lifetime of the star
#         particles.

#         Parameters
#         ----------
#         data : : ArepoHDF5Dataset | YTDataContainerDataset
#             The yt Dataset for the simulation.

#         Returns
#         -------
#         m_from_lifetime : NDArray
#             Array of mass limits.

#         """
#         stellar_ages = data["stars", "stellar_age"].value  # in Gyr
#         m_from_lifetime = self.stellar_model.mass_from_lifetime(stellar_ages)
#         return m_from_lifetime

#     def mass_limit_from_metallicity(
#         self,
#         data: ArepoHDF5Dataset | YTDataContainerDataset,
#     ) -> NDArray:
#         """
#         Calculate maximum considered stellar mass based on the metallicity of the
#         stellar particle, by comparing maximum distance at which planets can
#         form (Johnson2012) and inner edge of planetary HZ (Kopparapu2014).

#         Parameters
#         ----------
#         data : : ArepoHDF5Dataset | YTDataContainerDataset
#             The yt Dataset for the simulation.

#         Returns
#         -------
#         m_from_metallicity : NDArray
#             Array of mass limits.

#         """
#         fe_abundance = data["stars", "[Fe/H]"]

#         # calculate maximum rocky planet formation distance
#         crit_distance = self.planet_model.critical_formation_distance(fe_abundance)

#         # match maximum formation distance to inner habitable zone distance
#         m_from_metallicity = self.stellar_model.inner_HZ_inverse(crit_distance)
#         return m_from_metallicity

#     def mass_limit_from_temperature(
#         self,
#         data: ArepoHDF5Dataset | YTDataContainerDataset,
#     ) -> NDArray:
#         """
#         Calculate maximum considered stellar mass based on the maximum
#         stellar temperature.

#         Parameters
#         ----------
#         data : : ArepoHDF5Dataset | YTDataContainerDataset
#             The yt Dataset for the simulation.

#         Returns
#         -------
#         m_from_temp : NDArray
#             Array of mass limits.

#         """
#         cutoff_mass = self.stellar_model.mass_from_temperature(
#             self.planet_model.cutoff_temperature
#         )
#         m_from_temp = np.full_like(data["stars", "stellar_age"].value, cutoff_mass)
#         return m_from_temp
