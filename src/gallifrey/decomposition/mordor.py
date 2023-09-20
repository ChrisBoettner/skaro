import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import pynbody
from pynbody.snapshot import IndexedSubSnap
from pynbody.snapshot.gadgethdf import GadgetHDFSnap
from unyt import unyt_array, unyt_quantity

from gallifrey.data.paths import Path
from gallifrey.decomposition import decomposition
from gallifrey.halo import Halo
from gallifrey.utilities.logging import logger

# create Logger
logger = logger(__name__)

# put GadgetHDFSnap at front of snaphot priority list
# see https://pynbody.github.io/pynbody/tutorials/configuration.html
snap_class_config = pynbody.config["snap-class-priority"]
gadgethdf5_index = [
    snap_class == GadgetHDFSnap for snap_class in snap_class_config
].index(True)
snap_class_config.insert(0, snap_class_config.pop(gadgethdf5_index))


def gsoft(z: float) -> float:
    """
    Compute the softening length (this is for TNG50).

    Parameters
    ----------
    z : float
        The redshift of the snapshot.

    Returns
    -------
    float
        The TNG50 softening length.

    """
    if z <= 1:
        return 0.288
    else:
        return (0.288 * 2) / (1 + z)


def get_half_mass_radius(galaxy: IndexedSubSnap) -> float:
    """
    Calculate half mass radius of galaxy.

    Parameters
    ----------
    galaxy : IndexedSubSnap
        Cutout of galaxy in simulation snapshot.

    Returns
    -------
    float
        The half-mass radius.

    """
    prof = pynbody.analysis.profile.Profile(galaxy, ndim=3, type="log")
    return np.min(prof["rbins"][prof["mass_enc"] > 0.5 * prof["mass_enc"][-1]])


def calculate_galaxy_decomposition(
    snapshot_path: str,
    halo: Optional[Halo] = None,
    centre: Optional[unyt_array] = None,
    radius: Optional[unyt_quantity] = None,
) -> pd.DataFrame:
    """
    Calculate morphological decomposition using mordor code for a cosmo_sim snapshot for
    a galaxy in the snapshot. Galaxy is spherically cut out of snapshot and then
    processed.
    Returns Dataframe where 'Component' corresponds to the assigned galaxy component
    and 'ParticleIDs' to the evaluated star particles.

    The component assigment is as follows:
        0 - unbound/excluded
        1 - thin/cold disc
        2 - thick/warm disc
        3 - pseudo-bulge
        4 - bulge
        5 - stellar halo

    Parameters
    ----------
    snapshot_path : str
        Path to snapshot that is used as data source.
    halo : Halo, optional
        Halo object that contains information about the halo in question, include
        prescriptions for centre() and radius(). The default is None.
    centre : Optional[unyt_array], optional
        Unyt array describing the centre of the sphere around galaxy which is cut out.
        The default is None, which uses the .centre() method of halo.
    radius : Optional[unyt_quantity], optional
        Unyt array describing the radius of the sphere around galaxy which is cut out.
        The default is None, which uses 0.1 * virial_radius of the halo as determined by
        the .virial_radius() method of halo.

    Returns
    -------
    assignment : pd.DataFrame
        The (sorted) DataFrame containing the decomposition assigments and particle IDs.

    """
    # load data
    if not (snapshot_path[-3]).isdigit():
        logger.warn(
            "WARNING: Last three chars in snapshot_path are not an "
            "integer, which would usually be the snapshot number. This "
            "might be intended, but be catious that the path passed to "
            "pynbody must be the name of the first snapshot file without "
            " trailing .0.hdf5 or similar suffixes. Otherwise only part "
            " of the snapshot will be loaded."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = pynbody.load(snapshot_path)

    # cutout sphere around galaxy of interest
    if ((centre is None) or (radius is None)) and (halo is None):
        raise ValueError("Either halo, or radius AND centre must be given.")

    if centre is None:
        assert isinstance(halo, Halo)
        centre = halo.centre()
    if radius is None:
        assert isinstance(halo, Halo)
        radius = 0.1 * halo.virial_radius()

    centre_value = centre.to("code_length").value
    radius_value = radius.to("code_length").value

    galaxy = data[pynbody.filt.Sphere(radius_value, centre_value)]

    # centre data on sphere centre
    galaxy["pos"] -= centre_value

    # set physical units as default
    galaxy.physical_units()

    # define the softening as the Plummer-equivalent radius if not present
    if "eps" not in galaxy:
        galaxy["eps"] = pynbody.array.SimArray(
            2.8
            * gsoft(galaxy.properties["z"])
            * np.ones_like(galaxy["x"], dtype=galaxy["x"].dtype),
            "kpc",
        )

    # re-scale potential between simulation and pynbody
    # Arepo code: 1/a converts the potential in physical units, the other
    # 1/a accounts for the velocity units km/s * sqrt(a)
    galaxy["phi"] /= galaxy.properties["a"] ** 2

    # center the galaxy
    pynbody.analysis.halo.center(galaxy, wrap=True, mode="hyb")

    # get the half-mass radius of stars
    hmr = get_half_mass_radius(galaxy.s)

    # check if the galaxy has a strange shape through the distance of the centre of
    # mass of the stars
    sc = pynbody.analysis.halo.center(galaxy.s, retcen=True, mode="hyb")
    if np.sqrt(np.sum(sc * sc)) > max(0.5 * hmr, 2.8 * gsoft(galaxy.properties["z"])):
        # if necessary, re-centre on stars
        try:
            pynbody.analysis.halo.center(galaxy.s, mode="hyb")
        except BaseException:
            pynbody.analysis.halo.center(galaxy.s, mode="hyb", cen_size="3 kpc")
        hmr = get_half_mass_radius(galaxy.s)

    # define region where to compute the angular momentum to align the galaxy
    size = max(3 * hmr, 2.8 * gsoft(galaxy.properties["z"]))

    # rotate the galaxy in order to align its angular momentum with the z-axis
    pynbody.analysis.angmom.faceon(
        galaxy.s, disk_size=f"{size} kpc", cen=[0, 0, 0], vcen=[0, 0, 0]
    )

    # launch the decomposition,
    # immediate_mode needed to directly work with Arrays that are compatible with numba
    # read more: https://pynbody.github.io/pynbody/tutorials/performance.html
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with galaxy.immediate_mode:
            decomposition.morph(
                galaxy,
                j_circ_from_r=False,
                LogInterp=False,
                BoundOnly=True,
                Ecut=None,
                jThinMin=0.7,
                mode="cosmo_sim",
                dimcell=None,
            )

    # save decomposition
    assignment = np.array([galaxy.s["morph"], galaxy.s["iord"]])
    assignment_df = pd.DataFrame(assignment.T, columns=["Component", "ParticleIDs"])
    assignment_df = assignment_df.sort_values(by=["Component", "ParticleIDs"])

    return assignment_df


def galaxy_components(
    halo: Halo,
    snapshot_path: Optional[str] = None,
    decomposition_path: Optional[str] = None,
    save: Optional[bool] = True,
    **kwargs: dict[str, Any],
) -> pd.DataFrame:
    """

    Parameters
    ----------
    halo : Halo
        Halo object that contains information about the halo in question, include
        prescriptions for centre() and radius(), as well as descriptive properties of
        the simulation and halo (Halo ID, Simulation ID, resolution, snapshot number).
    snapshot_path : Optional[str], optional
        Path to snapshot that is used as data source. Needed if decomposition file is
        not found. The default is None.
    snapshot_path : Optional[str], optional
        Path to decomposition file, in order to load/save file. The default is None,
        which looks in the data/processed directory.
    save: Optional[bool], optional
        Decide if decomposition file should be saved, if it was just created.
    **kwargs: dict[str, Any]
        Further arguments passed to calculate_galaxy_decomposition.

    Returns
    -------
    assignment : pd.DataFrame
        The (sorted) DataFrame containing the decomposition assigments and particle IDs.

    """
    # choose path where decomposition file is loaded and saved
    if decomposition_path is None:
        path = Path().processed_data(
            f"{halo.resolution}/{halo.sim_id}/"
            f"snapshot_{halo.snapshot}_{halo.halo_id}_decomposition.csv"
        )
    else:
        path = decomposition_path

    try:
        # try to load decomposition
        assignment = pd.read_csv(path)
        logger.info("DECOMPOSITION: Loading decomposition file.")

    except FileNotFoundError:
        # if file isn't found, calculate decomposition
        logger.warn(
            "WARNING: Decomposition file not found. Calculating morphological "
            "decomposition."
        )

        if snapshot_path is None:
            raise ValueError(
                "snapshot_path must be given to calculate morphological "
                "decomposition."
            )

        assignment = calculate_galaxy_decomposition(snapshot_path, halo, **kwargs)
        if save:
            assignment.to_csv(path, index=None)

    return assignment
