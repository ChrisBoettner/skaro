import numpy as np
import pynbody
from pynbody.snapshot.gadgethdf import GadgetHDFSnap
from pynbody.snapshot import IndexedSubSnap

from gallifrey.decomposition import decomposition

# put GadgetHDFSnap at front of snaphot priority list
snap_class_config = pynbody.config["snap-class-priority"]
gadgethdf5_index = [snap_class == GadgetHDFSnap 
                    for snap_class in snap_class_config].index(True)
snap_class_config.insert(0, snap_class_config.pop(gadgethdf5_index))

def gsoft(z:float)-> float:
    '''
    Compute the softening length (this is for TNG50).

    Parameters
    ----------
    z : float
        The redshift of the snapshot.

    Returns
    -------
    float
        The TNG50 softening length.

    '''
    if z <= 1:
        return 0.288
    elif z > 1:
        return (0.288 * 2) / (1 + z)

def get_half_mass_radius(galaxy: IndexedSubSnap) -> float:
    '''
    Calculate half mass radius of galaxy.

    Parameters
    ----------
    galaxy : IndexedSubSnap
        Cutout of galaxy in simulation snapshot.

    Returns
    -------
    float
        The half-mass radius.

    '''
    prof = pynbody.analysis.profile.Profile(galaxy, ndim=3, type="log")
    return np.min(prof["rbins"][prof["mass_enc"] > 0.5 * prof["mass_enc"][-1]])


def galaxy_components(file_path, halo, centre=None, radius=None):
    # load data
    data = pynbody.load(file_path)
    
    # cutout sphere around galaxy of interest
    if centre is None:
        centre = halo.centre()
    if radius is None:
        radius = 0.1*halo.virial_radius()   
        
    centre_value = centre.to("code_length").value
    radius_value = radius.to("code_length").value
    
    galaxy = data[pynbody.filt.Sphere(radius_value, centre_value)]
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
    pynbody.analysis.halo.center(galaxy, wrap=True, mode='hyb')
  
	# get the half-mass radius of stars
    hmr = get_half_mass_radius(galaxy.s)
    
  	# check if the galaxy has a strange shape through the distance of the centre of 
    # mass of the stars
    sc = pynbody.analysis.halo.center(galaxy.s, retcen=True, mode='hyb')
    if np.sqrt(np.sum(sc*sc)) > max(0.5*hmr, 2.8*gsoft(galaxy.properties['z'])):
  	  # re-centre on stars, if necessary
      try:
          pynbody.analysis.halo.center(galaxy.s, mode='hyb')
      except:
          pynbody.analysis.halo.center(galaxy.s, mode='hyb', cen_size='3 kpc')
      hmr = get_half_mass_radius(galaxy.s)

    # define region where to compute the angular momentum to align the galaxy
    size = max(3 * hmr, 2.8 * gsoft(galaxy.properties["z"]))

    # rotate the galaxy in order to align its angular momentum with the z-axis
    pynbody.analysis.angmom.faceon(
        galaxy.s, disk_size=f"{size} kpc", cen=[0, 0, 0], vcen=[0, 0, 0])
    
    # launch the decomposition
    with galaxy.immediate_mode: 
        profiles = decomposition.morph(
            galaxy,
            j_circ_from_r=False,
            LogInterp=False,
            BoundOnly=True,
            Ecut=None,
            jThinMin=0.7,
            mode="cosmo_sim",
            dimcell=None,)

    breakpoint()