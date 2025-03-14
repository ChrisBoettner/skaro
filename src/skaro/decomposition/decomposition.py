from typing import Any

import numpy as np
import scipy.interpolate as interp
from pynbody import filt
from pynbody.analysis import profile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from skaro.decomposition import gravity
from skaro.decomposition.kdtree import ConstructKDTree

# Print some useful information
debug = False

########################################################################################


def FindMin(q: Any, m_E: Any, M_E: Any, bins: Any, sigma: Any) -> Any:
    """
    It looks for the minima in the distribution of energies q in the interval
    [m_E; M_E]

    Arguments:
    q -- N array with energy of particles
    m_E -- lower bound of the interval where to look for the minima
    M_E -- upper bound of the interval where to look for the minima
    bins -- bins in the interval used to bin the distribution q
    sigma -- standard deviation of smoothing kernel, higher values lead to more
    smoothing.

    Returns:
    array with the position of the minima of the distribution q, along with their
    values (NB: The values do depend on nbins)
    """

    # Minimum number of particles to perform a reliable Jcirc decomposition
    MinPart: Any = max(1000, 0.01 * len(q))
    arr = q[(q >= m_E) * (q <= M_E)]
    # Build the histogram
    hist = np.histogram(arr, bins=bins)
    smoothed_hist = gaussian_filter1d(hist[0], sigma)

    # Find the minima
    id_E: Any = find_peaks(-smoothed_hist)[0]
    energies = {id: ((bins[1:] + bins[:-1]) / 2)[id] for id in id_E}

    # exclude minima where one of the one of the components would have to few particles
    for id, energy in energies.items():
        flag = (len(arr[arr > energy]) < MinPart) or (len(arr[arr < energy]) < MinPart)
        if flag:
            energies.pop(id)

    if debug:
        print("Survived Energies:", list(energies.values()))
        from matplotlib import pyplot as plt

        plt.hist(arr, bins=bins, color="grey")
        plt.plot(
            (bins[1:] + bins[:-1]) / 2,
            smoothed_hist,
            color="red",
        )
        plt.scatter(
            list(energies.values()),
            smoothed_hist[list(energies.keys())],
            color="black",
            s=50,
            zorder=10,
        )
        plt.ylabel("Count")
        plt.xlabel("E")
        plt.show()

    # Return the central position of the bins
    return np.array([*energies.values()]), hist[0][list(energies.keys())]


########################################################################################


def RefineMin(
    q: Any,
    Vmin: Any,
    D: Any,
    Dmin: Any,
    shrink: Any,
) -> Any:
    """
    It recursively refines a minimum Vmin of the energy distribution q, within the
    interval of size D, centred on Vmin. Each time the
    refinement reduce the interval size of shrink as long as D>Dmin

    Arguments:
    q -- M array with energy of particles
    Vmin -- value of the minimum (Energy) to refine
    D -- initial size of the energy interval around Vmin
    Dmin -- shortest interval size allowed
    shrink -- factor to reduce the interval size, at each cycle

    Returns:
    refined position of Vmin
    """

    if D <= Dmin:
        m_E = Vmin - 0.5 * D
        M_E = Vmin + 0.5 * D
        arr = q[(q >= m_E) * (q <= M_E)]
        Vmin = np.median(arr)

    while D > Dmin:
        m_E = max(Vmin - 1.5 * D, q.min() + D)
        M_E = Vmin + 1.5 * D
        D = D / shrink
        arr = q[(q >= m_E) * (q <= M_E)]
        hist = np.histogram(arr, bins=np.arange(m_E, M_E, D))
        pid = np.argmin(hist[0])
        # Get the energy as the median within the selected bin
        Vmin = np.median(arr[(arr >= hist[1][pid]) * (arr <= (hist[1][pid + 1]))])

        if debug:
            from matplotlib import pyplot as plt

            print("Refining:", Vmin)
            plt.bar(hist[1][:-1], hist[0], hist[1][1:] - hist[1][:-1], align="edge")
            plt.show()

    return Vmin


########################################################################################


def morph(
    gal: Any,
    j_circ_from_r: Any = False,
    LogInterp: Any = False,
    BoundOnly: Any = False,
    refine: Any = False,
    Ecut: Any = None,
    jThinMin: Any = 0.7,
    mode: Any = "tree",
    theta: Any = 0.5,
    sigma: Any = 2,
    dimcell: Any = "1 kpc",
    DumpProb: Any = False,
) -> Any:
    """
    This tool provide a kinematic decomposition of stellar particles in gal.
    A SimArray 'morph' is generated for the stars and an integer (0-5) is assigned to
    each particle in order to identify its morpho-kinematic component:

    0 -- unbound/excluded
    1 -- thin/cold disc
    2 -- thick/warm disc
    3 -- pseudo-bulge
    4 -- bulge
    5 -- stellar halo

    Arguments:
    gal -- sim object - the galaxy to work on

    Keyword arguments:
    Ecut -- the energy boundary between bulge/pseudobulge and halo/thick disc.
    j_circ_from_r -- the circular angular momentum is computed as a function of radius,
    rather than as a function of orbital energy
    LogInterp -- use a logarithmic interpolation/extrapolation, instead of a linear one,
    to evaluate the circular angular momentum
    refine -- If True, try to refine energy minimum found. The default is False.
    BoundOnly -- enable it to exclude those particles with E>=0, |jz/jcirc|>=1.5>=1.5,
    |jp/jcirc|>=1.5; see Zana et al. 2022
    mode -- choose amongst 'direct', 'pm', 'tree', 'cosmo_sim', 'iso_sim', or
    'auxiliary'. If mode is 'cosmo_sim', an offset is applied
    DumpProb -- particles are assigned to the bulge or the halo according to a
    probabilistic scheme. If DumpProb is enabled, an additional SimArray is created
    and filled with a float for each stellar particle, where the integer part refers to
    the alternative morphological component (not assigned) and the decimal part to the
    probability of assignement. if prob=0, the particle has been assigned to the
    only possible component

    Parameters:
    j_disc_min -- minimum angular momentum (in terms of the circular angular momentum)
    that a particle must have to be part of the 'thin disc' component. Default is 0.7
    theta -- opening angle of the tree to tune force computation accuracy when mode
    is 'tree'. Default is 0.5
    sigma -- Standard devation used for smoothing the energy histogram when searching
    for minina. Default is 2
    dimcell -- cubic cell side. When mode is 'pm'. Default is dimcell=1 kpc

    Returns:
    profile object of the equatorial disc to test the success of the decomposition
    """

    # Additional parameters to find Ecut:
    m_bin = 80  # Minimum number of bins
    M_bin = 400  # Maximum number of bins
    shrink = 2  # Refinement factor to find Ecut (see RefineMin)
    # StartNbins = 25  # Starting bin number
    toll = 1.5  # Bin range: increase it to find more minima. With higher values,
    # it may not converge
    Emin = (
        -0.9
    )  # If more minima are found, those under Emin are discarded (in units of Emax)
    Mmin = 0.05  # Minimum mass required for the bound component if more minima are
    # found (fraction)

    # Minimum amount of bound particles to perform the analysis if 'BoundOnly' is
    # defined. It produces only a Warning signal
    CutPart = 5.0e3

    # -------

    # Evaluate kinetic and potential energies
    ke = gal["ke"]
    pe = gal["phi"]

    # Make sure the units are same
    gal["phi"].convert_units(ke.units)

    te = ke + pe
    gal["te"] = te
    te_star = gal.s["te"]
    # Maximum energy among stellar particles
    te_max = te_star.max()

    if BoundOnly:
        E_filt = te_star < 0

    # Offset the energy distributions
    gal["te"] -= te_max

    # To produce the disc rotation curve, filter the particles in a disc
    d = gal[filt.Disc("1 Mpc", gal["eps"].min())]
    M_r = 1.01 * np.max(gal["r"])
    m_r = 0.99 * np.min(gal["r"][gal["r"] > 0])
    pbins = np.logspace(np.log10(m_r), np.log10(M_r), 100, endpoint=True)
    disc_prof = profile.Profile(d, bins=pbins, type="log")

    # -------

    # The whole halo (dm, gas, stars) has to be evaluated to compute the correct
    # rotation curve
    tree_obj = grid_obj = None

    if mode == "tree":
        tree_obj = ConstructKDTree(
            np.float64(gal["pos"]), np.float64(gal["mass"]), np.float64(gal["eps"])
        )
    elif mode == "pm":
        grid_obj = gravity.ConstructGrid(gal, dimcell, disc_prof["rbins"])

    # Compute the potential energy on the midplane
    disc_prof._profiles["pot"] = gravity.GetPot(
        gal,
        disc_prof["rbins"],
        mode=mode,
        tree=tree_obj,
        theta=theta,
        grid=grid_obj,
        dimcell=dimcell,
    )

    # if the potential comes from a cosmological simulations, consider an offset with
    # the potential recalculated in isolation
    # NOTE: CHANGED BY ME, SHOULD WORK BUT NO PROMISES
    if mode == "cosmo_sim":
        offset = disc_prof._profiles["pot"] - disc_prof["phi"]
        # roff = (disc_prof["rbins"][-1] - disc_prof["rbins"][0]) * 0.2
        # proff = disc_prof["rbins"] - roff
        # boff = int(np.argwhere(np.abs(proff) == np.min(np.abs(proff))))
        # moff = np.nan
        # while np.isnan(moff):
        #    moff = np.nanmean(offset[boff - 1 : boff + 1])
        #    boff += 1
        moff = np.nanmedian(offset)
        if np.isfinite(moff):
            disc_prof._profiles["pot"] -= moff
        else:
            raise ValueError(
                "Couldn't calculate Potential offset. Maybe try with mode=='iso_sim'."
            )

    # Offset the midplane potential as for the te array
    disc_prof["pot"] -= te_max

    # -------

    # Evaluate the proper rotation curve
    disc_prof._profiles["v_circ"] = gravity.GetVcirc(
        gal,
        disc_prof["rbins"],
        mode=mode,
        tree=tree_obj,
        theta=theta,
        grid=grid_obj,
        dimcell=dimcell,
    )

    # -------

    # Compute the circular angular momentum for all the particles
    if j_circ_from_r:
        if LogInterp:
            j_from_r = interp.interp1d(
                np.log10(disc_prof["rbins"]),
                np.log10(disc_prof["j_circ"]),
                fill_value="extrapolate",
                bounds_error=False,
            )
            gal["j_circ"] = 10 ** j_from_r(np.log10(gal["r"]))
        else:
            j_from_r = interp.interp1d(
                disc_prof["rbins"],
                disc_prof["j_circ"],
                fill_value="extrapolate",
                bounds_error=False,
            )
            gal["j_circ"] = j_from_r(gal["r"])

    else:
        if LogInterp:
            j_from_E = interp.interp1d(
                np.log10(-disc_prof["E_circ"].in_units(ke.units))[::-1],
                np.log10(disc_prof["j_circ"])[::-1],
                fill_value="extrapolate",
                bounds_error=False,
            )
            gal["j_circ"] = 10 ** j_from_E(np.log10(-gal["te"]))
        else:
            j_from_E = interp.interp1d(
                disc_prof["E_circ"].in_units(ke.units),
                disc_prof["j_circ"],
                fill_value="extrapolate",
                bounds_error=False,
            )
            gal["j_circ"] = j_from_E(gal["te"])

    # Force nearly unbound particles into the spheroid either by setting their circular
    # angular momentum to infinity
    gal["j_circ"][np.where(gal["te"] > disc_prof["E_circ"].max())] = np.inf
    # or by allocating them into the last bin of the circular angular momentum
    # gal['j_circ'][np.where(gal['te'] > disc_prof['E_circ'].max())] = (
    # disc_prof['j_circ'][-1])

    # Handle those few particles with E<min(E_circ) for numerical fluctuations
    gal["j_circ"][np.where(gal["te"] < disc_prof["E_circ"].min())] = disc_prof[
        "j_circ"
    ][0]

    # -------

    # Compute the circularity
    gal["jz_by_jzcirc"] = gal["j"][:, 2] / gal["j_circ"]
    g_star = gal.star

    # --------------------------------------------------------------------------------------------------------

    if "morph" not in g_star:
        g_star._create_array("morph", dtype=int)
        g_star["morph"] = 0
    if DumpProb and "prob" not in g_star:
        g_star._create_array("prob", dtype=float)
        g_star["prob"] = 0

    # Exclude particles with E>=0 and or with too much vertical/parallel angular
    # momentum
    if BoundOnly:
        bound = (
            E_filt
            * (np.abs(g_star["jz_by_jzcirc"]) < 1.5)
            * (
                np.array(g_star["j2"] - g_star["jz"] ** 2) ** 0.5
                < 1.5 * g_star["j_circ"]
            )
        )
        g_star = g_star[bound]

        if len(g_star) < CutPart:
            print("WARNING: too few bound particles (%d)\n" % (len(g_star)))

    JzJcirc = g_star["jz_by_jzcirc"]
    mass = g_star["mass"]
    # Energy as a function of the energy of the most bound particle
    te = g_star["te"] / np.abs(g_star["te"]).max()

    # ----------------------------------------------------------------------------------

    if Ecut is None:
        # Fix the number of bin as a function of Npart
        NbinMax = max(min(int(0.5 * np.sqrt(len(te))), M_bin), m_bin)

        if debug:
            print("First round - q90")

        # This is to exclude the outer tail of bound particles
        M_E: float = np.quantile(te, 0.9)
        m_E: float = np.min(te)

        # find energy minima, but only for particles that are not in thin disk
        bins = np.histogram_bin_edges(
            te[JzJcirc < jThinMin], bins="fd", range=(m_E, M_E)
        )
        Ecut, E_val = FindMin(te[JzJcirc < jThinMin], m_E, M_E, bins, sigma=sigma)
        # If no minimum is found or the only minimum is too close to -1 (Maybe a GC?)
        if len(Ecut) == 0 or (len(Ecut) == 1 and Ecut < Emin):
            if debug:
                print("RE-evaluation ")

            M_E = np.max(te)
            bins = np.histogram_bin_edges(te, bins="fd", range=(m_E, M_E))
            Ecut, E_val = FindMin(te[JzJcirc < jThinMin], m_E, M_E, bins, sigma=sigma)
            Ecut = Ecut

        nbins = len(bins) - 1
        # If one or none minima are found
        if len(Ecut) <= 1:
            D = (M_E - m_E) / float(nbins)
            # Avoid the following loop
            nbins = NbinMax + 1
        else:
            D = (M_E - m_E) / float(nbins)
            lb = Ecut - (toll * D)
            rb = Ecut + (toll * D)

        # -------

        if debug:
            print("Energy before loop:", Ecut)

        if refine:
            while nbins < NbinMax:
                nbins = shrink * nbins
                D = D / shrink
                bins = np.histogram_bin_edges(te, bins=nbins, range=(m_E, M_E))
                pos_E_refined, val_refined = FindMin(
                    te[JzJcirc < jThinMin], m_E, M_E, bins, sigma=sigma
                )
                EcutTEMP = []
                E_valTEMP = []
                for i, v in enumerate(E_val):
                    pTEMP = pos_E_refined[
                        (pos_E_refined <= rb[i]) * (pos_E_refined >= lb[i])
                    ]
                    vTEMP = val_refined[
                        (pos_E_refined <= rb[i]) * (pos_E_refined >= lb[i])
                    ]
                    if len(pTEMP) > 0:
                        # A rifened position and value for each original minimum is
                        # stored.
                        # The value of the minima is summed to the original ones to
                        # avoid strange local minima
                        EcutTEMP.append(pTEMP[np.argmin(vTEMP)])
                        E_valTEMP.append(v + np.min(vTEMP))

                Ecut = np.array(EcutTEMP)
                E_val = np.array(E_valTEMP)

                if debug:
                    print("Energy in loop", Ecut)

                if len(Ecut) <= 1:
                    break

                lb = Ecut - (toll * D)
                rb = Ecut + (toll * D)

        # -------

        # If no energy cut is found
        if len(Ecut) == 0:
            Ecut = 0
        else:
            # Try to avoid strange nuclear minima with low mass if there are better
            # alternatives
            rel_filt = [
                bool((np.sum(mass[te < E]) / np.sum(mass) >= Mmin) + (E >= Emin))
                for E in Ecut
            ]
            if len(Ecut[rel_filt]) == 0:
                Ecut = Ecut[np.argmin(E_val)]
            else:
                Ecut = Ecut[rel_filt][np.argmin(E_val[rel_filt])]

            if debug:
                print("Refinement:")
            Ecut = RefineMin(te, Ecut, D, (M_E - m_E) / NbinMax, shrink)

    # print("Ecut = ", Ecut)

    # ----------------------------------------------------------------------------------

    E_low = te <= Ecut

    # Thin/cold disc:
    thin = np.where(JzJcirc > jThinMin)
    g_star["morph", thin[0]] = 1

    if DumpProb:
        g_star["prob", thin[0]] = 0

    # Pseudo-bulge:
    pbulge = np.where(E_low * (JzJcirc <= jThinMin))
    g_star["morph", pbulge[0]] = 3

    if DumpProb:
        g_star["prob", pbulge[0]] = 0

    # Bulge:
    dist_low = np.histogram(
        JzJcirc[E_low],
        bins=np.arange(np.nanmin(JzJcirc[E_low]), np.nanmax(JzJcirc[E_low]), 0.01),
        weights=mass[E_low],
    )

    if debug:
        from matplotlib import pyplot as plt

        plt.bar(
            dist_low[1][:-1],
            dist_low[0],
            dist_low[1][1:] - dist_low[1][:-1],
            align="edge",
        )
        plt.ylabel("Count")
        plt.xlabel("jz/jcirc")
        plt.show()

    PositiveCirc = JzJcirc[E_low] > 0
    c = 0.5 * (dist_low[1][1:] + dist_low[1][:-1])
    Bspl = interp.UnivariateSpline(c, dist_low[0], s=0)
    yBspl = Bspl(-JzJcirc[E_low][PositiveCirc])
    # The seed is fixed for reproducibility
    np.random.seed(42)
    p = yBspl / Bspl(JzJcirc[E_low][PositiveCirc])
    # [0;1)
    ra = np.random.random(len(yBspl))
    id_pos = np.where(E_low * (JzJcirc > 0))[0]
    id_b = id_pos[ra <= p]
    bulge = np.where((te <= Ecut) * (JzJcirc <= 0))
    # Some particles of the pseudobulge and thin disc are re-assigned
    bulge = np.concatenate((bulge[0], id_b))
    g_star["morph", bulge] = 4

    if DumpProb:
        # Ids in the disc area
        id_d = np.where(E_low * JzJcirc > jThinMin)[0]
        # ids in the pseudobulge area
        id_pb = np.where(E_low * (JzJcirc < jThinMin) * (JzJcirc > 0))[0]

        # Not selected as bulge particles
        g_star["prob"][id_pos[ra > p]] = 4 + p[ra > p]
        # Selected as bulge particles but may be disc
        mask = np.in1d(id_b, id_d, assume_unique=True)
        g_star["prob"][id_b[mask]] = 1 + 1 - p[ra <= p][mask]
        # Selected as bulge particle but may be pbulge
        mask = np.in1d(id_b, id_pb, assume_unique=True)
        g_star["prob"][id_b[mask]] = 3 + 1 - p[ra <= p][mask]

    if Ecut < 0:
        # Thick/warm disc
        thick = np.where((te > Ecut) * (JzJcirc < jThinMin))
        g_star["morph", thick[0]] = 2

        if DumpProb:
            g_star["prob", thick[0]] = 0

        # Halo
        dist_high = np.histogram(
            JzJcirc[~E_low],
            bins=np.arange(
                np.nanmin(JzJcirc[~E_low]), np.nanmax(JzJcirc[~E_low]), 0.01
            ),
            weights=mass[~E_low],
        )

        if debug:
            plt.bar(
                dist_high[1][:-1],
                dist_high[0],
                dist_high[1][1:] - dist_high[1][:-1],
                align="edge",
            )
            plt.ylabel("Count")
            plt.xlabel("jz/jcirc")
            plt.show()

        PositiveCirc = JzJcirc[~E_low] > 0
        c = 0.5 * (dist_high[1][1:] + dist_high[1][:-1])
        Hspl = interp.UnivariateSpline(c, dist_high[0], s=0)
        yHspl = Hspl(-JzJcirc[~E_low][PositiveCirc])

        # Ratio between negative tail and positive part
        p = yHspl / Hspl(JzJcirc[~E_low][PositiveCirc])
        ra = np.random.random(len(yHspl))
        id_pos = np.where((~E_low) * (JzJcirc > 0))[0]
        id_h = id_pos[ra <= p]

        halo = np.where((te > Ecut) * (JzJcirc <= 0))
        # Some particles of the thick disc are re-assigned
        halo = np.concatenate((halo[0], id_h))
        g_star["morph", halo] = 5

        if DumpProb:
            # ids in the thick disc area
            id_t = np.where((~E_low) * (JzJcirc < jThinMin) * (JzJcirc > 0))[0]

            # Not selected as halo particles
            g_star["prob"][id_pos[ra > p]] = 5 + p[ra > p]

            # Selected as halo particles but may be disc
            mask = np.in1d(id_h, id_d, assume_unique=True)
            g_star["prob"][id_h[mask]] = 1 + 1 - p[ra <= p][mask]
            # Selected as halo particle but may be thick disc
            mask = np.in1d(id_h, id_t, assume_unique=True)
            g_star["prob"][id_h[mask]] = 2 + 1 - p[ra <= p][mask]

    # Remove the offset before returning the profiles
    disc_prof["pot"] += te_max
    gal["te"] += te_max

    return disc_prof
