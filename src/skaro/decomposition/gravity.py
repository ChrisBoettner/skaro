from typing import Any

import numpy as np
from pynbody import array, units

from skaro.decomposition import kdtree

# Print some useful information
debug = False

########################################################################################


def ConstructGrid(
    f: Any,
    dimcell: Any,
    target: Any = None,
) -> Any:
    """
    ConstructGrid (based on pm in pynbody/gravity/calc.py) builds a density grid and
    Fourier transforms it

    Arguments:
    f -- sim object with potential source. Must include a shape (M,3) array with
    positions, a M array with masses, and a M array with softening radii
    dimcell -- linear size of the cubic cells. Can be a string with units or a scalar

    Keyword arguments:
    target -- optional shape (N,3) array of target positions that have to be included
    into the grid. If None, the grid is built only around the source positions

    Returns:
    pl -- origin of the grid
    kvecs -- shape (ngrid, ngrid, 0.5*ngrid + 1, 3) array with frequencies
    fou_rho_grid -- result of the N-dimensional discrete Fourier Transform
    """

    if isinstance(dimcell, str):
        dimcell = units.Unit(dimcell)

    if units.is_unit_like(dimcell):
        dimcell = float(
            dimcell.in_units(f["pos"].units, **f["pos"].conversion_context())
        )

    if dimcell < 0.25 * min(f["eps"]):
        print(
            "WARNING: dimcell is %.2f %s, lower than 25 percent of the softening "
            " length %.2f %s\n"
            % (dimcell, f["eps"].units, min(f["eps"]), f["eps"].units)
        )

    # Size of the region to evaluate
    if target is not None:
        if target.ndim == 1:
            target = np.array(
                [
                    pos
                    for r in target
                    for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]
                ],
                dtype=float,
            )
        source = np.concatenate((target, f["pos"]))
    else:
        source = f["pos"].view(np.ndarray)

    rm = np.max(np.max(source, axis=0) - np.min(source, axis=0)) + 2 * np.max(f["eps"])

    ngrid = int(rm / dimcell)
    if ngrid % 2:
        ngrid += 1

    if debug:
        print("Ncell = (%d)^3" % (ngrid))

    pm = 0.5 * (np.amax(source, axis=0) + np.amin(source, axis=0))

    pl = pm - 0.5 * ngrid * dimcell
    pu = pm + 0.5 * ngrid * dimcell

    grid, edges = np.histogramdd(
        f["pos"],
        bins=ngrid,
        range=[(pl[0], pu[0]), (pl[1], pu[1]), (pl[2], pu[2])],
        density=False,
        weights=f["mass"],
    )
    grid /= dimcell**3
    fou_rho_grid = np.fft.rfftn(grid)

    freqs = 2 * np.pi * np.fft.fftfreq(ngrid, d=dimcell)

    nhalf = int(0.5 * ngrid)
    kvecs = np.zeros((ngrid, ngrid, nhalf + 1, 3))
    kvecs[:, :, :, 0] = freqs.reshape((1, ngrid, 1, 1))
    kvecs[:, :, :, 1] = freqs.reshape((1, 1, ngrid, 1))
    kvecs[:, :, :, 2] = abs(freqs[: nhalf + 1].reshape((1, 1, 1, nhalf + 1)))

    return pl, kvecs, fou_rho_grid


########################################################################################


def pmesh(
    f: Any,
    dimcell: Any,
    target: Any = None,
) -> Any:
    """
    pmesh constructs a grid to sample the mass distribution f['mass'] and evaluates the
    gravitational potential either for the same mass distribution (f['pos']), or at
    the positions 'target'

    Arguments:
    f -- sim object with potential source. Must include a shape (M,3) array with
    positions, a M array with masses, and a M array with softening radii
    dimcell -- linear size of the cubic cells. Can be a string with units or a scalar

    Keyword arguments:
    target -- (optional) shape (N,3) array of target positions where to evaluate the
    gravitational potential

    Returns:
    numpy array, shape (M) or (N) with the potential energy evaluated either in
    f['pos'], or in target, respectively
    """

    if isinstance(dimcell, str):
        dimcell = units.Unit(dimcell)

    if units.is_unit_like(dimcell):
        dimcell = float(
            dimcell.in_units(f["pos"].units, **f["pos"].conversion_context())
        )

    lower, wnumber, fou_rho_grid = ConstructGrid(f, dimcell, target=target)
    ngrid = wnumber.shape[0]
    shape = [ngrid, ngrid, ngrid]
    wnsq = (wnumber**2).sum(axis=3)
    assert wnsq.shape == fou_rho_grid.shape

    # Softening is not considered here
    fou_pot_grid = np.zeros_like(fou_rho_grid)
    filt = np.where(wnsq != 0)
    fou_pot_grid[filt] = -4 * np.pi * fou_rho_grid[filt] / wnsq[filt]

    pot_grid = np.fft.irfftn(fou_pot_grid, shape)
    if target is None:
        ipos_I = np.array((f["pos"] - lower) / dimcell, dtype=int)
    else:
        ipos_I = np.array((target - lower) / dimcell, dtype=int)

    pot = np.array([pot_grid[x, y, z] for x, y, z in ipos_I])

    return pot


########################################################################################


def GetPot(
    f: Any,
    rxy_target: Any,
    mode: Any = None,
    tree: Any = None,
    theta: Any = 0.5,
    grid: Any = None,
    dimcell: Any = 0.5,
) -> Any:
    """
    GetPot evaluates the gravitational potential generated by 'f' at the radii
    'rxy_target', on the equatorial plane

    Arguments:
    f -- sim object - potential source composed by M particles. Must include a (M,3)
    array with positions, a M array with masses, and a M array with softening radii
    rxy_target -- target radii where to compute the potential

    Keyword arguments:
    mode -- gravitational potential computation mode ['pm', 'tree', else]. See mordor.py
    tree -- KD-tree object. If this parameter is given the potential is computed
    through this KD-tree object, otherwise a new tree is built
    grid-- output of ConstructGrid(). It contains the grid used to sample the 'f'.
    If None, a new grid is built.

    Parameters:
    theta -- cell opening angle used to control force accuracy when mode is 'tree';
    smaller is slower (runtime ~ theta^-3)
    dimcell -- cell size to control the accuracy of the potential evaluation when
    mode is 'pm'

    Returns:
    gravitational potential energy in rxy_target in km^2 s^-2
    """

    # Do four samples like Tipsy does
    rs = np.array(
        [
            pos
            for r in rxy_target
            for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]
        ],
        dtype=float,
    )

    if mode == "tree":
        if tree is None:
            tree = kdtree.ConstructKDTree(
                np.float64(f["pos"]), np.float64(f["mass"]), np.float64(f["eps"])
            )
        potential = kdtree.GetPotentialParallel(np.float64(rs), tree, theta=theta)
    elif mode == "pm":
        if grid is None:
            potential = pmesh(f, dimcell, rs)
        else:
            if isinstance(dimcell, str):
                dimcell = units.Unit(dimcell)
            if units.is_unit_like(dimcell):
                dimcell = float(
                    dimcell.in_units(f["pos"].units, **f["pos"].conversion_context())
                )

            ngrid = grid[1].shape[0]
            shape = [ngrid, ngrid, ngrid]
            wnsq = (grid[1] ** 2).sum(axis=3)
            fou_pot_grid = np.zeros_like(grid[2])
            filt = np.where(wnsq != 0)
            fou_pot_grid[filt] = -4 * np.pi * grid[2][filt] / wnsq[filt]
            pot_grid = np.fft.irfftn(fou_pot_grid, shape)
            ipos_I = np.array((rs - grid[0]) / dimcell, dtype=int)
            potential = np.array([pot_grid[x, y, z] for x, y, z in ipos_I])
    else:
        potential = kdtree.BruteForcePotentialTarget(rs, f["pos"], f["mass"], f["eps"])

    pots = []

    i = 0

    for r in rxy_target:
        # Do four samples
        pot = []
        for _pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
            pot.append(potential[i])
            i = i + 1

        pots.append(np.mean(pot))

    pots_u = units.G * f["mass"].units / f["pos"].units
    pots_sim = array.SimArray(pots, pots_u)
    pots_sim.sim = f.ancestor

    return pots_sim.in_units("km^2 s^-2")


########################################################################################


def GetVcirc(
    f: Any,
    rxy_target: Any,
    mode: Any = None,
    tree: Any = None,
    theta: Any = 0.5,
    grid: Any = None,
    dimcell: Any = 0.5,
) -> Any:
    """
    GetVcirc evaluates the circular velocity on the plane at the radii 'rxy_target',
    due to the particles in 'f'

    Arguments:
    f -- sim object - potential source composed by M particles. Must include a (M,3)
    array with positions, a M array with masses, and a M array with softening radii
    rxy_target -- target radii where to compute the potential

    Keyword arguments:
    mode -- gravitational potential computation mode ['pm', 'tree', else]. See mordor.py
    tree -- KD-tree object. If this parameter is given the potential is computed
    through this KD-tree object, otherwise a new tree is built
    grid-- output of ConstructGrid(). It contains the grid used to sample the
    'f'. If None, a new grid is built.

    Parameters:
    theta -- cell opening angle used to control force accuracy when mode is 'tree';
    smaller is slower (runtime ~ theta^-3)
    dimcell -- cell size to control the accuracy of the potential evaluation
    when mode is 'pm'

    Returns:
    circular velocity in rxy_target in km s^-1
    """

    # Do four samples
    rs = np.array(
        [
            pos
            for r in rxy_target
            for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]
        ],
        dtype=float,
    )

    if mode == "tree":
        if tree is None:
            tree = kdtree.ConstructKDTree(
                np.float64(f["pos"]), np.float64(f["mass"]), np.float64(f["eps"])
            )
        accel = kdtree.GetAccelParallel(
            np.float64(rs), tree, np.float64(f["eps"]), theta=theta
        )
    elif mode == "pm":
        if isinstance(dimcell, str):
            dimcell = units.Unit(dimcell)
        if units.is_unit_like(dimcell):
            dimcell = float(
                dimcell.in_units(f["pos"].units, **f["pos"].conversion_context())
            )

        if grid is None:
            grid = ConstructGrid(f, dimcell, rs)
        ngrid = grid[1].shape[0]
        shape = [ngrid, ngrid, ngrid]
        wnsq = (grid[1] ** 2).sum(axis=3)
        fou_pot_grid = np.zeros_like(grid[2])
        filt = np.where(wnsq != 0)
        fou_pot_grid[filt] = -4 * np.pi * grid[2][filt] / wnsq[filt]
        accel_grid = np.concatenate(
            (
                np.fft.irfftn(-1.0j * grid[1][:, :, :, 0] * fou_pot_grid, shape)[
                    :, :, :, np.newaxis
                ],
                np.fft.irfftn(-1.0j * grid[1][:, :, :, 1] * fou_pot_grid, shape)[
                    :, :, :, np.newaxis
                ],
                np.fft.irfftn(-1.0j * grid[1][:, :, :, 2] * fou_pot_grid, shape)[
                    :, :, :, np.newaxis
                ],
            ),
            axis=3,
        )
        ipos_I = np.array((rs - grid[0]) / dimcell, dtype=int)
        accel = np.array([accel_grid[x, y, z, :] for x, y, z in ipos_I])
    else:
        accel = kdtree.BruteForceAccelTarget(rs, f["pos"], f["mass"], f["eps"])

    vels = []

    i = 0
    for r in rxy_target:
        r_acc_r = []
        for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
            r_acc_r.append(np.dot(-accel[i, :], pos))
            i = i + 1

        vel2 = np.mean(r_acc_r)
        if vel2 > 0:
            vel = vel2**0.5
        else:
            vel = 0

        vels.append(vel)

    vels_u = (units.G * f["mass"].units / f["pos"].units) ** (1, 2)
    vels_sim = array.SimArray(vels, vels_u)
    vels_sim.sim = f.ancestor

    return vels_sim.in_units("km s^-1")
