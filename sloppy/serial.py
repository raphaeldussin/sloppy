import numpy as np
from numba import njit
from scipy.spatial import KDTree
from warnings import warn


def compute_cell_topo_stats(
    lon_c, lat_c, lon_src, lat_src, data_src, method="median", compute_res=True
):
    """Compute the stats (h median/mean, hmin, hmax, h2) for one grid cell

    Parameters
    ----------
    lon_c : np.ndarray
        2x2 array containing the longitudes of all 4 corners of the cell
    lat_c : np.ndarray
        2x2 array containing the latitudes of all 4 corners of the cell
    lon_src : np.ndarray
        2d array (ny, nx) containing the longitudes of the source topo data
    lat_src : np.ndarray
        2d array (ny, nx) containing the latitudes of the source topo data
    data_src : np.ndarray
        2d array (ny, nx) containing the the source topo data
    method (optional) : string
        estimation of cell bathy (median/mean). Defaults to median.
    Returns
    -------
    np.array containing
        [cell estimated depth, minimum depth in cell, maximum depth in cell, residual of plane fit, number of source points in cell]

    """

    # this wrapper is needed to handle edge cases (e.g. no points found,...)

    out = 1.0e20 * np.ones((5))
    if len(data_src.flatten()) < 1:
        out[4] = 0.0
    else:
        dout, dmin, dmax, residual2, npts = compute_cell_topo_stats_parallelogram(
            lon_c,
            lat_c,
            lon_src,
            lat_src,
            data_src,
            method=method,
            compute_res=compute_res,
        )

        out[0] = dout
        out[1] = dmin
        out[2] = dmax
        out[3] = residual2
        out[4] = float(npts)

    return out


@njit()
def compute_cell_topo_stats_parallelogram(
    lon_c,
    lat_c,
    lon_src,
    lat_src,
    data_src,
    method="median",
    eps=1e-16,
    compute_res=True,
):
    """Compute the stats (h median/mean, hmin, hmax, h2) for one grid cell

    Parameters
    ----------
    lon_c : np.ndarray
        2x2 array containing the longitudes of all 4 corners of the cell
    lat_c : np.ndarray
        2x2 array containing the latitudes of all 4 corners of the cell
    lon_src : np.ndarray
        2d array (ny, nx) containing the longitudes of the source topo data
    lat_src : np.ndarray
        2d array (ny, nx) containing the latitudes of the source topo data
    data_src : np.ndarray
        2d array (ny, nx) containing the the source topo data
    method (optional) : string
        estimation of cell bathy (median/mean). Defaults to median.
    Returns
    -------
    np.array containing
        [cell estimated depth, minimum depth in cell, maximum depth in cell, residual of plane fit, number of source points in cell]

    """

    ny, nx = data_src.shape[-2:]
    coords_in_cell = []
    data_src_in_cell = []

    for jj in range(ny):
        for ji in range(nx):

            # test if point falls in the cell, defined by 4 segments
            lat_lower_bnd = lat_c[0, 0] + (lon_src[jj, ji] - lon_c[0, 0]) * (
                (lat_c[0, -1] - lat_c[0, 0])
                / np.maximum(lon_c[0, -1] - lon_c[0, 0], eps)
            )

            if lat_src[jj, ji] >= lat_lower_bnd:
                lon_right_bnd = lon_c[0, -1] + (lat_src[jj, ji] - lat_c[0, -1]) * (
                    (lon_c[-1, -1] - lon_c[0, -1])
                    / np.maximum(lat_c[-1, -1] - lat_c[0, -1], eps)
                )

                if lon_src[jj, ji] <= lon_right_bnd:
                    lat_upper_bnd = lat_c[-1, 0] + (lon_src[jj, ji] - lon_c[-1, 0]) * (
                        (lat_c[-1, -1] - lat_c[-1, 0])
                        / np.maximum(lon_c[-1, -1] - lon_c[-1, 0], eps)
                    )

                    if lat_src[jj, ji] <= lat_upper_bnd:
                        lon_left_bnd = lon_c[0, 0] + (lat_src[jj, ji] - lat_c[0, 0]) * (
                            (lon_c[-1, 0] - lon_c[0, 0])
                            / np.maximum(lat_c[-1, 0] - lat_c[0, 0], eps)
                        )

                        if lon_src[jj, ji] >= lon_left_bnd:
                            # all checks passed, adding point to list
                            coords_in_cell.append(
                                [lon_src[jj, ji], lat_src[jj, ji], 1.0]
                            )
                            data_src_in_cell.append(1.0 * data_src[jj, ji])

    # count number of hits in cell
    npts = len(data_src_in_cell)

    if npts == 0:
        dout = 1.0e20
        dmin = 1.0e20
        dmax = 1.0e20
    else:
        A = np.array(coords_in_cell)
        d = np.array(data_src_in_cell)
        dmin = d.min()
        dmax = d.max()
        if method == "median":
            dout = np.median(d)
        elif method == "mean":
            dout = d.mean()  # assume identical weights for source cells to start

    residual2 = np.zeros((1))
    if compute_res:
        if npts >= 3:  # we need at least 3 points to fit a plane
            # plane fitting to get the residuals
            fitcoefs, residual2, _, _ = np.linalg.lstsq(A, d)
            if residual2 is None:
                residual2 = 1.0e20 * np.ones((1))
            if np.isnan(residual2):
                residual2 = 1.0e20 * np.ones((1))

    return dout, dmin, dmax, residual2[0], npts


def find_nearest_point(lon_src, lat_src, lon_tgt, lat_tgt, tree=None, is_carth=False):
    """Function to find the nearest point on the source grid given a
    value on the target grid

    Parameters
    ----------
    lon_src : np.ndarray
        1-D or 2-D array of longitude values in the source grid
    lat_src :
        1-D or 2-D array of latitude values in the source grid
    lon_tgt : float
        Longitude value to find on the source grid
    lat_tgt : float
        Latitude
    tree : scipy.Kdtree object, optional
        Precalculated kdtree object for 2D coordinates, by default None

    Returns
    -------
    np.ndarray
         x,y indices

    """

    if not is_carth:
        # standardize longitude to positive-definite values
        lon_src = np.mod(lon_src + 360, 360)
        lon_tgt = np.mod(lon_tgt + 360, 360)

    # use a kdtree to find indicies if source coordinates are 2-dimensional
    if len(lon_src.shape) == 2:
        if tree is None:
            raise ValueError("you must supply a kdtree")
        _, ravel_index = tree.query([lon_tgt, lat_tgt], k=1)
        indy, indx = np.unravel_index(ravel_index, lon_src.shape)

    # use a simple locator if source coordinates are vectors
    elif len(lon_src.shape) == 1:
        indx = (np.abs(lon_src - lon_tgt)).argmin()
        indy = (np.abs(lat_src - lat_tgt)).argmin()

    return indx, indy


def find_geographical_bounds(lon_c, lat_c):
    """find the bounding box"""

    # this is the simple case where there is no periodicity to worry about
    lonmin = lon_c.min()
    lonmax = lon_c.max()
    latmin = lat_c.min()
    latmax = lat_c.max()

    return lonmin, lonmax, latmin, latmax


def correct_for_poles_j_indices(jmin, jmax):
    """make sure that jmin < jmax, this can happen in polar regions"""

    if jmin > jmax:
        warn("flipping j indices")

    jmin_corrected = jmax if jmin > jmax else jmin
    jmax_corrected = jmin if jmin > jmax else jmax

    return jmin_corrected, jmax_corrected


def correct_for_periodicity(imin, imax):
    """return updated indices if subset is across disjoint regions
    by using periodicity of longitude to roll array

    Args:
        imin (int): i-index of SW corner
        imax (int): i-index of NE corner

    Returns:
        int: corrected imin and imax and roll for arrays
    """

    if imin > imax:
        warn("rolling i indices")

    iroll = -imax if imin > imax else 0
    imin_corrected = imin - imax if imin > imax else imin
    imax_corrected = None if imin > imax else imax

    return imin_corrected, imax_corrected, iroll


def subset_input_data(
    lon_c,
    lat_c,
    lon_in,
    lat_in,
    data_in,
    is_carth=False,
    is_stereo=False,
    coords2d=False,
    topotree=None,
):
    """_summary_

    Args:
        lon_c (_type_): _description_
        lat_c (_type_): _description_
        lon_in (_type_): _description_
        lat_in (_type_): _description_
        data_in (_type_): _description_
        is_carth (bool, optional): _description_. Defaults to False.
        is_stereo (bool, optional): _description_. Defaults to False.
        coords2d (bool, optional): _description_. Defaults to False.
        topotree (_type_, optional): _description_. Defaults to None.
    """

    # find geographical bounds of model cell
    lonmin, lonmax, latmin, latmax = find_geographical_bounds(lon_c, lat_c)
    # find index of SW corner in source data
    imin, jmin = find_nearest_point(
        lon_in,
        lat_in,
        lonmin,
        latmin,
        tree=topotree,
        is_carth=is_carth,
    )
    # find index of NE corner in source data
    imax, jmax = find_nearest_point(
        lon_in,
        lat_in,
        lonmax,
        latmax,
        tree=topotree,
        is_carth=is_carth,
    )

    if is_carth:
        # ensure growing order
        imin, imax = correct_for_poles_j_indices(imin, imax)
        jmin, jmax = correct_for_poles_j_indices(jmin, jmax)
        iroll = 0

        lon_src = lon_in[jmin:jmax, imin:imax]
        lat_src = lat_in[jmin:jmax, imin:imax]
    else:
        jmin, jmax = correct_for_poles_j_indices(jmin, jmax)

        # roll the array if necessary using periodicity
        imin, imax, iroll = correct_for_periodicity(imin, imax)

        if coords2d:
            lon_subsubset = np.roll(lon_in, iroll, axis=-1)[jmin:jmax, imin:imax]
            lat_subsubset = np.roll(lat_in, iroll, axis=-1)[jmin:jmax, imin:imax]
            lon_src, lat_src = lon_subsubset, lat_subsubset
        else:
            lon_subsubset = np.roll(lon_in, iroll, axis=0)[imin:imax]
            lat_subsubset = lat_in[jmin:jmax]
            lon_src, lat_src = np.meshgrid(lon_subsubset, lat_subsubset)

    # this is for 1d lon/lat on source grid, need to expand to 2d lon/lat
    if iroll != 0:
        topo_subsubset = np.roll(data_in, iroll, axis=-1)[jmin:jmax, imin:imax]
    else:
        topo_subsubset = data_in[jmin:jmax, imin:imax]

    return lon_src, lat_src, topo_subsubset
