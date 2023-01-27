import numpy as np
from numba import njit
from scipy.spatial import KDTree
from warnings import warn


def compute_cell_topo_stats(
    x_bnds,
    y_bnds,
    x_src,
    y_src,
    data_src,
    method="median",
    compute_res=True,
    tol=1e-6,
    algo="fast",
):
    """Compute the stats (h median/mean, hmin, hmax, h2) for one grid cell

    Parameters
    ----------
    x_bnds : np.ndarray
        2x2 array containing the x-coord/longitudes of all 4 corners of the cell
    y_bnds : np.ndarray
        2x2 array containing the y-coord/latitudes of all 4 corners of the cell
    x_src : np.ndarray
        2d array (ny, nx) containing the x-coord/longitudes of the source data
    y_src : np.ndarray
        2d array (ny, nx) containing the y-coord/latitudes of the source data
    data_src : np.ndarray
        2d array (ny, nx) containing the the source data
    method (optional) : string
        estimation of cell bathy (median/mean). Defaults to median.
    compute_res (optional): bool
        compute the residual of plane fit. Default=True
    tol (optional): float
        minimum distance allowed between points to trigger triangle algo,
        Defaults to 1e-6
    algo (optional): string
        use "sturdy" or "fast" algorithm. Defaults to "fast"

    Returns
    -------
    np.array containing
        [cell estimated depth, minimum depth in cell, maximum depth in cell, residual of plane fit, number of source points in cell]

    """

    # this wrapper is needed to handle edge cases (e.g. no points found,...)

    out = 1.0e20 * np.ones((5))
    out[4] = 0.0
    if len(data_src.flatten()) < 1:
        print("no points passed to algo")
        print("x/lon", x_bnds)
        print("y/lat", y_bnds)
        return out

    # detect singular points
    n_singular_pts = 0
    sing_top = False
    sing_bot = False
    sing_left = False
    sing_right = False

    # upper point is singular
    if np.allclose(x_bnds[-1, -1], x_bnds[-1, 0], atol=tol) and np.allclose(
        y_bnds[-1, -1], y_bnds[-1, 0], atol=tol
    ):
        n_singular_pts += 1
        sing_top = True

    # lower point is singular
    if np.allclose(x_bnds[0, -1], x_bnds[0, 0], atol=tol) and np.allclose(
        y_bnds[0, -1], y_bnds[0, 0], atol=tol
    ):
        n_singular_pts += 1
        sing_bot = True

    # left point is singular
    if np.allclose(x_bnds[-1, 0], x_bnds[0, 0], atol=tol) and np.allclose(
        y_bnds[-1, 0], y_bnds[0, 0], atol=tol
    ):
        n_singular_pts += 1
        sing_left = True

    # right point is singular
    if np.allclose(x_bnds[-1, -1], x_bnds[0, -1], atol=tol) and np.allclose(
        y_bnds[-1, -1], y_bnds[0, -1], atol=tol
    ):
        n_singular_pts += 1
        sing_right = True

    if n_singular_pts >= 2:
        warn("singular point or segment detected!!!")
        dout, dmin, dmax, residual2, npts = 1.0e36, 1.0e36, 1.0e36, 1.0e36, 0
    elif n_singular_pts == 1:
        # use triangle algo
        dout, dmin, dmax, residual2, npts = compute_cell_topo_stats_triangle(
            x_bnds,
            y_bnds,
            x_src,
            y_src,
            data_src,
            sing_top=sing_top,
            sing_bot=sing_bot,
            sing_left=sing_left,
            sing_right=sing_right,
            method=method,
            compute_res=compute_res,
        )
    elif n_singular_pts == 0:
        # reorder points
        x_bnds, y_bnds = reorder_bounds(x_bnds, y_bnds)
        # use parallelogram algo
        dout, dmin, dmax, residual2, npts = compute_cell_topo_stats_parallelogram(
            x_bnds,
            y_bnds,
            x_src,
            y_src,
            data_src,
            method=method,
            compute_res=compute_res,
            algo=algo,
        )
    else:
        raise ValueError("you should not arrive here")

    out[0] = dout
    out[1] = dmin
    out[2] = dmax
    out[3] = residual2
    out[4] = float(npts)

    return out


@njit
def compute_cell_topo_stats_triangle(
    x_bnds,
    y_bnds,
    x_src,
    y_src,
    data_src,
    sing_top=False,
    sing_bot=False,
    sing_left=False,
    sing_right=False,
    method="median",
    compute_res=True,
    rtol=0.001,
):

    # first build triangle from degenerated parallelogram
    if sing_top:
        x1 = 0.5 * (x_bnds[-1, 0] + x_bnds[-1, -1])
        y1 = 0.5 * (y_bnds[-1, 0] + y_bnds[-1, -1])
        x2, x3 = x_bnds[0, :]
        y2, y3 = y_bnds[0, :]

    if sing_bot:
        x1 = 0.5 * (x_bnds[0, 0] + x_bnds[0, -1])
        y1 = 0.5 * (y_bnds[0, 0] + y_bnds[0, -1])
        x2, x3 = x_bnds[-1, :]
        y2, y3 = y_bnds[-1, :]

    if sing_left:
        x1 = 0.5 * (x_bnds[0, 0] + x_bnds[-1, 0])
        y1 = 0.5 * (y_bnds[0, 0] + y_bnds[-1, 0])
        x2, x3 = x_bnds[:, -1]
        y2, y3 = y_bnds[:, -1]

    if sing_right:
        x1 = 0.5 * (x_bnds[0, -1] + x_bnds[-1, -1])
        y1 = 0.5 * (y_bnds[0, -1] + y_bnds[-1, -1])
        x2, x3 = x_bnds[:, 0]
        y2, y3 = y_bnds[:, 0]

    # total area of the triangle
    A_expected = np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    ny, nx = data_src.shape[-2:]
    coords_in_cell = []
    data_src_in_cell = []

    for jj in range(ny):
        for ji in range(nx):

            x = x_src[jj, ji]
            y = y_src[jj, ji]

            A1 = np.abs((x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)) / 2.0)
            A2 = np.abs((x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y)) / 2.0)
            A3 = np.abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2.0)

            if np.abs(A1 + A2 + A3 - A_expected) < rtol * A_expected:
                # check passed, adding point to list
                coords_in_cell.append([x_src[jj, ji], y_src[jj, ji], 1.0])
                data_src_in_cell.append(1.0 * data_src[jj, ji])

    # count number of hits in cell
    npts = len(data_src_in_cell)

    if npts == 0:
        dout = 1.0e20
        dmin = 1.0e20
        dmax = 1.0e20
    else:
        d = np.array(data_src_in_cell)
        dmin = d.min()
        dmax = d.max()
        if method == "median":
            dout = np.median(d)
        elif method == "mean":
            dout = d.mean()  # assume identical weights for source cells to start

    roughness = 0.0
    if compute_res:
        roughness = compute_roughness(coords_in_cell, data_src_in_cell)

    return dout, dmin, dmax, roughness, npts


@njit()
def compute_cell_topo_stats_parallelogram(
    x_bnds,
    y_bnds,
    x_src,
    y_src,
    data_src,
    method="median",
    eps=1e-16,
    compute_res=True,
    algo="fast",
):
    """Compute the stats (h median/mean, hmin, hmax, h2) for one grid cell

    Parameters
    ----------
    x_bnds : np.ndarray
        2x2 array containing the x-coord/longitudes of all 4 corners of the cell
    y_bnds : np.ndarray
        2x2 array containing the y-coord/latitudes of all 4 corners of the cell
    x_src : np.ndarray
        2d array (ny, nx) containing the x-coord/longitudes of the source data
    y_src : np.ndarray
        2d array (ny, nx) containing the y-coord/latitudes of the source data
    data_src : np.ndarray
        2d array (ny, nx) containing the the source data
    method (optional) : string
        estimation of cell bathy (median/mean). Defaults to median.
    eps (optional): float
        small number to avoid division by zero.  Defaults to 1e-16
    compute_res (optional): bool
        compute the residual of plane fit. Default=True
    algo (optional): string
        use "sturdy" or "fast" algorithm. Defaults to "fast"

    Returns
    -------
    np.array containing
        [cell estimated depth, minimum depth in cell, maximum depth in cell, residual of plane fit, number of source points in cell]

    """

    ny, nx = data_src.shape[-2:]
    coords_in_cell = []
    data_src_in_cell = []

    if algo == "sturdy":
        for jj in range(ny):
            for ji in range(nx):

                y_lower_bnd = y_bnds[0, 0] + (x_src[jj, ji] - x_bnds[0, 0]) * (
                    (y_bnds[0, -1] - y_bnds[0, 0])
                    / np.maximum(x_bnds[0, -1] - x_bnds[0, 0], eps)
                )
                y_upper_bnd = y_bnds[-1, 0] + (x_src[jj, ji] - x_bnds[-1, 0]) * (
                    (y_bnds[-1, -1] - y_bnds[-1, 0])
                    / np.maximum(x_bnds[-1, -1] - x_bnds[-1, 0], eps)
                )

                y_lower_bnd_c = np.minimum(y_lower_bnd, y_upper_bnd)
                y_upper_bnd_c = np.maximum(y_lower_bnd, y_upper_bnd)

                if (y_src[jj, ji] >= y_lower_bnd_c) and (
                    y_src[jj, ji] <= y_upper_bnd_c
                ):

                    x_right_bnd = x_bnds[0, -1] + (y_src[jj, ji] - y_bnds[0, -1]) * (
                        (x_bnds[-1, -1] - x_bnds[0, -1])
                        / np.maximum(y_bnds[-1, -1] - y_bnds[0, -1], eps)
                    )

                    x_left_bnd = x_bnds[0, 0] + (y_src[jj, ji] - y_bnds[0, 0]) * (
                        (x_bnds[-1, 0] - x_bnds[0, 0])
                        / np.maximum(y_bnds[-1, 0] - y_bnds[0, 0], eps)
                    )

                    x_left_bnd_c = np.minimum(x_left_bnd, x_right_bnd)
                    x_right_bnd_c = np.maximum(x_left_bnd, x_right_bnd)

                    if (x_src[jj, ji] >= x_left_bnd_c) and (
                        x_src[jj, ji] <= x_right_bnd_c
                    ):
                        coords_in_cell.append([x_src[jj, ji], y_src[jj, ji], 1.0])
                        data_src_in_cell.append(1.0 * data_src[jj, ji])

    elif algo == "fast":

        for jj in range(ny):
            for ji in range(nx):
                # test if point falls in the cell, defined by 4 segments
                y_lower_bnd = y_bnds[0, 0] + (x_src[jj, ji] - x_bnds[0, 0]) * (
                    (y_bnds[0, -1] - y_bnds[0, 0]) / (x_bnds[0, -1] - x_bnds[0, 0])
                )

                if y_src[jj, ji] >= y_lower_bnd:
                    x_right_bnd = x_bnds[0, -1] + (y_src[jj, ji] - y_bnds[0, -1]) * (
                        (x_bnds[-1, -1] - x_bnds[0, -1])
                        / (y_bnds[-1, -1] - y_bnds[0, -1])
                    )

                    if x_src[jj, ji] <= x_right_bnd:
                        y_upper_bnd = y_bnds[-1, 0] + (
                            x_src[jj, ji] - x_bnds[-1, 0]
                        ) * (
                            (y_bnds[-1, -1] - y_bnds[-1, 0])
                            / (x_bnds[-1, -1] - x_bnds[-1, 0])
                        )

                        if y_src[jj, ji] <= y_upper_bnd:
                            x_left_bnd = x_bnds[0, 0] + (
                                y_src[jj, ji] - y_bnds[0, 0]
                            ) * (
                                (x_bnds[-1, 0] - x_bnds[0, 0])
                                / (y_bnds[-1, 0] - y_bnds[0, 0])
                            )

                            if x_src[jj, ji] >= x_left_bnd:
                                # all checks passed, adding point to list
                                coords_in_cell.append(
                                    [x_src[jj, ji], y_src[jj, ji], 1.0]
                                )
                                data_src_in_cell.append(1.0 * data_src[jj, ji])

    # count number of hits in cell
    npts = len(data_src_in_cell)

    if npts == 0:
        dout = 1.0e20
        dmin = 1.0e20
        dmax = 1.0e20
    else:
        d = np.array(data_src_in_cell)
        dmin = d.min()
        dmax = d.max()
        if method == "median":
            dout = np.median(d)
        elif method == "mean":
            dout = d.mean()  # assume identical weights for source cells to start

    roughness = 0.0
    if compute_res:
        roughness = compute_roughness(coords_in_cell, data_src_in_cell)

    return dout, dmin, dmax, roughness, npts


@njit
def compute_roughness(coords_in_cell, data_src_in_cell):
    """compute the 'roughness' as the residual from a 2d linear fit to data

    Args:
    -----

    coords_in_cell : list of 3 items list
        triplets containing coordinates and order of fit, e.g.
        [[x1, y1, 1], [x2, y2, 1], ... [xn, yn, 1]]
        1 = linear fit to data
    data_src_in_cell: list
        data (elevation,...) for which to compute residual to fit, e.g.
        [1035., 1589., 1457., ..., 1789.]

    Returns:
    --------

    roughness: float
        residual (in units^2) of fit to data
    """

    # init roughness to zero
    roughness = 0.0
    # we need at least 3 points to fit a plane so 4 are needed
    # to have a residual
    if len(coords_in_cell) >= 4:
        # convert to numpy array to use linear algebra
        A = np.array(coords_in_cell)
        d = np.array(data_src_in_cell)
        # plane fitting is done with least square fitting to data
        fitcoefs, residual2, _, _ = np.linalg.lstsq(A, d)
        roughness = residual2[0] if len(residual2) == 1 else 0.0

    return roughness


def find_nearest_point(x_src, y_src, x_tgt, y_tgt, tree=None, is_carth=False):
    """Function to find the nearest point on the source grid given a
    value on the target grid

    Parameters
    ----------
    x_src : np.ndarray
        1-D or 2-D array of x-coord/longitude values in the source grid
    y_src :
        1-D or 2-D array of y-coord/latitude values in the source grid
    x_tgt : float
        Longitude value to find on the source grid
    y_tgt : float
        Latitude
    tree : scipy.Kdtree object, optional
        Precalculated kdtree object for 2D coordinates, by default None
    is_carth: bool
        coordinates are carhesian. Defaults to False (i.e. geographical)

    Returns
    -------
    np.ndarray
         x,y indices

    """

    if not is_carth:
        # standardize longitude to positive-definite values
        x_src = np.mod(x_src + 360, 360)
        x_tgt = np.mod(x_tgt + 360, 360)

    # use a kdtree to find indicies if source coordinates are 2-dimensional
    if len(x_src.shape) == 2:
        if tree is None:
            raise ValueError("you must supply a kdtree")
        _, ravel_index = tree.query([x_tgt, y_tgt], k=1)
        indy, indx = np.unravel_index(ravel_index, x_src.shape)

    # use a simple locator if source coordinates are vectors
    elif len(x_src.shape) == 1:
        indx = (np.abs(x_src - x_tgt)).argmin()
        indy = (np.abs(y_src - y_tgt)).argmin()

    return indx, indy


def find_geographical_bounds(x_bnds, y_bnds):
    """find the bounding box in which to subset source data,
    given by min/max of bounds of cell"""

    # this is the simple case where there is no periodicity to worry about
    xmin = x_bnds.min()
    xmax = x_bnds.max()
    ymin = y_bnds.min()
    ymax = y_bnds.max()

    return xmin, xmax, ymin, ymax


def correct_flipped_indices(kmin, kmax):
    """make sure that kmin < kmax, this can happen near singular points (polar regions)"""

    kmin_corrected = kmax if kmin > kmax else kmin
    kmax_corrected = kmin if kmin > kmax else kmax

    return kmin_corrected, kmax_corrected


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


def subset_source_data(
    x_bnds,
    y_bnds,
    x_in,
    y_in,
    data_in,
    is_carth=False,
    srctree=None,
):
    """subset the source data by finding the range of values for x/y in the cell,
    then finding the indices of the bottom-left (SW) and top-right (NE) in the coords
    of the source data, correct for periodicity and flipped indices if necessary.

    Args:
    -----

    x_bnds : np.ndarray
        2x2 array containing the x-coord/longitudes of all 4 corners of the cell
    y_bnds : np.ndarray
        2x2 array containing the y-coord/latitudes of all 4 corners of the cell
    x_in : np.ndarray
        2d array (ny, nx) containing the x-coord/longitudes of the source data
    y_in : np.ndarray
        2d array (ny, nx) containing the y-coord/latitudes of the source data
    data_in : np.ndarray
        2d array (ny, nx) containing the the source data
    is_carth: bool
        True for x/y in CRS, False for lat/lon
    srctree: scipy.KDtree
        KDtree with values of source x/y or lon/lat

    """

    # find geographical bounds of model cell
    xmin, xmax, ymin, ymax = find_geographical_bounds(x_bnds, y_bnds)
    # find index of SW corner in source data
    imin, jmin = find_nearest_point(
        x_in,
        y_in,
        xmin,
        ymin,
        tree=srctree,
        is_carth=is_carth,
    )
    # find index of NE corner in source data
    imax, jmax = find_nearest_point(
        x_in,
        y_in,
        xmax,
        ymax,
        tree=srctree,
        is_carth=is_carth,
    )

    if is_carth:
        # ensure growing order
        imin, imax = correct_flipped_indices(imin, imax)
        jmin, jmax = correct_flipped_indices(jmin, jmax)
        iroll = 0

        x_src = x_in[jmin:jmax, imin:imax]
        y_src = y_in[jmin:jmax, imin:imax]
    else:
        jmin, jmax = correct_flipped_indices(jmin, jmax)

        # roll the array if necessary using periodicity
        imin, imax, iroll = correct_for_periodicity(imin, imax)

        if len(x_in.shape) == 2:
            x_subsubset = np.roll(x_in, iroll, axis=-1)[jmin:jmax, imin:imax]
            y_subsubset = np.roll(y_in, iroll, axis=-1)[jmin:jmax, imin:imax]
            x_src, y_src = x_subsubset, y_subsubset
        elif len(x_in.shape) == 1:
            x_subsubset = np.roll(x_in, iroll, axis=0)[imin:imax]
            y_subsubset = y_in[jmin:jmax]
            x_src, y_src = np.meshgrid(x_subsubset, y_subsubset)
        else:
            raise ValueError("x/y must be 1d or 2d arrays")

    # this is for 1d lon/lat on source grid, need to expand to 2d lon/lat
    if iroll != 0:
        topo_subsubset = np.roll(data_in, iroll, axis=-1)[jmin:jmax, imin:imax]
    else:
        topo_subsubset = data_in[jmin:jmax, imin:imax]

    return x_src, y_src, topo_subsubset


def reorder_bounds(x_bnds, y_bnds):
    """_summary_

    Args:
        x_bnds (_type_): _description_
        y_bnds (_type_): _description_
    """

    #if (x_bnds[:, -1] - x_bnds[:, 0]).min() <= -180.:  # revisit this for non lat/lon coords
    #    # we cross the zero line
    #    return x_bnds, y_bnds

    xmean = x_bnds.mean()
    ymean = y_bnds.mean()

    angle = np.arctan2(y_bnds - ymean, x_bnds - xmean)
    idx_sorted = np.argsort(np.mod(angle.flatten() + 2 * np.pi, 2 * np.pi))

    topright = np.unravel_index(idx_sorted[0], x_bnds.shape)
    topleft = np.unravel_index(idx_sorted[1], x_bnds.shape)
    botleft = np.unravel_index(idx_sorted[2], x_bnds.shape)
    botright = np.unravel_index(idx_sorted[3], x_bnds.shape)

    x_bnds_reorder = np.array(
        [[x_bnds[botleft], x_bnds[botright]], [x_bnds[topleft], x_bnds[topright]]]
    )
    y_bnds_reorder = np.array(
        [[y_bnds[botleft], y_bnds[botright]], [y_bnds[topleft], y_bnds[topright]]]
    )

    assert x_bnds.shape == x_bnds_reorder.shape
    assert y_bnds.shape == y_bnds_reorder.shape

    return x_bnds_reorder, y_bnds_reorder
