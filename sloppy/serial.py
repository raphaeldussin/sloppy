from numba import njit
import numpy as np


@njit
def compute_cell_topo_stats(lon_c, lat_c, lon_src, lat_src, data_src, method="median"):
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
    list
        [cell estimated depth, minimum depth in cell, maximum depth in cell, residual of plane fit, number of source points in cell]

    """

    ny, nx = data_src.shape

    coords_in_cell = []
    data_src_in_cell = []

    for jj in range(ny):
        for ji in range(nx):

            # test if point falls in the cell, defined by 4 segments
            lat_lower_bnd = lat_c[0, 0] + (lon_src[jj, ji] - lon_c[0, 0]) * (
                (lat_c[0, -1] - lat_c[0, 0]) / (lon_c[0, -1] - lon_c[0, 0])
            )

            if lat_src[jj, ji] >= lat_lower_bnd:
                lon_right_bnd = (
                    lon_c[0, -1]
                    + (lat_src[jj, ji] - lat_c[0, -1])
                    + ((lon_c[-1, -1] - lon_c[0, -1]) / (lat_c[-1, -1] - lat_c[0, -1]))
                )

                if lon_src[jj, ji] <= lon_right_bnd:
                    lat_upper_bnd = lat_c[-1, 0] + (-lon_src[jj, ji] + lon_c[-1, 0]) * (
                        (lat_c[-1, -1] - lat_c[-1, 0]) / (lon_c[-1, -1] - lon_c[-1, 0])
                    )

                    if lat_src[jj, ji] <= lat_upper_bnd:
                        lon_left_bnd = (
                            lon_c[0, 0]
                            + (-lat_src[jj, ji] + lat_c[0, 0])
                            + (
                                (lon_c[-1, 0] - lon_c[0, 0])
                                / (lat_c[-1, 0] - lat_c[0, 0])
                            )
                        )

                        if lon_src[jj, ji] >= lon_left_bnd:
                            # all checks passed, adding point to list
                            coords_in_cell.append(
                                [lon_src[jj, ji], lat_src[jj, ji], 1.0]
                            )
                            data_src_in_cell.append(1.0 * data_src[jj, ji])

    # count number of hits in cell
    npts = len(data_src_in_cell)

    A = np.array(coords_in_cell)
    d = np.array(data_src_in_cell)

    if npts == 0:
        dout = 1.0e20
        dmin = 1.0e20
        dmax = 1.0e20
    else:
        dmin = d.min()
        dmax = d.max()
        if method == "median":
            dout = np.median(d)
        elif method == "mean":
            dout = d.mean()  # assume identical weights for source cells to start

    if npts >= 3:  # we need at least 3 points to fit a plane
        # plane fitting to get the residuals
        fitcoefs, residual2, _, _ = np.linalg.lstsq(A, d)
    else:
        residual2 = np.array([0.0])

    return dout, dmin, dmax, residual2, npts
