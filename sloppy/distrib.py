from warnings import warn

import numpy as np
from scipy.spatial import KDTree
import xarray as xr
from tqdm import tqdm

from sloppy.serial import (
    compute_cell_topo_stats,
    find_geographical_bounds,
    find_nearest_point,
)


def compute_block(
    lon_model_block,
    lat_model_block,
    topo_subset,
    topo_lon_subset,
    topo_lat_subset,
):

    nyb, nxb = lon_model_block.shape
    h_out = np.empty((nyb, nxb, 5))

    coords2d = True if len(topo_lon_subset) == 2 else False

    if coords2d:
        topo_lon_subset360 = np.mod(topo_lon_subset + 360, 360)
        topotree = KDTree(
            list(zip(topo_lon_subset360.flatten(), topo_lat_subset.flatten()))
        )
    else:
        topotree = None

    # loop over all grid cells
    for jj in tqdm(range(nyb - 1)):
        for ji in range(nxb - 1):
            lon_c = lon_model_block[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model_block[jj : jj + 2, ji : ji + 2]

            # find geographical bounds of model cell
            lonmin, lonmax, latmin, latmax = find_geographical_bounds(lon_c, lat_c)
            # find index of SW corner in source data
            imin, jmin = find_nearest_point(
                topo_lon_subset, topo_lat_subset, lonmin, latmin, tree=topotree
            )
            # find index of NE corner in source data
            imax, jmax = find_nearest_point(
                topo_lon_subset, topo_lat_subset, lonmax, latmax, tree=topotree
            )

            # this is for 1d lon/lat on source grid, need to expand to 2d lon/lat
            topo_subsubset = topo_subset[jmin:jmax, imin:imax]
            if coords2d:
                lon_subsubset = topo_lon_subset[jmin:jmax, imin:imax]
                lat_subsubset = topo_lat_subset[jmin:jmax, imin:imax]
                lon_src, lat_src = lon_subsubset, lat_subsubset
            else:
                lon_subsubset = topo_lon_subset[imin:imax]
                lat_subsubset = topo_lat_subset[jmin:jmax]
                lon_src, lat_src = np.meshgrid(lon_subsubset, lat_subsubset)

            if len(topo_subsubset.flatten()) > 1:
                out = compute_cell_topo_stats(
                    lon_c, lat_c, lon_src, lat_src, topo_subsubset
                )
            else:
                out = np.zeros((5))

            if out[4] <= 4:
                warn(
                    "not enough source points to compute stats, switching to nearest neighbor"
                )
                print(lon_c, lat_c)
                print(imin, imax)
                print(lon_src, lat_src)
                # TO DO: add nearest neghbors

            h_out[jj, ji, :] = out

    return h_out
