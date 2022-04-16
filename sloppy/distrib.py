from warnings import warn

import numpy as np
from scipy.spatial import KDTree
import xarray as xr
from tqdm import tqdm

from sloppy.serial import (
    compute_cell_topo_stats,
    correct_for_periodicity,
    correct_for_poles_j_indices,
    find_geographical_bounds,
    find_nearest_point,
)


def compute_block(
    lon_model_block,
    lat_model_block,
    topo_subset,
    topo_lon_subset,
    topo_lat_subset,
    is_carth=False,
    is_stereo=False,
    PROJSTRING=None,
):

    nyb, nxb = lon_model_block.shape  # number of corners = N+1, M+1 number of centers
    h_out = np.empty((nyb - 1, nxb - 1, 5))

    if is_stereo:
        if PROJSTRING is not None:
            from pyproj import CRS, Transformer

            # create the coordinate reference system
            crs = CRS.from_proj4(PROJSTRING)
            # create the projection from lon/lat to x/y
            proj = Transformer.from_crs(crs.geodetic_crs, crs)
            # override lon/lat by x/y of CRS
            lon_model_block, lat_model_block = proj.transform(
                lon_model_block, lat_model_block
            )
            topo_lon_subset, topo_lat_subset = proj.transform(
                topo_lon_subset, topo_lat_subset
            )

    coords2d = True if len(topo_lon_subset.shape) == 2 else False

    if coords2d:
        if not is_stereo:
            topo_lon_subset360 = np.mod(topo_lon_subset + 360, 360)
        else:
            topo_lon_subset360 = topo_lon_subset
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
                topo_lon_subset,
                topo_lat_subset,
                lonmin,
                latmin,
                tree=topotree,
                is_carth=is_carth,
            )
            # find index of NE corner in source data
            imax, jmax = find_nearest_point(
                topo_lon_subset,
                topo_lat_subset,
                lonmax,
                latmax,
                tree=topotree,
                is_carth=is_carth,
            )

            # this is necessary in polar regions
            jmin, jmax = correct_for_poles_j_indices(jmin, jmax)
            # roll the array if necessary using periodicity
            if not is_stereo:
                imin, imax, iroll = correct_for_periodicity(imin, imax)
            else:
                iroll = 0

            # this is for 1d lon/lat on source grid, need to expand to 2d lon/lat
            if iroll != 0:
                topo_subsubset = np.roll(topo_subset, iroll, axis=-1)[
                    jmin:jmax, imin:imax
                ]
            else:
                topo_subsubset = topo_subset[jmin:jmax, imin:imax]

            if coords2d:
                lon_subsubset = np.roll(topo_lon_subset, iroll, axis=-1)[
                    jmin:jmax, imin:imax
                ]
                lat_subsubset = np.roll(topo_lat_subset, iroll, axis=-1)[
                    jmin:jmax, imin:imax
                ]
                lon_src, lat_src = lon_subsubset, lat_subsubset
            else:
                lon_subsubset = np.roll(topo_lon_subset, iroll, axis=0)[imin:imax]
                lat_subsubset = topo_lat_subset[jmin:jmax]
                lon_src, lat_src = np.meshgrid(lon_subsubset, lat_subsubset)

            if len(topo_subsubset.flatten()) > 1:
                out = compute_cell_topo_stats(
                    lon_c,
                    lat_c,
                    lon_src,
                    lat_src,
                    topo_subsubset,
                )
            else:
                out = np.zeros((5))

            if out[4] <= 4:
                warn(
                    f"not enough source points (= {out[4]}/{len(lon_src)}) in cell {lonmin} - {lonmax}/{latmin} - {latmax} \
                      to compute stats at (j,i) = ({jj},{ji}), switching to nearest neighbor"
                )
                # print(f"subset of source grid is {imin} - {imax} / {jmin} - {jmax}")
                # print(lon_src.shape)
                # print(lat_src.shape)
                # print(lon_src.min(), lon_src.max())
                # print(lat_src.min(), lat_src.max())
                # print(imin, imax)
                # print(lon_src, lat_src)
                # TO DO: add nearest neghbors

            h_out[jj, ji, :] = out

    return h_out
