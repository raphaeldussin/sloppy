from warnings import warn

import numpy as np
from scipy.spatial import KDTree
import xarray as xr
from tqdm import tqdm

from sloppy.serial import (
    compute_cell_topo_stats,
    subset_source_data,
    adjust_xcoord_across_discontinuity
)


def compute_block_brute(
    lon_model_block,
    lat_model_block,
    topo_subset,
    topo_lon_subset,
    topo_lat_subset,
    is_carth=False,
    is_stereo=False,
    PROJSTRING=None,
    residual=True,
    algo="fast",
):


    nyb, nxb = lon_model_block.shape  # number of corners = N+1, M+1 number of centers
    h_out = np.empty((5, nyb - 1, nxb - 1))

    if not is_carth:
        topo_lon_subset = np.mod(topo_lon_subset + 360, 360)

    lon_model_block = np.mod(lon_model_block + 360., 360.)

    # loop over all grid cells
    for jj in tqdm(range(nyb - 1)):
        for ji in range(nxb - 1):
            lon_c = lon_model_block[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model_block[jj : jj + 2, ji : ji + 2]


            #topo_lon_subset_tmp = topo_lon_subset.copy()
            #lon_c_tmp = lon_c.copy()
            #if not is_carth:
            #    if lon_c[-1,-1] < lon_c[-1,0]:
            #        lon_c_tmp[:,0] -= 360
            #        topo_lon_subset_tmp = np.where(topo_lon_subset > 180, topo_lon_subset - 360, topo_lon_subset)

            debug=True if ((ji == 28) and (jj == 32)) else False

            out = compute_cell_topo_stats(
                lon_c,
                lat_c,
                topo_lon_subset,
                topo_lat_subset,
                topo_subset,
                compute_res=residual,
                algo=algo,
                is_carth=is_carth,
                debug=debug,
            )

            h_out[:, jj, ji] = out

    return h_out


def compute_block(
    lon_model_block,
    lat_model_block,
    topo_subset,
    topo_lon_subset,
    topo_lat_subset,
    is_carth=False,
    is_stereo=False,
    PROJSTRING=None,
    residual=True,
    algo="fast",
):

    nyb, nxb = lon_model_block.shape  # number of corners = N+1, M+1 number of centers
    h_out = np.empty((5, nyb - 1, nxb - 1))

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
            # this becomes carthesian
            is_carth = True

    coords2d = True if len(topo_lon_subset.shape) == 2 else False

    if coords2d:
        if not is_carth:
            topo_lon_subset360 = np.mod(topo_lon_subset + 360, 360)
        else:
            topo_lon_subset360 = topo_lon_subset
        srctree = KDTree(
            list(zip(topo_lon_subset360.flatten(), topo_lat_subset.flatten()))
        )
    else:
        topo_lon_subset = np.mod(topo_lon_subset + 360, 360)
        srctree = None


    lon_model_block = np.mod(lon_model_block + 360., 360.)

    # loop over all grid cells
    #for jj in range(nyb - 1):
    for jj in tqdm(range(nyb - 1), dynamic_ncols=True):
        for ji in range(nxb - 1):
            lon_c = lon_model_block[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model_block[jj : jj + 2, ji : ji + 2]

            #if not is_carth:
            #    lon_c, topo_lon_subset = adjust_xcoord_across_discontinuity(lon_c, topo_lon_subset)

            lon_src, lat_src, topo_subsubset = subset_source_data(
                lon_c,
                lat_c,
                topo_lon_subset,
                topo_lat_subset,
                topo_subset,
                is_carth=is_carth,
                srctree=srctree,
            )

            if len(topo_subsubset.flatten()) > 1:
                out = compute_cell_topo_stats(
                    lon_c,
                    lat_c,
                    lon_src,
                    lat_src,
                    topo_subsubset,
                    compute_res=residual,
                    algo=algo,
                    is_carth=is_carth,
                )
            else:
                print(f"WARNING: no points found for target point ({ji}, {jj})")
                out = np.zeros((5))

            h_out[:, jj, ji] = out

    return h_out
