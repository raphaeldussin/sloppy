from warnings import warn

import numpy as np
from scipy.spatial import KDTree
import xarray as xr
from tqdm import tqdm

from sloppy.serial import (
    compute_cell_topo_stats,
    subset_source_data,
    adjust_xcoord_across_discontinuity,
)


def xblock(
    da_topo,
    da_lon_model,
    da_lat_model,
    topo_lon_name="lon",
    topo_lat_name="lat",
    is_carth=False,
    is_stereo=False,
    PROJSTRING=None,
    compute_residual=True,
    algo="fast",
    debug=False,
):

    h_out = compute_block(da_lon_model.values,
                          da_lat_model.values,
                          da_topo.values,
                          da_topo[topo_lon_name].values,
                          da_topo[topo_lat_name].values,
                          is_carth=is_carth,
                          is_stereo=is_stereo,
                          PROJSTRING=PROJSTRING,
                          residual=compute_residual,
                          algo=algo,
                          debug=debug,
                          )

    ds_out = xr.Dataset()
    ds_out["h"] = xr.DataArray(data=h_out[0,:,:], dims=("y", "x"))
    ds_out["hmin"] =  xr.DataArray(data=h_out[1,:,:], dims=("y", "x"))
    ds_out["hmax"] =  xr.DataArray(data=h_out[2,:,:], dims=("y", "x"))
    ds_out["h2"] =  xr.DataArray(data=h_out[3,:,:], dims=("y", "x"))
    ds_out["npts"] =  xr.DataArray(data=h_out[4,:,:], dims=("y", "x"))

    return ds_out



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
    debug=False,
):

    nyb, nxb = lon_model_block.shape  # number of corners = N+1, M+1 number of centers
    h_out = np.empty((5, nyb - 1, nxb - 1))

    if not is_carth:
        topo_lon_subset = np.mod(topo_lon_subset + 360, 360)

    if not is_carth:
        lon_model_block = np.mod(lon_model_block + 360.0, 360.0)

    # loop over all grid cells
    for jj in tqdm(range(nyb - 1)):
        for ji in range(nxb - 1):
            lon_c = lon_model_block[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model_block[jj : jj + 2, ji : ji + 2]

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
    debug=False,
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

    if not is_carth:
        topo_lon_subset = np.mod(topo_lon_subset + 360, 360)

    if not is_carth:
        lon_model_block = np.mod(lon_model_block + 360.0, 360.0)

    coords2d = True if len(topo_lon_subset.shape) == 2 else False

    if coords2d:  # also not carth
        srctree = KDTree(
            list(zip(topo_lon_subset.flatten(), topo_lat_subset.flatten()))
        )
    else:
        srctree = None

    # loop over all grid cells
    for jj in tqdm(range(nyb - 1), dynamic_ncols=True):
        for ji in range(nxb - 1):
            lon_c = lon_model_block[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model_block[jj : jj + 2, ji : ji + 2]

            lon_src, lat_src, topo_subsubset = subset_source_data(
                lon_c,
                lat_c,
                topo_lon_subset,
                topo_lat_subset,
                topo_subset,
                is_carth=is_carth,
                srctree=srctree,
            )

            if debug:
                print(f"lon grid = {lon_c}")
                print(f"lat grid = {lat_c}")
                print(f"lon topo min/max = {lon_src.min()} {lon_src.max()}")
                print(f"lat topo min/max = {lat_src.min()} {lat_src.max()}")

            if not coords2d:
                lon_src_2d, lat_src_2d = np.meshgrid(lon_src, lat_src)
            else:
                lon_src_2d = lon_src
                lat_src_2d = lat_src

            if len(topo_subsubset.flatten()) > 1:
                out = compute_cell_topo_stats(
                    lon_c,
                    lat_c,
                    lon_src_2d,
                    lat_src_2d,
                    topo_subsubset,
                    compute_res=residual,
                    algo=algo,
                    is_carth=is_carth,
                    debug=debug,
                )
            else:
                print(f"WARNING: no points found for target point ({ji}, {jj})")
                out = np.zeros((5))

            h_out[:, jj, ji] = out

    return h_out
