from warnings import warn

import numpy as np
from scipy.spatial import KDTree
import xarray as xr
from tqdm import tqdm

from sloppy.serial import (
    compute_cell_topo_stats,
    subset_input_data,
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
    residual=True,
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

            lon_src, lat_src, topo_subsubset = subset_input_data(
                lon_c,
                lat_c,
                topo_lon_subset,
                topo_lat_subset,
                topo_subset,
                is_carth=is_carth,
                is_stereo=is_stereo,
                coords2d=coords2d,
                topotree=topotree,
            )

            if len(topo_subsubset.flatten()) > 1:
                out = compute_cell_topo_stats(
                    lon_c,
                    lat_c,
                    lon_src,
                    lat_src,
                    topo_subsubset,
                    compute_res=residual,
                )
            else:
                out = np.zeros((5))

            # if out[4] < 1:
            # print(lon_src.shape)
            # print(f" ntotal pts = len(lat_src)")
            # import matplotlib.pyplot as plt
            # print(lon_c)
            # print(lat_c)
            # plt.figure()
            # plt.plot(lon_c, lat_c, "ro")
            # plt.plot(lon_src, lat_src, "ko")
            # plt.show()
            # print()
            # print(lon_src.min(), lon_src.max())
            # print(lat_src.min(), lat_src.max())
            # print(lon_c)
            # print(lat_c)
            # if out[4] <= 4:
            #    warn(
            #        f"not enough source points (= {out[4]}/{len(lon_src)}) in cell {lonmin} - {lonmax}/{latmin} - {latmax} \
            #          to compute stats at (j,i) = ({jj},{ji}), switching to nearest neighbor"
            #    )
            # print(f"subset of source grid is {imin} - {imax} / {jmin} - {jmax}")
            # print(lon_src.shape)
            # print(lat_src.shape)
            # print(lon_src.min(), lon_src.max())
            # print(lat_src.min(), lat_src.max())
            # print(imin, imax)
            # print(lon_src, lat_src)
            # TO DO: add nearest neghbors

            h_out[:, jj, ji] = out

    return h_out
