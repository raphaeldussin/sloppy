from warnings import warn

import numpy as np
import xarray as xr
from tqdm import tqdm

from sloppy.serial import compute_cell_topo_stats


def compute_block(
    datopo,
    lon_model_block,
    lat_model_block,
    topovar="elevation",
    topolon="lon",
    topolat="lat",
):

    nyb, nxb = lon_model_block.shape
    h_out = np.empty((nyb, nxb, 5))

    # print(nyb, nxb)

    for jj in tqdm(range(nyb - 1)):
        for ji in range(nxb - 1):
            lon_c = lon_model_block[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model_block[jj : jj + 2, ji : ji + 2]

            lonmin = lon_c.min()
            lonmax = lon_c.max()
            latmin = lat_c.min()
            latmax = lat_c.max()

            # this is for 1d lon/lat on source grid, need to expand to 2d lon/lat
            datopo_subset = datopo.sel(
                {topolon: slice(lonmin, lonmax), topolat: slice(latmin, latmax)}
            )

            lon_src, lat_src = np.meshgrid(
                datopo_subset[topolon], datopo_subset[topolat]
            )

            out = compute_cell_topo_stats(
                lon_c, lat_c, lon_src, lat_src, datopo_subset[topovar].values
            )

            if out[4] <= 4:
                warn(
                    "not enough source points to compute stats, switching to nearest neighbor"
                )
                # TO DO: add nearest neghbors

            h_out[jj, ji, :] = out

    return h_out
