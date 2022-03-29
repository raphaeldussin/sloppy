import numpy as np
import xarray as xr
from sloppy.serial import compute_cell_topo_stat


def compute_block(
    datopo,
    lon_model_block,
    lat_model_block,
    topovar="elevation",
    topolon="lon",
    topolat="lat",
):

    nyb, nxb = lon_model_block.shape
    h_out = np.empty((ny, nx, 5))

    for jj in tqdm(range(ny - 1)):
        for ji in range(nx - 1):
            lon_c = lon_model[jj : jj + 2, ji : ji + 2]
            lat_c = lat_model[jj : jj + 2, ji : ji + 2]

            lonmin = lon_c.min()
            lonmax = lon_c.max()
            latmin = lat_c.min()
            latmax = lat_c.max()

            # this is for 1d lon/lat on source grid, need to expand to 2d lon/lat
            datopo_subset = datopo.sel(
                {"topolon": slice(lonmin, lonmax), "topolat": slice(latmin, latmax)}
            )

            lon_src, lat_src = np.meshgrid(datopo[topolon], datopo[topolat])

            h, hmin, hmax, h2, npts = compute_cell_topo_stats(
                lon_c, lat_c, lon_src, lat_src, datopo[topovar].values
            )
            # cast all in one array to make things easier in dask
            h_out[jj, ji, 0] = h
            h_out[jj, ji, 1] = hmin
            h_out[jj, ji, 2] = hmax
            h2_out[jj, ji, 3] = h2
            h2_out[jj, ji, 4] = float(npts)

            return h_out
