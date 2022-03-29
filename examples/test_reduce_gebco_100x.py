import numpy as np
import xarray as xr

from sloppy.distrib import compute_block

gebco = xr.open_dataset("/media/raphael/L2/topography/GEBCO_2021.nc")

# make up a target grid
lon_model, lat_model = np.meshgrid(gebco.lon[::100], gebco.lat[::100])

out = compute_block(gebco, lon_model, lat_model)
