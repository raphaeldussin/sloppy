import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sloppy.distrib import compute_block

gebco = xr.open_dataset("/local2/home/GEBCO_2021/without_bedmachine/GEBCO_2021.nc")
gebco_10x = gebco["elevation"].isel(lon=slice(0, -1, 10), lat=slice(0, -1, 10))

# make up a target grid
# lon_model, lat_model = np.meshgrid(gebco.lon[::100], gebco.lat[::100])
lon_model, lat_model = np.meshgrid(
    np.linspace(-180, 180, 361), np.linspace(-90, 90, 181)
)

out = compute_block(
    lon_model,
    lat_model,
    gebco_10x.values,
    gebco_10x["lon"].values,
    gebco_10x["lat"].values,
    residual=False,
    algo="fast",
)


plt.figure()
plt.pcolormesh(out[0, :, :])
plt.colorbar()
plt.show()

out = compute_block(
    lon_model,
    lat_model,
    gebco_10x.values,
    gebco_10x["lon"].values,
    gebco_10x["lat"].values,
    residual=True,
    algo="sturdy",
)


plt.figure()
plt.pcolormesh(out[3, :, :])
plt.colorbar()
plt.show()
