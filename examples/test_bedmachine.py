import numpy as np
import xarray as xr

from sloppy.distrib import compute_block


def add_lon_lat(ds, PROJSTRING, x="x", y="y", chunks={}):
    """add longitude and latitude as compute from the inverse projection
    given in PROJSTRING
    PARAMETERS:
    -----------
    ds: xarray.Dataset
    PROJSTRING: str
    """
    from pyproj import CRS, Transformer

    # create the coordinate reference system
    crs = CRS.from_proj4(PROJSTRING)
    # create the projection from lon/lat to x/y
    proj = Transformer.from_crs(crs.geodetic_crs, crs)
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    # compute the lon/lat
    lon, lat = proj.transform(xx, yy, direction="INVERSE")
    # add to dataset
    ds["lon"] = xr.DataArray(data=lon, dims=("y", "x"))
    ds["lat"] = xr.DataArray(data=lat, dims=("y", "x"))
    ds["lon"].attrs = dict(units="degrees_east")
    ds["lat"].attrs = dict(units="degrees_north")
    return ds


PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

bedmachine = xr.open_dataset(
    "/media/raphael/L2/topography/BedMachineAntarctica_2020-07-15_v02.nc"
)

bedmachine_10x = bedmachine.isel(x=slice(0, -1, 10), y=slice(0, -1, 10))
bedmachine_10x = add_lon_lat(bedmachine_10x, PROJSTRING)

# make up a target grid
# lon_model, lat_model = np.meshgrid(gebco.lon[::100], gebco.lat[::100])
lon_model, lat_model = np.meshgrid(
    # np.linspace(-180, 180, 181), np.linspace(-90, 90, 91)
    np.linspace(-180, 180, 181),
    np.linspace(-90, -60, 16),
)

out = compute_block(
    lon_model,
    lat_model,
    # bedmachine_10x["surface"].values,
    # bedmachine_10x["lon"].values,
    bedmachine_10x["lat"].values,
    bedmachine_10x["lon"].values,
    bedmachine_10x["lat"].values,
    is_stereo=True,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=False,
    algo="fast",
)

import matplotlib.pyplot as plt

plt.figure()
plt.pcolormesh(out[0, :, :], vmax=-60)
plt.colorbar()

plt.figure()
plt.pcolormesh(out[4, :, :])
plt.colorbar()
plt.show()

from pyproj import CRS, Transformer

# create the coordinate reference system
crs = CRS.from_proj4(PROJSTRING)
# create the projection from lon/lat to x/y
proj = Transformer.from_crs(crs.geodetic_crs, crs)
xx, yy = proj.transform(lon_model, lat_model)

plt.figure()
plt.pcolormesh(xx, yy, out[0, :, :], vmax=-60)
plt.colorbar()

plt.figure()
plt.pcolormesh(xx, yy, out[4, :, :])
plt.colorbar()
plt.show()


bedmachine_5x = bedmachine.isel(x=slice(0, -1, 5), y=slice(0, -1, 5))
bedmachine_5x = add_lon_lat(bedmachine_5x, PROJSTRING)

out = compute_block(
    lon_model,
    lat_model,
    bedmachine_5x["bed"].values,
    # bedmachine_5x["lat"].values,
    bedmachine_5x["lon"].values,
    bedmachine_5x["lat"].values,
    is_stereo=True,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=True,
    algo="sturdy",
)

plt.figure()
plt.pcolormesh(xx, yy, out[0, :, :], vmax=5000)
plt.colorbar()

plt.figure()
plt.pcolormesh(xx, yy, out[3, :, :])
plt.colorbar()

plt.figure()
plt.pcolormesh(xx, yy, out[4, :, :])
plt.colorbar()



plt.show()
