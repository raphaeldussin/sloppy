import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from cartopy import crs as ccrs

from sloppy.distrib import compute_block
from sloppy.distrib import compute_block_brute


def proj_xy(lon, lat, PROJSTRING):
    """ """
    from pyproj import CRS, Transformer

    # create the coordinate reference system
    crs = CRS.from_proj4(PROJSTRING)
    # create the projection from lon/lat to x/y
    proj = Transformer.from_crs(crs.geodetic_crs, crs)
    # compute the lon/lat
    xx, yy = proj.transform(lon, lat, direction="FORWARD")
    return xx, yy


PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

#---------------- bedmachine + reduction
bedmachine = xr.open_dataset(
    "/local2/home/BedMachineAntarctica_2020-07-15_v02.nc"
)

bedmachine_10x = bedmachine.isel(x=slice(0, -1, 10), y=slice(0, -1, 10))
#bedmachine_10x = add_lon_lat(bedmachine_10x, PROJSTRING)

bedmachine_5x = bedmachine.isel(x=slice(0, -1, 5), y=slice(0, -1, 5))
#bedmachine_5x = add_lon_lat(bedmachine_5x, PROJSTRING)

#xx_bm, yy_bm = np.meshgrid(bedmachine_10x["x"].values, bedmachine_10x["y"].values)
xx_bm, yy_bm = np.meshgrid(bedmachine_5x["x"].values, bedmachine_5x["y"].values)

#xx_bm_full, yy_bm_full = np.meshgrid(bedmachine["x"].values, bedmachine["y"].values)

#plt.figure()
#plt.pcolormesh(xx_bm, cmap="RdBu_r")
#plt.title("x bedmachine")
#plt.colorbar()
#plt.figure()
#plt.pcolormesh(yy_bm, cmap="RdBu_r")
#plt.title("y bedmachine")
#plt.colorbar()
##plt.show()


#----------------- read lon/lat MOM6 grid corners until 60S

#hgrid = xr.open_dataset("/home/Olga.Sergienko/gridtopo_sandbox/iOM4/73E82S_025deg.nc")
hgrid = xr.open_dataset("/home/Olga.Sergienko/gridtopo_sandbox/iOM4/SP_025deg.nc")
j60s=602

lon_model = hgrid["x"][0:j60s:2, 0::2]
lat_model = hgrid["y"][0:j60s:2, 0::2]

xx_model, yy_model = proj_xy(lon_model, lat_model, PROJSTRING)

plt.figure()
plt.pcolormesh(xx_model, cmap="RdBu_r")
plt.title("x model")
plt.colorbar()
plt.figure()
plt.pcolormesh(yy_model, cmap="RdBu_r")
plt.title("y model")
plt.colorbar()
plt.show()




subplot_kws = dict(
    projection=ccrs.SouthPolarStereo(central_longitude=0.0), facecolor="grey"
)
plt.figure(figsize=[10, 8])
ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0.0))
# ax.stock_img()
plt.pcolormesh(
    lon_model,
    lat_model,
    xx_model,
    shading="auto",
    cmap="jet",
    transform=ccrs.PlateCarree())

plt.colorbar()
ax.set_extent([-180, 180, -55, -90], ccrs.PlateCarree())
ax.gridlines(color="black", alpha=0.5, linestyle="--")


plt.figure(figsize=[10, 8])
ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0.0))
# ax.stock_img()
plt.pcolormesh(
    lon_model,
    lat_model,
    yy_model,
    shading="auto",
    cmap="jet",
    transform=ccrs.PlateCarree())

plt.colorbar()
ax.set_extent([-180, 180, -55, -90], ccrs.PlateCarree())
ax.gridlines(color="black", alpha=0.5, linestyle="--")



#plt.show()


#thk10x = compute_block_brute(
#    xx_model,
#    yy_model,
#    bedmachine_10x["thickness"].values,
#    xx_bm,
#    yy_bm,
#    is_stereo=False,
#    is_carth=False,
#    PROJSTRING=PROJSTRING,
#    residual=True,
#)
#
#
##out = xr.Dataset()
##out["thickness"] = xr.DataArray(data=thk10x[0, :, :], dims=("y", "x"))
##out.to_netcdf("thk_remapped.nc")
#
#plt.figure()
#plt.pcolormesh(xx_model, yy_model, thk10x[0, :, :], vmax=5000)
#plt.colorbar()
#plt.title("10x downsampled - thickness")
#
#plt.show()


thk10x = compute_block(
    xx_model,
    yy_model,
    bedmachine_5x["thickness"].values,
    xx_bm,
    yy_bm,
    is_stereo=False,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=True,
)


out = xr.Dataset()
out["thickness"] = xr.DataArray(data=thk10x[0, :, :], dims=("y", "x"))
out.to_netcdf("thk_remapped.nc")

plt.figure()
plt.pcolormesh(xx_model, yy_model, thk10x[0, :, :], vmax=5000)
plt.colorbar()
plt.title("10x downsampled - thickness")

plt.show()
































#subplot_kws = dict(
#    projection=ccrs.SouthPolarStereo(central_longitude=0.0), facecolor="grey"
#)
#plt.figure(figsize=[10, 8])
#ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0.0))
## ax.stock_img()
#plt.pcolormesh(
#    lon_model,
#    lat_model,
#    thk5x[0, :, :],
#    shading="auto",
#    cmap="jet",
#    transform=ccrs.PlateCarree(),
#), plt.clim(-0, 4000), plt.colorbar()
#ax.set_extent([-180, 180, -55, -90], ccrs.PlateCarree())
#ax.gridlines(color="black", alpha=0.5, linestyle="--")
#
plt.show()

















































istop


# wrong
# lon_model = hgrid["x"][1::2,1::2]
# lat_model = hgrid["y"][1::2,1::2]
# and does not reproduce olga's problem

out10x = compute_block(
    lon_model,
    lat_model,
    bedmachine_10x["lat"].values,
    bedmachine_10x["lon"].values,
    bedmachine_10x["lat"].values,
    is_stereo=True,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=False,
    algo="fast",
)

out5x = compute_block(
    lon_model,
    lat_model,
    bedmachine_5x["lat"].values,
    bedmachine_5x["lon"].values,
    bedmachine_5x["lat"].values,
    is_stereo=True,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=False,
    algo="fast",
)


mask_5x = np.where(hgrid["area"][0::2, 0::2] > 2500 * 2500, 1, 0)
mask_10x = np.where(hgrid["area"][0::2, 0::2] > 5000 * 5000, 1, 0)

plt.figure()
plt.pcolormesh(mask_5x, cmap="binary_r")
plt.title("5x downsampled")

plt.figure()
plt.pcolormesh(mask_10x, cmap="binary_r")
plt.title("10x downsampled")

# create the coordinate reference system
crs = CRS.from_proj4(PROJSTRING)
# create the projection from lon/lat to x/y
proj = Transformer.from_crs(crs.geodetic_crs, crs)
xx, yy = proj.transform(lon_model, lat_model)

plt.figure()
plt.pcolormesh(xx, yy, mask_5x, cmap="binary_r")
plt.title("5x downsampled")

plt.figure()
plt.pcolormesh(xx, yy, mask_10x, cmap="binary_r")
plt.title("10x downsampled")

plt.figure()
plt.pcolormesh(xx, yy, out10x[0, :, :], vmax=-60)
plt.colorbar()
plt.title("10x downsampled - lat")

plt.figure()
plt.pcolormesh(xx, yy, out10x[4, :, :])
plt.colorbar()
plt.title("10x downsampled - npts")

plt.figure()
plt.pcolormesh(xx, yy, out5x[0, :, :], vmax=-60)
plt.colorbar()
plt.title("5x downsampled - lat")

plt.figure()
plt.pcolormesh(xx, yy, out5x[4, :, :])
plt.colorbar()
plt.title("5x downsampled - npts")

plt.show()


# topo test

thk5x = compute_block(
    lon_model,
    lat_model,
    bedmachine_5x["thickness"].values,
    bedmachine_5x["lon"].values,
    bedmachine_5x["lat"].values,
    is_stereo=True,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=True,
)


plt.figure()
plt.pcolormesh(xx, yy, thk5x[0, :, :], vmax=5000)
plt.colorbar()
plt.title("5x downsampled - thickness")

subplot_kws = dict(
    projection=ccrs.SouthPolarStereo(central_longitude=0.0), facecolor="grey"
)
plt.figure(figsize=[10, 8])
ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0.0))
# ax.stock_img()
plt.pcolormesh(
    lon_model,
    lat_model,
    thk5x[0, :, :],
    shading="auto",
    cmap="jet",
    transform=ccrs.PlateCarree(),
), plt.clim(-0, 4000), plt.colorbar()
ax.set_extent([-180, 180, -55, -90], ccrs.PlateCarree())
ax.gridlines(color="black", alpha=0.5, linestyle="--")

plt.show()
