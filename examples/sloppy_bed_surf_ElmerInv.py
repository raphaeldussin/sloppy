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
    lon = np.mod(lon+360.,360.)
    # add to dataset
    ds["lon"] = xr.DataArray(data=lon, dims=("y", "x"))
    ds["lat"] = xr.DataArray(data=lat, dims=("y", "x"))
    ds["lon"].attrs = dict(units="degrees_east")
    ds["lat"].attrs = dict(units="degrees_north")
    return ds



PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

#---------------- bedmachine + reduction

#bedmachine = xr.open_dataset(
#    "/net2/rnd/BedMachineAntarctica_2020-07-15_v02.nc"
#)

elmerInv = xr.open_dataset(
   "/archive/ovs/elmerAnt_inv/inv_MuBeta4km_SI_fld.nc"
)

xx_bm_full, yy_bm_full = np.meshgrid(elmerInv["x"].values, elmerInv["y"].values)

#----------------- read lon/lat MOM6 grid corners until 60S

#gridname = "73E82S_025deg"
gridname = "SP_025deg"
hgrid = xr.open_dataset(f"/work/ovs/iOM4/{gridname}.nc")
iOM4dir = "/work/ovs/iOM4/"
j60s=602

hgrid = hgrid.set_coords(["x", "y"])

lon_model = hgrid["x"][0:j60s:2, 0::2]
lon_model = np.mod(lon_model+360.,360.)
lat_model = hgrid["y"][0:j60s:2, 0::2]

xx_model, yy_model = proj_xy(lon_model, lat_model, PROJSTRING)
lon_lat_elm=add_lon_lat(elmerInv, PROJSTRING)

print(xx_model.shape)

#plt.figure()
#plt.pcolormesh(lon_lat_elm["lon"].values)
#plt.colorbar()
#
#plt.figure()
#plt.pcolormesh(lon_lat_elm["lat"].values)
#plt.colorbar()
#
#plt.show()



### remapping

out = xr.Dataset()

#nu = compute_block(
#    lon_model.values,
#    lat_model.values,
#    #elmerInv["nu_fld"].values,
#    elmerInv["thickness"].values,
#    lon_lat_elm["lon"].values,
#    lon_lat_elm["lat"].values,
#    is_stereo=False,
#    is_carth=False,
#    PROJSTRING=PROJSTRING,
#    residual=False,
#)

nu = compute_block(
    xx_model,
    yy_model,
    #elmerInv["nu_fld"].values,
    elmerInv["thickness"].values,
    xx_bm_full,
    yy_bm_full,
    is_stereo=False,
    is_carth=True,
    PROJSTRING=PROJSTRING,
    residual=False,
)

out = xr.Dataset()
out["nu"] = xr.DataArray(data=nu[0, :, :], dims=("y", "x"))
out.to_netcdf(f"nu_ElmverInv_remapped_iOM4_{gridname}.nc")

