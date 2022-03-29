""" test_coords.py : module to test with different coordinates """

from tkinter import W

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import KDTree

from sloppy.serial import find_nearest_point

# NumPy random generator seed
np.random.seed(42)

# longitude and latitude for sample GEBCO grid
lon_gebco = np.linspace(-180, 180, 500)
lat_gebco = np.linspace(-90, 90, 500)
elevation = np.random.randint(-10000, 10000 + 1, size=(len(lat_gebco), len(lon_gebco)))

# convert to xarray.DataArray
da_lon_gebco = xr.DataArray(lon_gebco, dims=("lon"), coords={"lon": lon_gebco})
da_lat_gebco = xr.DataArray(lat_gebco, dims=("lat"), coords={"lat": lat_gebco})
da_elevation = xr.DataArray(elevation, dims=("lat", "lon"))

# GEBCO test dataset
dset = xr.Dataset(
    {
        "lon": da_lon_gebco,
        "lat": da_lat_gebco,
        "elevation": da_elevation,
    }
)

# lon/lat bedmachine
x_bedmachine = np.linspace(-3333000, 3333000, 500)
y_bedmachine = np.linspace(-3333000, 3333000, 500)
# proj string:
PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
# create the coordinate reference system
crs = CRS.from_proj4(PROJSTRING)
# create the projection from lon/lat to x/y
proj = Transformer.from_crs(crs.geodetic_crs, crs)
# make x,y 2d arrays
xx, yy = np.meshgrid(x_bedmachine, y_bedmachine)
# compute the lon/lat
lon_bedmachine, lat_bedmachine = proj.transform(xx, yy, direction="INVERSE")
# make some sample data
bed = np.random.normal(1.0, 1.0, size=(len(lat_bedmachine), len(lon_bedmachine)))

# convert to xarray.DataArray
da_lon_bedmachine = xr.DataArray(
    lon_bedmachine, dims=("latitude", "longitude")
)  # ,coords={"latitude":lat_bedmachine,"longitude":lon_bedmachine})
da_lat_bedmachine = xr.DataArray(
    lat_bedmachine, dims=("latitude", "longitude")
)  # ,coords={"latitude":lat_bedmachine,"longitude":lon_bedmachine})
da_bed = xr.DataArray(bed, dims=("latitude", "longitude"))

# BEDMACHINE test dataset
dset = xr.Dataset(
    {
        "LONG": da_lon_bedmachine,
        "LAT": da_lat_bedmachine,
        "BED": da_bed,
    }
)


def test_find_nearest_point_1d__1():
    """unit test for 1-dimensional coordinates"""
    ifound, jfound = find_nearest_point(lon_gebco, lat_gebco, 80.0, 60.0)
    assert np.allclose(lon_gebco[ifound], 80.0, rtol=0.01)
    assert np.allclose(lat_gebco[jfound], 60.0, rtol=0.01)
    # assert 0 == 1


def test_find_nearest_point_2d__1():
    """unit test for 2-dimensional coordinates"""

    lon_bedmachine360 = np.mod(lon_bedmachine + 360, 360)
    tree = KDTree(list(zip(lon_bedmachine360.flatten(), lat_bedmachine.flatten())))
    ifound, jfound = find_nearest_point(
        lon_bedmachine360, lat_bedmachine, 80.0, -70.0, tree=tree
    )
    assert np.allclose(lon_bedmachine[jfound, ifound], 80.0, rtol=0.01)
    assert np.allclose(lat_bedmachine[jfound, ifound], -70.0, rtol=0.01)


def test_find_nearest_point_2d__2():
    """test that a missing kdtree raises an error"""
    with pytest.raises(ValueError):
        find_nearest_point(lon_bedmachine, lat_bedmachine, 80.0, 60.0)
