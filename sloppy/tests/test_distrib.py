# requires pytest-datafiles
import xarray as xr
import numpy as np
import os
import pytest


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data/",
)


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


@pytest.mark.parametrize("gridfile", ["mom6_grid_global_coarse.nc"])
@pytest.mark.parametrize("topofile", ["gebco_reduced.nc"])
def test_regular_to_mom6global_brute(topofile, gridfile):

    from sloppy.distrib import compute_block_brute

    # read files
    dstopo = xr.open_dataset(FIXTURE_DIR + topofile, decode_times=False)
    dsgrid = xr.open_dataset(FIXTURE_DIR + gridfile, decode_times=False)

    if len(dstopo["lon"].values.shape) == 1:
        # gebco has 1d lon/lat, needs to be 2d
        lontopo, lattopo = np.meshgrid(dstopo["lon"].values, dstopo["lat"].values)
    else:
        lontopo, lattopo = dstopo["lon"].values, dstopo["lat"].values

    assert len(lontopo.shape) == 2
    assert len(lattopo.shape) == 2

    # compute elevation
    elev = compute_block_brute(
        dsgrid["x"][0::2, 0::2].values,
        dsgrid["y"][0::2, 0::2].values,
        dstopo["elevation"].values,
        lontopo,
        lattopo,
        is_stereo=False,
        is_carth=False,
        PROJSTRING=None,
        residual=True,
    )

    assert elev[0, :, :].max() <= 99999.0
    assert elev[0, :, :].min() >= -99999.0

    assert elev[4, :, :].min() >= 10
    assert elev[4, :, :].max() <= 3000


@pytest.mark.parametrize("gridfile", ["mom6_grid_global_coarse.nc"])
@pytest.mark.parametrize("topofile", ["bedmachine_reduced.nc"])
def test_polarstereo_to_mom6_brute(topofile, gridfile):

    from sloppy.distrib import compute_block_brute

    # read files
    dstopo = xr.open_dataset(FIXTURE_DIR + topofile, decode_times=False)
    dsgrid = xr.open_dataset(FIXTURE_DIR + gridfile, decode_times=False)

    lontopo, lattopo = dstopo["lon"].values, dstopo["lat"].values

    assert len(lontopo.shape) == 2
    assert len(lattopo.shape) == 2

    # take corner points on MOM6 grid, only south of 60S
    lon_model = dsgrid["x"][0:30:2, 0::2].values
    lat_model = dsgrid["y"][0:30:2, 0::2].values

    # compute in lon/lat space
    elev1 = compute_block_brute(
        lon_model,
        lat_model,
        dstopo["elevation"].values,
        lontopo,
        lattopo,
        is_stereo=False,
        is_carth=False,
        PROJSTRING=None,
        residual=True,
    )

    assert elev1[0, :, :].max() <= 99999.0
    assert elev1[0, :, :].min() >= -99999.0

    # compute in carthesian (polar stereo) space

    PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    xx_model, yy_model = proj_xy(lon_model, lat_model, PROJSTRING)

    if len(dstopo["x"].values.shape) == 1:
        xtopo, ytopo = np.meshgrid(dstopo["x"], dstopo["y"])

    elev2 = compute_block_brute(
        xx_model,
        yy_model,
        dstopo["elevation"].values,
        xtopo,
        ytopo,
        is_stereo=False,  # not used in brute
        is_carth=True,
        PROJSTRING=None,  # not used in brute
        residual=True,
    )

    assert elev2[0, :, :].max() <= 99999.0
    assert elev2[0, :, :].min() >= -99999.0


@pytest.mark.parametrize("gridfile", ["mom6_grid_global_coarse.nc"])
@pytest.mark.parametrize("topofile", ["gebco_reduced.nc"])
def test_regular_to_mom6global_optim(topofile, gridfile):

    from sloppy.distrib import compute_block

    # read files
    dstopo = xr.open_dataset(FIXTURE_DIR + topofile, decode_times=False)
    dsgrid = xr.open_dataset(FIXTURE_DIR + gridfile, decode_times=False)

    lontopo, lattopo = np.meshgrid(dstopo["lon"].values, dstopo["lat"].values)
    assert len(lontopo.shape) == 2
    assert len(lattopo.shape) == 2

    ## compute elevation
    elev = compute_block(
        dsgrid["x"][0::2, 0::2].values,
        dsgrid["y"][0::2, 0::2].values,
        dstopo["elevation"].values,
        lontopo,
        lattopo,
        is_stereo=False,
        is_carth=False,
        PROJSTRING=None,
        residual=True,
    )

    assert elev[0, :, :].max() <= 99999.0
    assert elev[0, :, :].min() >= -99999.0

    # compute elevation
    elev = compute_block(
        dsgrid["x"][0::2, 0::2].values,
        dsgrid["y"][0::2, 0::2].values,
        dstopo["elevation"].values,
        dstopo["lon"].values,
        dstopo["lat"].values,
        is_stereo=False,
        is_carth=False,
        PROJSTRING=None,
        residual=True,
    )

    assert elev[0, :, :].max() <= 99999.0
    assert elev[0, :, :].min() >= -99999.0


@pytest.mark.parametrize("gridfile", ["mom6_grid_global_coarse.nc"])
@pytest.mark.parametrize("topofile", ["bedmachine_reduced.nc"])
def test_polarstereo_to_mom6_optim(topofile, gridfile):

    from sloppy.distrib import compute_block

    PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    # read files
    dstopo = xr.open_dataset(FIXTURE_DIR + topofile, decode_times=False)
    dsgrid = xr.open_dataset(FIXTURE_DIR + gridfile, decode_times=False)

    lontopo, lattopo = dstopo["lon"].values, dstopo["lat"].values

    assert len(lontopo.shape) == 2
    assert len(lattopo.shape) == 2

    # take corner points on MOM6 grid, only south of 60S
    lon_model = dsgrid["x"][0:30:2, 0::2].values
    lat_model = dsgrid["y"][0:30:2, 0::2].values

    # compute in lon/lat space
    elev1 = compute_block(
        lon_model,
        lat_model,
        dstopo["elevation"].values,
        lontopo,
        lattopo,
        is_stereo=True,
        is_carth=False,
        PROJSTRING=PROJSTRING,
        residual=True,
    )

    assert elev1[0, :, :].max() <= 99999.0
    assert elev1[0, :, :].min() >= -99999.0

    # compute in carthesian (polar stereo) space

    xx_model, yy_model = proj_xy(lon_model, lat_model, PROJSTRING)

    if len(dstopo["x"].values.shape) == 1:
        xtopo, ytopo = np.meshgrid(dstopo["x"], dstopo["y"])

    elev2 = compute_block(
        xx_model,
        yy_model,
        dstopo["elevation"].values,
        xtopo,
        ytopo,
        is_stereo=False,
        is_carth=True,
        PROJSTRING=None,
        residual=True,
    )

    assert elev2[0, :, :].max() <= 99999.0
    assert elev2[0, :, :].min() >= -99999.0
