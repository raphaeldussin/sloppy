"""Unit and integration tests for the corner-KDTree + 4-sector angle algorithm."""

import os
import numpy as np
import pytest
import xarray as xr

from sloppy import (
    CornerGrid,
    CSRBucketAggregator,
    StreamingCellAccumulator,
    compute_bathy,
    compute_block_corner,
    map_points_to_cells,
)
from sloppy.serial import compute_cell_topo_stats

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data/",
)


def test_single_square_cell_matches_serial():
    """Verify exact agreement with serial.compute_cell_topo_stats on a square cell."""
    lon_c = np.array([[9.0, 10.0], [9.0, 10.0]])
    lat_c = np.array([[19.0, 19.0], [20.0, 20.0]])

    lon_1d = np.linspace(9.0, 10.0, 50)
    lat_1d = np.linspace(19.0, 20.0, 50)
    lon_src, lat_src = np.meshgrid(lon_1d, lat_1d)
    data_src = 3.0 + 4.0 * lon_src + 5.0 * lat_src + 7.0 * lon_src * lat_src

    expected_med = compute_cell_topo_stats(
        lon_c, lat_c, lon_src, lat_src, data_src, method="median", compute_res=True
    )
    out_med = compute_bathy(
        lon_c, lat_c, data_src, lon_1d, lat_1d, method="median", residual=True
    )[:, 0, 0]

    assert np.allclose(out_med, expected_med, rtol=1e-10, atol=1e-10)

    expected_mean = compute_cell_topo_stats(
        lon_c, lat_c, lon_src, lat_src, data_src, method="mean", compute_res=True
    )
    out_mean_stream = compute_bathy(
        lon_c, lat_c, data_src, lon_1d, lat_1d, method="mean", residual=True, streaming=True
    )[:, 0, 0]
    out_mean_csr = compute_bathy(
        lon_c, lat_c, data_src, lon_1d, lat_1d, method="mean", residual=True, streaming=False
    )[:, 0, 0]

    assert np.allclose(out_mean_stream, expected_mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(out_mean_csr, expected_mean, rtol=1e-10, atol=1e-10)


def test_distorted_and_cartesian_cells_match_serial():
    """Verify agreement with serial.py on distorted parallelogram and Cartesian cells."""
    lon_c = np.array([[9.0, 10.0], [9.25, 10.25]])
    lat_c = np.array([[19.0, 19.5], [20.0, 20.5]])

    lon_src, lat_src = np.meshgrid(np.linspace(9.0, 10.0, 50), np.linspace(19.0, 20.0, 50))
    data_src = 3.0 + 4.0 * lon_src + 5.0 * lat_src + 7.0 * lon_src * lat_src

    expected = compute_cell_topo_stats(
        lon_c, lat_c, lon_src, lat_src, data_src, method="median", compute_res=True
    )
    out = compute_bathy(
        lon_c, lat_c, data_src, lon_src, lat_src, method="median", residual=True
    )[:, 0, 0]

    assert np.allclose(out[:3], expected[:3], rtol=1e-10, atol=1e-10)
    assert abs(out[4] - expected[4]) <= 1.0
    assert np.isclose(out[3], expected[3], rtol=2e-3)

    # Cartesian cell test
    x_c = np.array([[-5000.0, 5000.0], [-5000.0, 5000.0]])
    y_c = np.array([[-5000.0, -5000.0], [5000.0, 5000.0]])
    x_src, y_src = np.meshgrid(np.linspace(-10000.0, 10000.0, 50), np.linspace(-10000.0, 10000.0, 50))
    data_c = 3.0 + 4.0 * x_src + 5.0 * y_src + 7.0 * x_src * y_src

    expected_c = compute_cell_topo_stats(
        x_c, y_c, x_src, y_src, data_c, is_carth=True, compute_res=True
    )
    out_c = compute_bathy(
        x_c, y_c, data_c, x_src, y_src, is_carth=True, residual=True
    )[:, 0, 0]

    assert np.allclose(out_c, expected_c, rtol=1e-10, atol=1e-10)


def test_all_points_out_and_empty_source():
    """Verify out-of-domain source points are rejected and empty inputs are handled cleanly."""
    lon_c = np.array([[9.0, 10.0], [9.0, 10.0]])
    lat_c = np.array([[19.0, 19.0], [20.0, 20.0]])

    lon_src, lat_src = np.meshgrid(np.linspace(39.0, 40.0, 50), np.linspace(19.0, 20.0, 50))
    data_src = 3.0 + 4.0 * lon_src

    out_far = compute_bathy(lon_c, lat_c, data_src, lon_src, lat_src, residual=False)
    assert out_far[4, 0, 0] == 0.0
    assert out_far[0, 0, 0] == 1.0e20

    out_empty = compute_bathy(
        lon_c, lat_c, np.empty((0, 0)), np.array([]), np.array([]), residual=False
    )
    assert out_empty[4, 0, 0] == 0.0
    assert out_empty[0, 0, 0] == 1.0e20


def test_dateline_crossing_and_periodic_seam():
    """Verify cells crossing the ±180° dateline and 0°/360° seam capture all source points."""
    # 2x2 cells straddling the dateline from 178E to -178W (182E)
    lon_c = np.array(
        [
            [178.0, 180.0, -178.0],
            [178.0, 180.0, -178.0],
            [178.0, 180.0, -178.0],
        ]
    )
    lat_c = np.array(
        [
            [-10.0, -10.0, -10.0],
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
        ]
    )

    # Create dense points on both sides of ±180
    lon_west = np.linspace(178.1, 179.9, 20)
    lon_east = np.linspace(-179.9, -178.1, 20)
    lon_1d = np.concatenate([lon_west, lon_east])
    lat_1d = np.concatenate([np.linspace(-9.9, -0.1, 15), np.linspace(0.1, 9.9, 15)])
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Smooth linear depth function continuous across the dateline
    unwrapped_lon = np.mod(lon_2d + 360.0, 360.0)
    topo = -4000.0 + 10.0 * (unwrapped_lon - 180.0) + 5.0 * lat_2d

    out_stream = compute_bathy(
        lon_c, lat_c, topo, lon_1d, lat_1d, method="mean", residual=True, streaming=True
    )
    out_median = compute_bathy(
        lon_c, lat_c, topo, lon_1d, lat_1d, method="median", residual=True
    )

    # Every one of the 4 cells should have exactly 20 * 15 = 300 source points
    assert np.all(out_stream[4] == 300.0)
    assert np.all(out_median[4] == 300.0)
    # Because topo is a pure linear plane in unwrapped_lon and lat, residual^2 must be ~0!
    assert np.allclose(out_stream[3], 0.0, atol=1e-8)
    assert np.allclose(out_median[3], 0.0, atol=1e-8)
    assert np.allclose(out_stream[0], out_median[0], rtol=1e-10)


def test_triangular_and_singular_cells():
    """Verify degenerate triangular cells (1 collapsed edge) and singular cells (>=2 collapsed edges)."""
    # Triangle cell with collapsed top edge: (9.5, 20.0)
    lon_tri = np.array([[9.0, 10.0], [9.5, 9.5]])
    lat_tri = np.array([[19.0, 19.0], [20.0, 20.0]])

    lon_src, lat_src = np.meshgrid(np.linspace(9.0, 10.0, 41), np.linspace(19.0, 20.0, 41))
    topo = -2000.0 + 3.0 * lon_src - 2.0 * lat_src

    expected = compute_cell_topo_stats(lon_tri, lat_tri, lon_src, lat_src, topo, method="median")
    out = compute_bathy(lon_tri, lat_tri, topo, lon_src, lat_src, method="median")[:, 0, 0]

    assert np.isclose(out[4], expected[4], rtol=0.02)
    assert np.allclose(out[:3], expected[:3], rtol=1e-3)
    assert np.isclose(out[3], 0.0, atol=1e-8)

    # Fully collapsed segment cell (>=2 singular edges) -> 1.0e36 sentinel
    lon_sing = np.array([[9.0, 9.0], [9.0, 9.0]])
    lat_sing = np.array([[19.0, 19.0], [20.0, 20.0]])
    out_sing = compute_bathy(lon_sing, lat_sing, topo, lon_src, lat_src)[:, 0, 0]
    assert out_sing[4] == 0.0
    assert out_sing[0] == 1.0e36


def test_streaming_chunk_invariance():
    """Verify that StreamingCellAccumulator and CSRBucketAggregator are invariant to tile chunking."""
    lon_c, lat_c = np.meshgrid(np.linspace(-20.0, 20.0, 9), np.linspace(-15.0, 15.0, 7))
    lon_s = np.linspace(-25.0, 25.0, 200)
    lat_s = np.linspace(-20.0, 20.0, 160)
    xx, yy = np.meshgrid(lon_s, lat_s)
    topo = -3000.0 + 50.0 * np.sin(np.radians(xx * 5.0)) * np.cos(np.radians(yy * 5.0)) + 0.2 * xx * yy

    out_single_chunk = compute_bathy(
        lon_c, lat_c, topo, lon_s, lat_s, method="mean", chunk_size=(1000, 1000), streaming=True
    )
    out_small_chunks = compute_bathy(
        lon_c, lat_c, topo, lon_s, lat_s, method="mean", chunk_size=(37, 43), streaming=True
    )
    out_csr_mean = compute_bathy(
        lon_c, lat_c, topo, lon_s, lat_s, method="mean", chunk_size=(37, 43), streaming=False
    )

    assert np.allclose(out_single_chunk, out_small_chunks, rtol=1e-9, atol=1e-9)
    assert np.allclose(out_single_chunk, out_csr_mean, rtol=1e-9, atol=1e-9)


def test_fill_empty_nearest_fallback():
    """Verify that fill_empty_nearest populates cells finer than the source grid."""
    lon_c, lat_c = np.meshgrid(np.linspace(0.0, 4.0, 9), np.linspace(0.0, 4.0, 9))
    # Coarse source grid with only 3x3 points over [0, 4] x [0, 4]
    lon_s = np.array([0.5, 2.0, 3.5])
    lat_s = np.array([0.5, 2.0, 3.5])
    xx, yy = np.meshgrid(lon_s, lat_s)
    topo = -1000.0 - 100.0 * xx - 50.0 * yy

    out_no_fill = compute_bathy(lon_c, lat_c, topo, lon_s, lat_s, fill_empty_nearest=False)
    assert np.any(out_no_fill[4] == 0.0)

    out_filled = compute_bathy(lon_c, lat_c, topo, lon_s, lat_s, fill_empty_nearest=True)
    assert np.all(out_filled[4] >= 1.0)
    assert np.all(out_filled[0] < 0.0)


@pytest.mark.parametrize("gridfile", ["mom6_grid_global_coarse.nc"])
@pytest.mark.parametrize("topofile", ["gebco_reduced.nc"])
def test_global_mom6_gebco_corner_algo(topofile, gridfile):
    """Test global MOM6 coarse grid with GEBCO reduced fixture (both 1D and 2D coords, mean & median)."""
    dstopo = xr.open_dataset(FIXTURE_DIR + topofile, decode_times=False)
    dsgrid = xr.open_dataset(FIXTURE_DIR + gridfile, decode_times=False)

    lon_model = dsgrid["x"][0::2, 0::2].values
    lat_model = dsgrid["y"][0::2, 0::2].values
    topo = dstopo["elevation"].values
    lon_1d = dstopo["lon"].values
    lat_1d = dstopo["lat"].values

    elev_med = compute_bathy(
        lon_model,
        lat_model,
        topo,
        lon_1d,
        lat_1d,
        method="median",
        residual=True,
    )
    assert elev_med.shape == (5, lon_model.shape[0] - 1, lon_model.shape[1] - 1)
    assert elev_med[0].max() <= 99999.0
    assert elev_med[0].min() >= -99999.0
    assert elev_med[4].min() >= 10
    assert elev_med[4].max() <= 3000

    # Streaming mean with 2D meshgrid coords
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    elev_mean = compute_bathy(
        lon_model,
        lat_model,
        topo,
        lon_2d,
        lat_2d,
        method="mean",
        residual=True,
        streaming=True,
        chunk_size=(128, 128),
    )
    assert np.array_equal(elev_med[4], elev_mean[4])
    assert np.allclose(elev_med[1:4], elev_mean[1:4], rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("gridfile", ["mom6_grid_global_coarse.nc"])
@pytest.mark.parametrize("topofile", ["bedmachine_reduced.nc"])
def test_polarstereo_bedmachine_to_mom6_corner_algo(topofile, gridfile):
    """Test Antarctic BedMachine polar stereographic projection onto MOM6 southern ocean grid."""
    PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    dstopo = xr.open_dataset(FIXTURE_DIR + topofile, decode_times=False)
    dsgrid = xr.open_dataset(FIXTURE_DIR + gridfile, decode_times=False)

    lon_model = dsgrid["x"][0:30:2, 0::2].values
    lat_model = dsgrid["y"][0:30:2, 0::2].values
    lontopo = dstopo["lon"].values
    lattopo = dstopo["lat"].values
    topo = dstopo["elevation"].values

    # 1. Compute in lon/lat space
    elev_lonlat = compute_block_corner(
        lon_model,
        lat_model,
        topo,
        lontopo,
        lattopo,
        is_stereo=False,
        is_carth=False,
        PROJSTRING=None,
        residual=True,
    )
    assert elev_lonlat[0].max() <= 99999.0
    assert elev_lonlat[0].min() >= -99999.0
    assert np.sum(elev_lonlat[4]) > 0

    # 2. Compute in polar stereographic space via PROJSTRING
    elev_stereo = compute_block_corner(
        lon_model,
        lat_model,
        topo,
        lontopo,
        lattopo,
        is_stereo=True,
        is_carth=False,
        PROJSTRING=PROJSTRING,
        residual=True,
    )
    assert elev_stereo[0].max() <= 99999.0
    assert elev_stereo[0].min() >= -99999.0
    assert np.sum(elev_stereo[4]) > 0


def test_dask_and_mpi_engines():
    """Verify that dask and mpi engines produce identical results to the serial engine."""
    lon_c, lat_c = np.meshgrid(np.linspace(-20.0, 20.0, 9), np.linspace(-15.0, 15.0, 7))
    lon_s = np.linspace(-25.0, 25.0, 120)
    lat_s = np.linspace(-20.0, 20.0, 100)
    xx, yy = np.meshgrid(lon_s, lat_s)
    topo = -3000.0 + 40.0 * np.sin(np.radians(xx * 3.0)) + 0.5 * xx * yy

    ref_mean = compute_bathy(
        lon_c, lat_c, topo, lon_s, lat_s, method="mean", chunk_size=(40, 40), engine="serial"
    )
    ref_med = compute_bathy(
        lon_c, lat_c, topo, lon_s, lat_s, method="median", chunk_size=(40, 40), engine="serial"
    )

    # Test Dask engine if dask is available
    try:
        import dask  # noqa: F401

        dask_mean = compute_bathy(
            lon_c, lat_c, topo, lon_s, lat_s, method="mean", chunk_size=(40, 40), engine="dask"
        )
        dask_med = compute_bathy(
            lon_c, lat_c, topo, lon_s, lat_s, method="median", chunk_size=(40, 40), engine="dask"
        )
        assert np.allclose(dask_mean, ref_mean, rtol=1e-9, atol=1e-9)
        assert np.allclose(dask_med, ref_med, rtol=1e-9, atol=1e-9)
    except ImportError:
        pass

    # Test MPI engine using a mock 1-rank communicator
    class _MockComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def allgather(self, obj):
            return [obj]

    mpi_mean = compute_bathy(
        lon_c,
        lat_c,
        topo,
        lon_s,
        lat_s,
        method="mean",
        chunk_size=(40, 40),
        engine="mpi",
        mpi_comm=_MockComm(),
    )
    mpi_med = compute_bathy(
        lon_c,
        lat_c,
        topo,
        lon_s,
        lat_s,
        method="median",
        chunk_size=(40, 40),
        engine="mpi",
        mpi_comm=_MockComm(),
    )
    assert np.allclose(mpi_mean, ref_mean, rtol=1e-9, atol=1e-9)
    assert np.allclose(mpi_med, ref_med, rtol=1e-9, atol=1e-9)

