import numpy as np

from sloppy.serial import compute_cell_topo_stats


def test_all_points_in_square_cell():

    lon_c = np.array([[9, 10], [9, 10]])
    lat_c = np.array([[19, 19], [20, 20]])

    lon_src, lat_src = np.meshgrid(np.linspace(9, 10, 50), np.linspace(19, 20, 50))
    data_src = 3 + 4 * lon_src + 5 * lat_src + 7 * lon_src * lat_src

    out = compute_cell_topo_stats(lon_c, lat_c, lon_src, lat_src, data_src)

    assert isinstance(out, np.ndarray)

    npts = out[4]
    assert npts == len(data_src.flatten())


def test_all_points_out():

    lon_c = np.array([[9, 10], [9, 10]])
    lat_c = np.array([[19, 19], [20, 20]])

    lon_src, lat_src = np.meshgrid(np.linspace(39, 40, 50), np.linspace(19, 20, 50))
    data_src = 3 + 4 * lon_src + 5 * lat_src + 7 * lon_src * lat_src

    out = compute_cell_topo_stats(lon_c, lat_c, lon_src, lat_src, data_src)

    npts = out[4]
    assert npts == 0.0


def test_no_points():

    lon_c = np.array([[9, 10], [9, 10]])
    lat_c = np.array([[19, 19], [20, 20]])

    out = compute_cell_topo_stats(
        lon_c, lat_c, np.array([]), np.array([]), np.array([])
    )


def test_distorted_cell():

    lon_c = np.array([[9, 10], [9.25, 10.25]])
    lat_c = np.array([[19, 19.5], [20, 20.5]])

    lon_src, lat_src = np.meshgrid(np.linspace(9, 10, 50), np.linspace(19, 20, 50))
    data_src = 3 + 4 * lon_src + 5 * lat_src + 7 * lon_src * lat_src

    out = compute_cell_topo_stats(lon_c, lat_c, lon_src, lat_src, data_src)

    npts = out[4]
    assert npts < len(data_src.flatten())
