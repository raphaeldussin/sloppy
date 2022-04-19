import numpy as np

from sloppy.serial import compute_cell_topo_stats


def test_all_points_in_square_cell():

    lon_c = np.array([[9, 10], [9, 10]])
    lat_c = np.array([[19, 19], [20, 20]])

    lon_src, lat_src = np.meshgrid(np.linspace(9, 10, 50), np.linspace(19, 20, 50))
    data_src = 3 + 4 * lon_src + 5 * lat_src + 7 * lon_src * lat_src

    out = compute_cell_topo_stats(
        lon_c, lat_c, lon_src, lat_src, data_src, compute_res=False
    )

    assert isinstance(out, np.ndarray)

    npts = out[4]
    assert npts == len(data_src.flatten())


def test_all_points_out():

    lon_c = np.array([[9, 10], [9, 10]])
    lat_c = np.array([[19, 19], [20, 20]])

    lon_src, lat_src = np.meshgrid(np.linspace(39, 40, 50), np.linspace(19, 20, 50))
    data_src = 3 + 4 * lon_src + 5 * lat_src + 7 * lon_src * lat_src

    out = compute_cell_topo_stats(
        lon_c, lat_c, lon_src, lat_src, data_src, compute_res=False
    )

    npts = out[4]
    assert npts == 0.0


def test_no_points():

    lon_c = np.array([[9, 10], [9, 10]])
    lat_c = np.array([[19, 19], [20, 20]])

    out = compute_cell_topo_stats(
        lon_c, lat_c, np.array([]), np.array([]), np.array([]), compute_res=False
    )


def test_distorted_cell():

    lon_c = np.array([[9, 10], [9.25, 10.25]])
    lat_c = np.array([[19, 19.5], [20, 20.5]])

    lon_src, lat_src = np.meshgrid(np.linspace(9, 10, 50), np.linspace(19, 20, 50))
    data_src = 3 + 4 * lon_src + 5 * lat_src + 7 * lon_src * lat_src

    out = compute_cell_topo_stats(
        lon_c, lat_c, lon_src, lat_src, data_src, compute_res=False
    )

    npts = out[4]
    assert npts < len(data_src.flatten())


def test_carth_geom():

    x_c = np.array([[-5000, 5000], [-5000, 5000]])
    y_c = np.array([[-5000, -5000], [5000, 5000]])

    x_src, y_src = np.meshgrid(
        np.linspace(-10000, 10000, 50), np.linspace(-10000, 10000, 50)
    )
    data_src = 3 + 4 * x_src + 5 * y_src + 7 * x_src * y_src

    out = compute_cell_topo_stats(x_c, y_c, x_src, y_src, data_src, compute_res=False)

    npts = out[4]
    assert npts > 20

    x_src, y_src = np.meshgrid(np.array([0.0]), np.array([0.0]))
    data_src = 3 + 4 * x_src + 5 * y_src + 7 * x_src * y_src

    out = compute_cell_topo_stats(x_c, y_c, x_src, y_src, data_src, compute_res=False)

    npts = out[4]
    assert npts == 1

    x_c = np.array([[-5000, 5000], [-5000, 5000]])
    y_c = np.array([[-5000, -5000], [0, 0]])

    x_src, y_src = np.meshgrid(np.array([0.0]), np.array([-1000.0]))
    data_src = 10 * np.ones((1, 1))
    out = compute_cell_topo_stats(x_c, y_c, x_src, y_src, data_src, compute_res=False)
    assert out[4] == 1

    x_src, y_src = np.meshgrid(np.array([-50.0]), np.array([-500.0]))
    data_src = 10 * np.ones((1, 1))
    out = compute_cell_topo_stats(x_c, y_c, x_src, y_src, data_src, compute_res=False)
    assert out[4] == 1

    x_src, y_src = np.meshgrid(np.array([50.0]), np.array([-500.0]))
    data_src = 10 * np.ones((1, 1))
    out = compute_cell_topo_stats(x_c, y_c, x_src, y_src, data_src, compute_res=False)
    assert out[4] == 1
