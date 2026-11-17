"""High-level chunked and distributed runners for bathymetry & roughness computation."""

from typing import Optional, Tuple, Union
import numpy as np

from sloppy.grid import CornerGrid, project_coords, standardize_lon
from sloppy.mapper import map_points_to_cells
from sloppy.stats import CSRBucketAggregator, StreamingCellAccumulator


def _iter_source_tiles(
    topo: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    chunk_size: Tuple[int, int] = (4096, 4096),
):
    """Yield `(x_tile, y_tile, z_tile)` slices over the 2D source dataset without full meshgrid allocation."""
    ny_s, nx_s = topo.shape
    cy, cx = chunk_size
    is_1d = x_src.ndim == 1 and y_src.ndim == 1

    for j0 in range(0, ny_s, cy):
        j1 = min(j0 + cy, ny_s)
        for i0 in range(0, nx_s, cx):
            i1 = min(i0 + cx, nx_s)
            z_tile = np.asarray(topo[j0:j1, i0:i1])
            if is_1d:
                x_tile = np.asarray(x_src[i0:i1])
                y_tile = np.asarray(y_src[j0:j1])
            else:
                x_tile = np.asarray(x_src[j0:j1, i0:i1])
                y_tile = np.asarray(y_src[j0:j1, i0:i1])
            yield x_tile, y_tile, z_tile


def _process_single_tile(
    grid: CornerGrid,
    x_tile: np.ndarray,
    y_tile: np.ndarray,
    z_tile: np.ndarray,
    project_source: bool,
    streaming: bool,
    workers: int = -1,
) -> Union[StreamingCellAccumulator, CSRBucketAggregator]:
    """Map one source tile to target cells and return its partial accumulator."""
    cell_ids, dx_cell, dy_cell = map_points_to_cells(
        grid, x_tile, y_tile, project_source=project_source, workers=workers
    )
    if streaming:
        acc = StreamingCellAccumulator(
            grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
        )
    else:
        acc = CSRBucketAggregator(
            grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
        )
    acc.update(cell_ids, dx_cell, dy_cell, z_tile)
    return acc


def compute_bathy(
    lon_model: np.ndarray,
    lat_model: np.ndarray,
    topo: np.ndarray,
    topo_lon: np.ndarray,
    topo_lat: np.ndarray,
    method: str = "median",
    residual: bool = True,
    is_carth: bool = False,
    is_stereo: bool = False,
    proj_string: Optional[str] = None,
    cyclic_x: Optional[bool] = None,
    chunk_size: Tuple[int, int] = (4096, 4096),
    engine: str = "serial",
    backend: str = "auto",
    streaming: Optional[bool] = None,
    fill_empty_nearest: bool = False,
    mpi_comm=None,
) -> np.ndarray:
    """Compute target ocean model cell bathymetry and sub-grid roughness using the corner-KDTree + angle test.

    Parameters
    ----------
    lon_model, lat_model : np.ndarray, shape (ny + 1, nx + 1)
        Target model grid cell corner coordinates.
    topo : np.ndarray or xarray.DataArray, shape (ny_src, nx_src)
        Source bathymetry / elevation array.
    topo_lon, topo_lat : np.ndarray
        1D `(nx_src,)`, `(ny_src,)` or 2D `(ny_src, nx_src)` source coordinates.
    method : {"median", "mean"}, default="median"
        Statistic for cell depth (`out[0]`). `"mean"` defaults to `StreamingCellAccumulator`
        with strictly O(N_cells) memory; `"median"` uses `CSRBucketAggregator`.
    residual : bool, default=True
        Whether to compute the 2D linear plane-fit residual squared (`out[3]`).
    is_carth : bool, default=False
        True if input coordinates are already in a planar Cartesian reference frame.
    is_stereo : bool, default=False
        True if input lon/lat should be projected using `proj_string`.
    proj_string : str, optional
        PROJ4 string used when `is_stereo=True`.
    cyclic_x : bool, optional
        Whether the model grid is periodic in X (auto-detected if None).
    chunk_size : tuple of (int, int), default=(4096, 4096)
        2D tile dimensions for streaming source data without full-grid meshgrid memory allocation.
    engine : {"serial", "dask", "mpi", "gpu"}, default="serial"
        Execution engine across source tiles.
    backend : {"auto", "cpu", "scipy", "pykdtree", "gpu"}, default="auto"
        KDTree and array hardware backend.
    streaming : bool, optional
        Force streaming sufficient-statistics accumulator (only valid with `method="mean"`).
        Defaults to True when `method="mean"` and False when `method="median"`.
    fill_empty_nearest : bool, default=False
        If True, target cells with 0 source points (`npts == 0`) fall back to the nearest
        source point elevation at the cell center.
    mpi_comm : mpi4py.MPI.Comm, optional
        MPI communicator when `engine="mpi"`.

    Returns
    -------
    h_out : np.ndarray, shape (5, ny, nx)
        `[depth (median/mean), min_depth, max_depth, roughness_residual2, npts]`
    """
    if method not in ("median", "mean"):
        raise ValueError(f"method must be 'median' or 'mean'; got '{method}'")

    if streaming is None:
        streaming = method == "mean"
    elif streaming and method == "median":
        raise ValueError("Streaming sufficient-statistics mode only supports method='mean'; use streaming=False for exact median.")

    if engine == "gpu":
        backend = "gpu"

    effective_proj = proj_string if (is_stereo and proj_string is not None) else None
    project_source = effective_proj is not None

    grid = CornerGrid(
        lon_model,
        lat_model,
        is_carth=is_carth,
        proj_string=effective_proj,
        cyclic_x=cyclic_x,
        backend=backend,
    )

    x_src = np.asarray(topo_lon)
    y_src = np.asarray(topo_lat)

    if topo.size == 0 or x_src.size == 0 or y_src.size == 0:
        acc = StreamingCellAccumulator(
            grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
        )
        return acc.finalize(compute_res=residual)

    if engine in ("serial", "gpu"):
        if streaming:
            acc = StreamingCellAccumulator(
                grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
            )
        else:
            acc = CSRBucketAggregator(
                grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
            )

        for x_tile, y_tile, z_tile in _iter_source_tiles(topo, x_src, y_src, chunk_size=chunk_size):
            cell_ids, dx_cell, dy_cell = map_points_to_cells(
                grid, x_tile, y_tile, project_source=project_source
            )
            acc.update(cell_ids, dx_cell, dy_cell, z_tile)

        if streaming:
            out = acc.finalize(compute_res=residual)
        else:
            out = acc.finalize(method=method, compute_res=residual)

    elif engine == "dask":
        import dask

        delayed_tasks = []
        for x_tile, y_tile, z_tile in _iter_source_tiles(topo, x_src, y_src, chunk_size=chunk_size):
            task = dask.delayed(_process_single_tile)(
                grid, x_tile, y_tile, z_tile, project_source, streaming, workers=1
            )
            delayed_tasks.append(task)

        partial_accs = dask.compute(*delayed_tasks)
        base_acc = partial_accs[0]
        for other_acc in partial_accs[1:]:
            base_acc.merge(other_acc)

        if streaming:
            out = base_acc.finalize(compute_res=residual)
        else:
            out = base_acc.finalize(method=method, compute_res=residual)

    elif engine == "mpi":
        if mpi_comm is None:
            from mpi4py import MPI

            mpi_comm = MPI.COMM_WORLD

        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()

        tiles = list(_iter_source_tiles(topo, x_src, y_src, chunk_size=chunk_size))
        my_tiles = tiles[rank::size]

        if streaming:
            local_acc = StreamingCellAccumulator(
                grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
            )
        else:
            local_acc = CSRBucketAggregator(
                grid.n_cells, grid.ny, grid.nx, cell_n_singular=grid.cell_n_singular
            )

        for x_tile, y_tile, z_tile in my_tiles:
            cell_ids, dx_cell, dy_cell = map_points_to_cells(
                grid, x_tile, y_tile, project_source=project_source
            )
            local_acc.update(cell_ids, dx_cell, dy_cell, z_tile)

        gathered = mpi_comm.allgather(local_acc)
        merged = gathered[0]
        for other in gathered[1:]:
            merged.merge(other)
        if streaming:
            out = merged.finalize(compute_res=residual)
        else:
            out = merged.finalize(method=method, compute_res=residual)
    else:
        raise ValueError(f"Unknown engine '{engine}'. Choose from 'serial', 'dask', 'mpi', 'gpu'.")

    if fill_empty_nearest:
        _fill_empty_cells_nearest(out, grid, topo, x_src, y_src, project_source)

    return out


def _fill_empty_cells_nearest(
    out: np.ndarray,
    grid: CornerGrid,
    topo: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    project_source: bool,
):
    """Fill target cells where `npts == 0` using nearest-neighbor source value at cell center."""
    empty_mask = (out[4] == 0.0) & (grid.cell_n_singular.reshape(grid.ny, grid.nx) < 2)
    if not np.any(empty_mask):
        return

    # Cell centers in grid coordinates
    cx = (grid.cell_ref_x + np.mean(grid.cell_vx, axis=1)).reshape(grid.ny, grid.nx)
    cy = (grid.cell_ref_y + np.mean(grid.cell_vy, axis=1)).reshape(grid.ny, grid.nx)
    target_x = cx[empty_mask]
    target_y = cy[empty_mask]
    if not grid.is_carth:
        target_x = standardize_lon(target_x)

    if x_src.ndim == 1 and y_src.ndim == 1 and not project_source:
        sx = standardize_lon(x_src) if not grid.is_carth else x_src
        sy = y_src
        ix = np.abs(sx[:, None] - target_x[None, :]).argmin(axis=0)
        iy = np.abs(sy[:, None] - target_y[None, :]).argmin(axis=0)
        vals = np.asarray(topo)[iy, ix]
    else:
        from scipy.spatial import KDTree

        if x_src.ndim == 1 and y_src.ndim == 1:
            xx, yy = np.meshgrid(x_src, y_src)
        else:
            xx, yy = x_src, y_src
        if project_source and grid.proj_string is not None:
            xx, yy = project_coords(xx, yy, grid.proj_string)
        elif not grid.is_carth:
            xx = standardize_lon(xx)
        tree = KDTree(np.column_stack((xx.ravel(), yy.ravel())))
        _, idx = tree.query(np.column_stack((target_x, target_y)), k=1, workers=-1)
        vals = np.asarray(topo).ravel()[idx]

    out[0, empty_mask] = vals
    out[1, empty_mask] = vals
    out[2, empty_mask] = vals
    out[3, empty_mask] = 0.0
    out[4, empty_mask] = 1.0


def compute_block_corner(
    lon_model_block: np.ndarray,
    lat_model_block: np.ndarray,
    topo_subset: np.ndarray,
    topo_lon_subset: np.ndarray,
    topo_lat_subset: np.ndarray,
    is_carth: bool = False,
    is_stereo: bool = False,
    PROJSTRING: Optional[str] = None,
    residual: bool = True,
    method: str = "median",
    engine: str = "serial",
    backend: str = "auto",
) -> np.ndarray:
    """Drop-in replacement for `sloppy.distrib.compute_block` using the corner-KDTree + angle algorithm."""
    return compute_bathy(
        lon_model_block,
        lat_model_block,
        topo_subset,
        topo_lon_subset,
        topo_lat_subset,
        method=method,
        residual=residual,
        is_carth=is_carth,
        is_stereo=is_stereo,
        proj_string=PROJSTRING,
        engine=engine,
        backend=backend,
    )
