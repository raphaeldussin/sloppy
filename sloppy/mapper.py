"""Point-to-cell mapping using nearest model corner KDTree lookup and 4-sector angle test."""

from typing import Tuple
import numpy as np
from numba import njit, prange

from sloppy.grid import CornerGrid, project_coords, standardize_lon

TWO_PI = 2.0 * np.pi


@njit(inline="always")
def _unwrap_dx(x: float, x_ref: float, is_carth: bool) -> float:
    if is_carth:
        return x - x_ref
    diff = (x - x_ref + 180.0) % 360.0
    if diff < 0.0:
        diff += 360.0
    return diff - 180.0


@njit(inline="always")
def _pos_mod_2pi(angle: float) -> float:
    res = angle % TWO_PI
    if res < 0.0:
        res += TWO_PI
    return res


@njit(inline="always")
def _point_in_cell_ccw(
    x_p: float,
    y_p: float,
    c: int,
    cell_ref_x: np.ndarray,
    cell_ref_y: np.ndarray,
    cell_vx: np.ndarray,
    cell_vy: np.ndarray,
    cell_max_r2: np.ndarray,
    cell_n_singular: np.ndarray,
    is_carth: bool,
) -> Tuple[bool, float, float]:
    """Check if point (x_p, y_p) lies inside cell `c` and return local offsets (dx_c, dy_c)."""
    if cell_n_singular[c] >= 2:
        return False, 0.0, 0.0

    dx_c = _unwrap_dx(x_p, cell_ref_x[c], is_carth)
    dy_c = y_p - cell_ref_y[c]

    if (dx_c * dx_c + dy_c * dy_c) > cell_max_r2[c]:
        return False, dx_c, dy_c

    for m in range(4):
        m_next = (m + 1) & 3
        vx0 = cell_vx[c, m]
        vy0 = cell_vy[c, m]
        ex = cell_vx[c, m_next] - vx0
        ey = cell_vy[c, m_next] - vy0
        len2 = ex * ex + ey * ey
        if len2 > 1e-20:
            cross = ex * (dy_c - vy0) - ey * (dx_c - vx0)
            if cross < -1e-14 * np.sqrt(len2):
                return False, dx_c, dy_c

    return True, dx_c, dy_c


@njit(parallel=True)
def assign_points_to_cells_kernel(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    nearest_corner_ids: np.ndarray,
    dist2: np.ndarray,
    corner_x: np.ndarray,
    corner_y: np.ndarray,
    corner_sector_start: np.ndarray,
    corner_sector_span: np.ndarray,
    corner_adj_cells: np.ndarray,
    corner_max_dist2: np.ndarray,
    corner_extra_offsets: np.ndarray,
    corner_extra_cells: np.ndarray,
    cell_ref_x: np.ndarray,
    cell_ref_y: np.ndarray,
    cell_vx: np.ndarray,
    cell_vy: np.ndarray,
    cell_max_r2: np.ndarray,
    cell_n_singular: np.ndarray,
    is_carth: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each source point to a target ocean model cell via the 4-sector angle test."""
    n_pts = x_pts.shape[0]
    cell_ids = np.full(n_pts, -1, dtype=np.int64)
    dx_cell = np.zeros(n_pts, dtype=np.float64)
    dy_cell = np.zeros(n_pts, dtype=np.float64)

    for p in prange(n_pts):
        xp = x_pts[p]
        yp = y_pts[p]
        if np.isnan(xp) or np.isnan(yp):
            continue

        k = nearest_corner_ids[p]
        if k < 0 or dist2[p] > corner_max_dist2[k]:
            continue

        # Local unwrapped offset from matched corner k
        dx_k = _unwrap_dx(xp, corner_x[k], is_carth)
        dy_k = yp - corner_y[k]
        theta_p = np.arctan2(dy_k, dx_k)

        matched_c = -1
        matched_q = -1

        # 1. Primary 4-sector angle test around corner k
        for q in range(4):
            span = corner_sector_span[k, q]
            if span >= 0.0:
                rel_angle = _pos_mod_2pi(theta_p - corner_sector_start[k, q])
                if rel_angle <= span + 1e-12:
                    c_cand = corner_adj_cells[k, q]
                    if c_cand >= 0:
                        inside, dxc, dyc = _point_in_cell_ccw(
                            xp,
                            yp,
                            c_cand,
                            cell_ref_x,
                            cell_ref_y,
                            cell_vx,
                            cell_vy,
                            cell_max_r2,
                            cell_n_singular,
                            is_carth,
                        )
                        if inside:
                            matched_c = c_cand
                            dx_cell[p] = dxc
                            dy_cell[p] = dyc
                            break
                        matched_q = q

        # 2. Fallback check across remaining cells sharing corner k (handles boundary corners & skewed cells)
        if matched_c < 0:
            for q in range(4):
                if q == matched_q:
                    continue
                c_cand = corner_adj_cells[k, q]
                if c_cand >= 0:
                    inside, dxc, dyc = _point_in_cell_ccw(
                        xp,
                        yp,
                        c_cand,
                        cell_ref_x,
                        cell_ref_y,
                        cell_vx,
                        cell_vy,
                        cell_max_r2,
                        cell_n_singular,
                        is_carth,
                    )
                    if inside:
                        matched_c = c_cand
                        dx_cell[p] = dxc
                        dy_cell[p] = dyc
                        break

        # 3. Extended fan fallback (only non-empty for corners in tripolar singular fans or high-aspect-ratio cells)
        if matched_c < 0:
            ext_start = corner_extra_offsets[k]
            ext_end = corner_extra_offsets[k + 1]
            for idx in range(ext_start, ext_end):
                c_cand = corner_extra_cells[idx]
                inside, dxc, dyc = _point_in_cell_ccw(
                    xp,
                    yp,
                    c_cand,
                    cell_ref_x,
                    cell_ref_y,
                    cell_vx,
                    cell_vy,
                    cell_max_r2,
                    cell_n_singular,
                    is_carth,
                )
                if inside:
                    matched_c = c_cand
                    dx_cell[p] = dxc
                    dy_cell[p] = dyc
                    break

        cell_ids[p] = matched_c

    return cell_ids, dx_cell, dy_cell


def map_points_to_cells(
    grid: CornerGrid,
    x_src: np.ndarray,
    y_src: np.ndarray,
    project_source: bool = False,
    workers: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map source coordinates (1D vectors or 2D arrays) to target grid cell indices."""
    x_arr = np.asarray(x_src, dtype=np.float64)
    y_arr = np.asarray(y_src, dtype=np.float64)

    if x_arr.size == 0 or y_arr.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    if x_arr.ndim == 1 and y_arr.ndim == 1:
        x_2d, y_2d = np.meshgrid(x_arr, y_arr)
        x_flat = x_2d.ravel()
        y_flat = y_2d.ravel()
    elif x_arr.ndim == 2 and y_arr.ndim == 2:
        if x_arr.shape != y_arr.shape:
            raise ValueError(f"2D x_src and y_src must have the same shape; got {x_arr.shape} vs {y_arr.shape}")
        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()
    else:
        raise ValueError("x_src and y_src must both be 1D vectors or both be 2D arrays.")

    if project_source and grid.proj_string is not None:
        x_flat, y_flat = project_coords(x_flat, y_flat, grid.proj_string)
    elif not grid.is_carth:
        x_flat = standardize_lon(x_flat)

    nearest_corners, dist2 = grid.kdtree.query(x_flat, y_flat, workers=workers)

    return assign_points_to_cells_kernel(
        x_flat,
        y_flat,
        nearest_corners,
        dist2,
        grid.corner_x,
        grid.corner_y,
        grid.corner_sector_start,
        grid.corner_sector_span,
        grid.corner_adj_cells,
        grid.corner_max_dist2,
        grid.corner_extra_offsets,
        grid.corner_extra_cells,
        grid.cell_ref_x,
        grid.cell_ref_y,
        grid.cell_vx,
        grid.cell_vy,
        grid.cell_max_r2,
        grid.cell_n_singular,
        grid.is_carth,
    )
