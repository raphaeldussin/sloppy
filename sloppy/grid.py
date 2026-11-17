"""Target ocean model grid indexing: corner KDTree, periodic ghost corners, and 4-sector edge angles."""

from typing import Optional, Tuple
import numpy as np

from sloppy.backend import CornerKDTreeIndex


def standardize_lon(lon: np.ndarray) -> np.ndarray:
    """Standardize longitude in degrees to [-180.0, 180.0)."""
    return np.mod(lon + 180.0, 360.0) - 180.0


def unwrap_lon_offset(lon: np.ndarray, lon_ref: np.ndarray) -> np.ndarray:
    """Compute locally unwrapped longitude difference `lon - lon_ref` in [-180.0, 180.0)."""
    return np.mod(lon - lon_ref + 180.0, 360.0) - 180.0


def project_coords(
    lon: np.ndarray, lat: np.ndarray, proj_string: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Project geographical (lon, lat) coordinates to Cartesian (x, y) using pyproj."""
    from pyproj import CRS, Transformer

    crs = CRS.from_proj4(proj_string)
    proj = Transformer.from_crs(crs.geodetic_crs, crs)
    xx, yy = proj.transform(lon, lat)
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


class CornerGrid:
    """Precomputed target model grid representation indexed by cell corners.

    For a model grid with `(ny, nx)` cells and `(ny + 1, nx + 1)` corners:
    1. Builds a 2D KDTree on the corners (augmented with periodic dateline ghost
       corners when `is_carth=False`).
    2. Precomputes the 4 radiating edge angles `(East, North, West, South)` at
       each corner `(J, I)` and the angular sector spans of the 4 adjacent cells:
       - Quadrant 0 (NE): cell `(J, I)`
       - Quadrant 1 (NW): cell `(J, I - 1)`
       - Quadrant 2 (SW): cell `(J - 1, I - 1)`
       - Quadrant 3 (SE): cell `(J - 1, I)`
    3. Precomputes local counterclockwise vertex offsets for each cell and a CSR
       fallback list (`corner_extra_cells`) for tripolar singular pole fans or
       strongly anisotropic cells where a non-incident corner lies within a cell's
       circumcircle.
    """

    def __init__(
        self,
        x_corners: np.ndarray,
        y_corners: np.ndarray,
        is_carth: bool = False,
        proj_string: Optional[str] = None,
        cyclic_x: Optional[bool] = None,
        ghost_margin_deg: float = 25.0,
        backend: str = "auto",
    ):
        x_c = np.asarray(x_corners, dtype=np.float64)
        y_c = np.asarray(y_corners, dtype=np.float64)
        if x_c.ndim != 2 or y_c.ndim != 2 or x_c.shape != y_c.shape:
            raise ValueError(
                f"x_corners and y_corners must be 2D arrays of identical shape; got {x_c.shape} and {y_c.shape}"
            )
        if x_c.shape[0] < 2 or x_c.shape[1] < 2:
            raise ValueError("Corner arrays must have shape at least (2, 2) to define at least 1 cell.")

        self.proj_string = proj_string
        if proj_string is not None:
            x_c, y_c = project_coords(x_c, y_c, proj_string)
            is_carth = True

        self.is_carth = bool(is_carth)
        if not self.is_carth:
            x_c = standardize_lon(x_c)

        self.ny_c, self.nx_c = x_c.shape
        self.ny = self.ny_c - 1
        self.nx = self.nx_c - 1
        self.n_corners = self.ny_c * self.nx_c
        self.n_cells = self.ny * self.nx

        # Auto-detect periodic X seam when geographical and not explicitly specified
        if cyclic_x is None:
            if not self.is_carth and self.nx >= 3:
                dx_seam = unwrap_lon_offset(x_c[:, 0], x_c[:, -1])
                dy_seam = y_c[:, 0] - y_c[:, -1]
                cyclic_x = bool(
                    np.allclose(dx_seam, 0.0, atol=1e-4) and np.allclose(dy_seam, 0.0, atol=1e-4)
                )
            else:
                cyclic_x = False
        self.cyclic_x = bool(cyclic_x)

        self.x_corners_2d = np.ascontiguousarray(x_c, dtype=np.float64)
        self.y_corners_2d = np.ascontiguousarray(y_c, dtype=np.float64)
        self.corner_x = self.x_corners_2d.ravel()
        self.corner_y = self.y_corners_2d.ravel()

        self._build_cell_geometry()
        self._build_corner_topology_and_angles()
        self._build_kdtree(ghost_margin_deg=ghost_margin_deg, backend=backend)
        self._build_extra_cell_candidates()

    def _build_cell_geometry(self):
        """Precompute per-cell reference origin, CCW vertex offsets, and degenerate flags."""
        x_c = self.x_corners_2d
        y_c = self.y_corners_2d
        n_cells = self.n_cells

        # Reference origin for cell (j, i) is its SW corner (j, i)
        ref_x = np.ascontiguousarray(x_c[:-1, :-1].ravel(), dtype=np.float64)
        ref_y = np.ascontiguousarray(y_c[:-1, :-1].ravel(), dtype=np.float64)

        # Gather the 4 corners of each cell: SW(0,0), SE(0,1), NE(1,1), NW(1,0)
        raw_x = np.empty((n_cells, 4), dtype=np.float64)
        raw_y = np.empty((n_cells, 4), dtype=np.float64)

        raw_x[:, 0] = x_c[:-1, :-1].ravel()
        raw_x[:, 1] = x_c[:-1, 1:].ravel()
        raw_x[:, 2] = x_c[1:, 1:].ravel()
        raw_x[:, 3] = x_c[1:, :-1].ravel()

        raw_y[:, 0] = y_c[:-1, :-1].ravel()
        raw_y[:, 1] = y_c[:-1, 1:].ravel()
        raw_y[:, 2] = y_c[1:, 1:].ravel()
        raw_y[:, 3] = y_c[1:, :-1].ravel()

        if not self.is_carth:
            dx = unwrap_lon_offset(raw_x, ref_x[:, None])
        else:
            dx = raw_x - ref_x[:, None]
        dy = raw_y - ref_y[:, None]

        # Count coinciding corners (to detect degenerate cells with >=2 collapsed edges)
        tol = 1e-6
        n_singular = (
            (np.isclose(dx[:, 2], dx[:, 3], atol=tol) & np.isclose(dy[:, 2], dy[:, 3], atol=tol)).astype(np.int32)
            + (np.isclose(dx[:, 1], dx[:, 0], atol=tol) & np.isclose(dy[:, 1], dy[:, 0], atol=tol)).astype(np.int32)
            + (np.isclose(dx[:, 3], dx[:, 0], atol=tol) & np.isclose(dy[:, 3], dy[:, 0], atol=tol)).astype(np.int32)
            + (np.isclose(dx[:, 2], dx[:, 1], atol=tol) & np.isclose(dy[:, 2], dy[:, 1], atol=tol)).astype(np.int32)
        )
        self.cell_n_singular = np.ascontiguousarray(n_singular, dtype=np.int8)

        # Order the 4 vertices strictly counterclockwise around their centroid
        cx = dx.mean(axis=1, keepdims=True)
        cy = dy.mean(axis=1, keepdims=True)
        ang = np.mod(np.arctan2(dy - cy, dx - cx) + 2.0 * np.pi, 2.0 * np.pi)
        order = np.argsort(ang, axis=1)

        cell_vx = np.take_along_axis(dx, order, axis=1)
        cell_vy = np.take_along_axis(dy, order, axis=1)

        self.cell_ref_x = ref_x
        self.cell_ref_y = ref_y
        self.cell_vx = np.ascontiguousarray(cell_vx, dtype=np.float64)
        self.cell_vy = np.ascontiguousarray(cell_vy, dtype=np.float64)
        self.cell_max_r2 = np.ascontiguousarray(
            np.max(cell_vx**2 + cell_vy**2, axis=1) * 1.5 + 1e-9, dtype=np.float64
        )

    def _build_corner_topology_and_angles(self):
        """Precompute 4 radiating edge angles, sector spans, and 4 adjacent cell IDs per corner."""
        ny_c, nx_c = self.ny_c, self.nx_c
        ny, nx = self.ny, self.nx
        x_c = self.x_corners_2d
        y_c = self.y_corners_2d

        adj_cells = np.full((ny_c, nx_c, 4), -1, dtype=np.int64)
        JJ, II = np.meshgrid(np.arange(ny_c), np.arange(nx_c), indexing="ij")

        for q, (dj, di) in enumerate([(0, 0), (0, -1), (-1, -1), (-1, 0)]):
            jc = JJ + dj
            ic = II + di
            if self.cyclic_x:
                ic = np.mod(ic, nx)
                valid = (jc >= 0) & (jc < ny)
            else:
                valid = (jc >= 0) & (jc < ny) & (ic >= 0) & (ic < nx)

            cell_idx = jc * nx + ic
            adj_cells[:, :, q] = np.where(valid, cell_idx, -1)

        edge_angles = np.full((ny_c, nx_c, 4), np.nan, dtype=np.float64)
        edge_valid = np.zeros((ny_c, nx_c, 4), dtype=np.bool_)

        dir_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for d, (dj, di) in enumerate(dir_offsets):
            jn = JJ + dj
            in_ = II + di
            if self.cyclic_x and di != 0:
                in_ = np.mod(in_, nx)
                valid = (jn >= 0) & (jn < ny_c)
            else:
                valid = (jn >= 0) & (jn < ny_c) & (in_ >= 0) & (in_ < nx_c)

            jn_safe = np.clip(jn, 0, ny_c - 1)
            in_safe = np.clip(in_, 0, nx_c - 1)

            xn = x_c[jn_safe, in_safe]
            yn = y_c[jn_safe, in_safe]
            if not self.is_carth:
                dx = unwrap_lon_offset(xn, x_c)
            else:
                dx = xn - x_c
            dy = yn - y_c

            len2 = dx * dx + dy * dy
            valid = valid & (len2 > 1e-20)
            edge_valid[:, :, d] = valid
            edge_angles[:, :, d] = np.where(valid, np.arctan2(dy, dx), 0.0)

        sector_start = np.zeros((ny_c, nx_c, 4), dtype=np.float64)
        sector_span = np.zeros((ny_c, nx_c, 4), dtype=np.float64)

        two_pi = 2.0 * np.pi
        for q in range(4):
            d1 = q
            d2 = (q + 1) % 4
            both_valid = edge_valid[:, :, d1] & edge_valid[:, :, d2] & (adj_cells[:, :, q] >= 0)
            th1 = edge_angles[:, :, d1]
            th2 = edge_angles[:, :, d2]
            diff = np.mod(th2 - th1, two_pi)
            ccw = diff <= np.pi
            s_start = np.where(ccw, th1, th2)
            s_span = np.where(ccw, diff, two_pi - diff)

            sector_start[:, :, q] = np.where(both_valid, s_start, 0.0)
            sector_span[:, :, q] = np.where(both_valid, s_span, -1.0)

        corner_max_dist2 = np.zeros(self.n_corners, dtype=np.float64)
        adj_flat = adj_cells.reshape(self.n_corners, 4)
        for q in range(4):
            cid = adj_flat[:, q]
            mask = cid >= 0
            valid_cids = cid[mask]
            if not self.is_carth:
                dx_c = unwrap_lon_offset(self.cell_ref_x[valid_cids], self.corner_x[mask])[:, None] + self.cell_vx[valid_cids]
            else:
                dx_c = (self.cell_ref_x[valid_cids] - self.corner_x[mask])[:, None] + self.cell_vx[valid_cids]
            dy_c = (self.cell_ref_y[valid_cids] - self.corner_y[mask])[:, None] + self.cell_vy[valid_cids]
            max_d2 = np.max(dx_c * dx_c + dy_c * dy_c, axis=1)
            corner_max_dist2[mask] = np.maximum(corner_max_dist2[mask], max_d2 * 2.25 + 1e-6)

        self.corner_edge_angles = np.ascontiguousarray(edge_angles.reshape(self.n_corners, 4), dtype=np.float64)
        self.corner_sector_start = np.ascontiguousarray(sector_start.reshape(self.n_corners, 4), dtype=np.float64)
        self.corner_sector_span = np.ascontiguousarray(sector_span.reshape(self.n_corners, 4), dtype=np.float64)
        self.corner_adj_cells = np.ascontiguousarray(adj_flat, dtype=np.int64)
        self.corner_max_dist2 = np.ascontiguousarray(corner_max_dist2, dtype=np.float64)

    def _build_kdtree(self, ghost_margin_deg: float, backend: str):
        """Build KDTree on corners, adding ±360° ghost corners near the dateline if geographical."""
        base_coords = np.column_stack((self.corner_x, self.corner_y))
        base_ids = np.arange(self.n_corners, dtype=np.int64)

        valid_mask = np.isfinite(base_coords[:, 0]) & np.isfinite(base_coords[:, 1])
        base_coords = base_coords[valid_mask]
        base_ids = base_ids[valid_mask]

        if not self.is_carth and base_coords.shape[0] > 0:
            max_dlon = float(np.max(np.abs(self.cell_vx))) if self.n_cells > 0 else 5.0
            margin = max(float(ghost_margin_deg), 2.0 * max_dlon)

            west_mask = base_coords[:, 0] < (-180.0 + margin)
            east_mask = base_coords[:, 0] > (180.0 - margin)

            coords_list = [base_coords]
            ids_list = [base_ids]

            if np.any(west_mask):
                west_ghost = base_coords[west_mask].copy()
                west_ghost[:, 0] += 360.0
                coords_list.append(west_ghost)
                ids_list.append(base_ids[west_mask])

            if np.any(east_mask):
                east_ghost = base_coords[east_mask].copy()
                east_ghost[:, 0] -= 360.0
                coords_list.append(east_ghost)
                ids_list.append(base_ids[east_mask])

            tree_coords = np.vstack(coords_list)
            tree_ids = np.concatenate(ids_list)
        else:
            tree_coords = base_coords
            tree_ids = base_ids

        self.kdtree = CornerKDTreeIndex(tree_coords, tree_ids, backend=backend)

    def _build_extra_cell_candidates(self):
        """Build CSR fallback candidate list for corners near singular pole fans or high-aspect-ratio cells.

        By the triangle inequality, any point P inside cell `c` lies within `r_cell` of cell `c`'s
        center `(cx, cy)` and within `r_cell` of at least one of cell `c`'s 4 corners. Therefore,
        any non-incident corner `k` that could be the 2D nearest neighbor to any point P inside
        cell `c` must lie within distance `2 * r_cell` of `(cx, cy)`.
        """
        from scipy.spatial import KDTree as _ScipyKDTree

        extra_lists = [set() for _ in range(self.n_corners)]

        cx_local = np.mean(self.cell_vx, axis=1)
        cy_local = np.mean(self.cell_vy, axis=1)
        r_cell = np.sqrt(
            np.max((self.cell_vx - cx_local[:, None]) ** 2 + (self.cell_vy - cy_local[:, None]) ** 2, axis=1)
        )

        cx = self.cell_ref_x + cx_local
        cy = self.cell_ref_y + cy_local
        if not self.is_carth:
            cx = standardize_lon(cx)

        valid_cells = np.where((self.cell_n_singular < 2) & np.isfinite(cx) & np.isfinite(cy))[0]
        if valid_cells.size > 0 and self.kdtree.tree_coords.shape[0] > 0:
            ball_tree = _ScipyKDTree(self.kdtree.tree_coords)
            tree_to_cid = self.kdtree.tree_to_corner_id

            for c in valid_cells:
                search_r = float(2.0 * r_cell[c] + 1e-6)
                tree_hits = ball_tree.query_ball_point([cx[c], cy[c]], r=search_r)
                adj_set = int(c)
                max_d2_c = (2.0 * r_cell[c]) ** 2 + 1e-4
                for tidx in tree_hits:
                    k = int(tree_to_cid[tidx])
                    adj = self.corner_adj_cells[k]
                    if adj_set != adj[0] and adj_set != adj[1] and adj_set != adj[2] and adj_set != adj[3]:
                        extra_lists[k].add(adj_set)
                    if max_d2_c > self.corner_max_dist2[k]:
                        self.corner_max_dist2[k] = max_d2_c

        offsets = np.zeros(self.n_corners + 1, dtype=np.int64)
        for k in range(self.n_corners):
            offsets[k + 1] = offsets[k] + len(extra_lists[k])

        total_extra = offsets[self.n_corners]
        extra_cells = np.empty(total_extra, dtype=np.int64)
        pos = 0
        for k in range(self.n_corners):
            for c in sorted(extra_lists[k]):
                extra_cells[pos] = c
                pos += 1

        self.corner_extra_offsets = offsets
        self.corner_extra_cells = extra_cells
