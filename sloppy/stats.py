"""Cell bathymetry and roughness statistics: Welford/Chan streaming sufficient statistics and CSR bucketed exact median."""

from typing import Optional, Tuple
import numpy as np
from numba import njit, prange


@njit
def accumulate_streaming_stats_kernel(
    cell_ids: np.ndarray,
    dx_cell: np.ndarray,
    dy_cell: np.ndarray,
    z_pts: np.ndarray,
    counts: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    moments: np.ndarray,
):
    """Accumulate per-cell counts, min, max, and Welford centered co-moments in a single pass.

    `moments[c]` stores:
        0: mean_x
        1: mean_y
        2: mean_z
        3: C_xx = sum((dx - mean_x)^2)
        4: C_xy = sum((dx - mean_x)*(dy - mean_y))
        5: C_yy = sum((dy - mean_y)^2)
        6: C_xz = sum((dx - mean_x)*(z - mean_z))
        7: C_yz = sum((dy - mean_y)*(z - mean_z))
        8: C_zz = sum((z - mean_z)^2)
    """
    n_pts = cell_ids.shape[0]
    for p in range(n_pts):
        c = cell_ids[p]
        if c < 0:
            continue
        z = float(z_pts[p])
        if np.isnan(z):
            continue

        dx = dx_cell[p]
        dy = dy_cell[p]

        n_old = counts[c]
        n_new = n_old + 1
        counts[c] = n_new

        if z < z_min[c]:
            z_min[c] = z
        if z > z_max[c]:
            z_max[c] = z

        mx = moments[c, 0]
        my = moments[c, 1]
        mz = moments[c, 2]

        ddx = dx - mx
        ddy = dy - my
        ddz = z - mz

        inv_n = 1.0 / float(n_new)
        moments[c, 0] = mx + ddx * inv_n
        moments[c, 1] = my + ddy * inv_n
        moments[c, 2] = mz + ddz * inv_n

        if n_old > 0:
            w = float(n_old) * inv_n
            moments[c, 3] += w * ddx * ddx
            moments[c, 4] += w * ddx * ddy
            moments[c, 5] += w * ddy * ddy
            moments[c, 6] += w * ddx * ddz
            moments[c, 7] += w * ddy * ddz
            moments[c, 8] += w * ddz * ddz


@njit
def merge_streaming_stats_kernel(
    counts_a: np.ndarray,
    z_min_a: np.ndarray,
    z_max_a: np.ndarray,
    moments_a: np.ndarray,
    counts_b: np.ndarray,
    z_min_b: np.ndarray,
    z_max_b: np.ndarray,
    moments_b: np.ndarray,
):
    """Merge accumulator B into accumulator A in-place using Chan's parallel covariance formula."""
    n_cells = counts_a.shape[0]
    for c in range(n_cells):
        nb = counts_b[c]
        if nb == 0:
            continue
        if z_min_b[c] < z_min_a[c]:
            z_min_a[c] = z_min_b[c]
        if z_max_b[c] > z_max_a[c]:
            z_max_a[c] = z_max_b[c]

        na = counts_a[c]
        if na == 0:
            counts_a[c] = nb
            for m in range(9):
                moments_a[c, m] = moments_b[c, m]
            continue

        nab = na + nb
        counts_a[c] = nab

        ddx = moments_b[c, 0] - moments_a[c, 0]
        ddy = moments_b[c, 1] - moments_a[c, 1]
        ddz = moments_b[c, 2] - moments_a[c, 2]

        fb = float(nb) / float(nab)
        moments_a[c, 0] += ddx * fb
        moments_a[c, 1] += ddy * fb
        moments_a[c, 2] += ddz * fb

        w = float(na) * fb
        moments_a[c, 3] += moments_b[c, 3] + w * ddx * ddx
        moments_a[c, 4] += moments_b[c, 4] + w * ddx * ddy
        moments_a[c, 5] += moments_b[c, 5] + w * ddy * ddy
        moments_a[c, 6] += moments_b[c, 6] + w * ddx * ddz
        moments_a[c, 7] += moments_b[c, 7] + w * ddy * ddz
        moments_a[c, 8] += moments_b[c, 8] + w * ddz * ddz


@njit(parallel=True)
def finalize_streaming_stats_kernel(
    counts: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    moments: np.ndarray,
    cell_n_singular: np.ndarray,
    compute_res: bool,
) -> np.ndarray:
    """Finalize Welford/Chan centered moments into (5, n_cells) output: [mean, min, max, roughness, npts]."""
    n_cells = counts.shape[0]
    out = np.empty((5, n_cells), dtype=np.float64)

    for c in prange(n_cells):
        if cell_n_singular[c] >= 2:
            out[0, c] = 1.0e36
            out[1, c] = 1.0e36
            out[2, c] = 1.0e36
            out[3, c] = 1.0e36
            out[4, c] = 0.0
            continue

        n = counts[c]
        if n == 0:
            out[0, c] = 1.0e20
            out[1, c] = 1.0e20
            out[2, c] = 1.0e20
            out[3, c] = 0.0
            out[4, c] = 0.0
            continue

        nf = float(n)
        out[0, c] = moments[c, 2]
        out[1, c] = z_min[c]
        out[2, c] = z_max[c]
        out[4, c] = nf

        roughness = 0.0
        if compute_res and n >= 4:
            cxx = moments[c, 3]
            cxy = moments[c, 4]
            cyy = moments[c, 5]
            cxz = moments[c, 6]
            cyz = moments[c, 7]
            czz = moments[c, 8]

            det = cxx * cyy - cxy * cxy
            scale = cxx + cyy
            if scale > 0.0 and det > 1e-20 * scale * scale:
                bx = (cyy * cxz - cxy * cyz) / det
                by = (cxx * cyz - cxy * cxz) / det
                res2 = czz - (bx * cxz + by * cyz)
                if res2 > 0.0:
                    roughness = res2

        out[3, c] = roughness

    return out


class StreamingCellAccumulator:
    """Fixed-memory O(N_cells) accumulator for online/distributed bathymetry statistics.

    Uses Welford's online algorithm and Chan's parallel covariance merge formula for
    numerical stability across arbitrary depths and chunk sizes.
    """

    def __init__(self, n_cells: int, ny: int, nx: int, cell_n_singular: Optional[np.ndarray] = None):
        self.n_cells = int(n_cells)
        self.ny = int(ny)
        self.nx = int(nx)
        if cell_n_singular is None:
            self.cell_n_singular = np.zeros(self.n_cells, dtype=np.int8)
        else:
            self.cell_n_singular = np.ascontiguousarray(cell_n_singular, dtype=np.int8)

        self.counts = np.zeros(self.n_cells, dtype=np.int64)
        self.z_min = np.full(self.n_cells, np.inf, dtype=np.float64)
        self.z_max = np.full(self.n_cells, -np.inf, dtype=np.float64)
        self.moments = np.zeros((self.n_cells, 9), dtype=np.float64)

    def update(self, cell_ids: np.ndarray, dx_cell: np.ndarray, dy_cell: np.ndarray, z_pts: np.ndarray):
        """Accumulate a batch of mapped source points into the cell statistics."""
        if cell_ids.size == 0:
            return
        accumulate_streaming_stats_kernel(
            cell_ids,
            dx_cell,
            dy_cell,
            np.asarray(z_pts, dtype=np.float64).ravel(),
            self.counts,
            self.z_min,
            self.z_max,
            self.moments,
        )

    def merge(self, other: "StreamingCellAccumulator") -> "StreamingCellAccumulator":
        """Merge another StreamingCellAccumulator into this one in-place and return self."""
        merge_streaming_stats_kernel(
            self.counts,
            self.z_min,
            self.z_max,
            self.moments,
            other.counts,
            other.z_min,
            other.z_max,
            other.moments,
        )
        return self

    def finalize(self, compute_res: bool = True) -> np.ndarray:
        """Compute final (5, ny, nx) statistics array: [mean, min, max, roughness, npts]."""
        out_flat = finalize_streaming_stats_kernel(
            self.counts,
            self.z_min,
            self.z_max,
            self.moments,
            self.cell_n_singular,
            compute_res,
        )
        return out_flat.reshape((5, self.ny, self.nx))


@njit
def build_csr_buckets_kernel(
    cell_ids: np.ndarray,
    dx_cell: np.ndarray,
    dy_cell: np.ndarray,
    z_pts: np.ndarray,
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group valid points by cell_id into CSR arrays in O(N_pts + N_cells) time."""
    n_pts = cell_ids.shape[0]
    counts = np.zeros(n_cells, dtype=np.int64)

    for p in range(n_pts):
        c = cell_ids[p]
        if c >= 0 and not np.isnan(z_pts[p]):
            counts[c] += 1

    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    for c in range(n_cells):
        offsets[c + 1] = offsets[c] + counts[c]

    total_valid = offsets[n_cells]
    dx_csr = np.empty(total_valid, dtype=np.float64)
    dy_csr = np.empty(total_valid, dtype=np.float64)
    z_csr = np.empty(total_valid, dtype=np.float64)
    cursor = offsets[:-1].copy()

    for p in range(n_pts):
        c = cell_ids[p]
        if c >= 0:
            z = float(z_pts[p])
            if not np.isnan(z):
                idx = cursor[c]
                cursor[c] = idx + 1
                dx_csr[idx] = dx_cell[p]
                dy_csr[idx] = dy_cell[p]
                z_csr[idx] = z

    return offsets, dx_csr, dy_csr, z_csr


@njit(parallel=True)
def compute_csr_cell_stats_kernel(
    offsets: np.ndarray,
    dx_csr: np.ndarray,
    dy_csr: np.ndarray,
    z_csr: np.ndarray,
    cell_n_singular: np.ndarray,
    use_median: bool,
    compute_res: bool,
) -> np.ndarray:
    """Compute exact per-cell statistics (median or mean, min, max, roughness, npts) from CSR buckets."""
    n_cells = offsets.shape[0] - 1
    out = np.empty((5, n_cells), dtype=np.float64)

    for c in prange(n_cells):
        if cell_n_singular[c] >= 2:
            out[0, c] = 1.0e36
            out[1, c] = 1.0e36
            out[2, c] = 1.0e36
            out[3, c] = 1.0e36
            out[4, c] = 0.0
            continue

        start = offsets[c]
        end = offsets[c + 1]
        npts = end - start

        if npts == 0:
            out[0, c] = 1.0e20
            out[1, c] = 1.0e20
            out[2, c] = 1.0e20
            out[3, c] = 0.0
            out[4, c] = 0.0
            continue

        z_slice = z_csr[start:end]
        dmin = np.min(z_slice)
        dmax = np.max(z_slice)
        if use_median:
            dout = np.median(z_slice)
        else:
            dout = np.mean(z_slice)

        roughness = 0.0
        if compute_res and npts >= 4:
            A = np.empty((npts, 3), dtype=np.float64)
            for k in range(npts):
                A[k, 0] = dx_csr[start + k]
                A[k, 1] = dy_csr[start + k]
                A[k, 2] = 1.0
            _, residual2, _, _ = np.linalg.lstsq(A, z_slice)
            if len(residual2) == 1:
                roughness = residual2[0]

        out[0, c] = dout
        out[1, c] = dmin
        out[2, c] = dmax
        out[3, c] = roughness
        out[4, c] = float(npts)

    return out


class CSRBucketAggregator:
    """Collects mapped cell points and computes exact median/mean and plane-fit roughness via CSR buckets."""

    def __init__(self, n_cells: int, ny: int, nx: int, cell_n_singular: Optional[np.ndarray] = None):
        self.n_cells = int(n_cells)
        self.ny = int(ny)
        self.nx = int(nx)
        if cell_n_singular is None:
            self.cell_n_singular = np.zeros(self.n_cells, dtype=np.int8)
        else:
            self.cell_n_singular = np.ascontiguousarray(cell_n_singular, dtype=np.int8)

        self._cell_id_chunks = []
        self._dx_chunks = []
        self._dy_chunks = []
        self._z_chunks = []

    def update(self, cell_ids: np.ndarray, dx_cell: np.ndarray, dy_cell: np.ndarray, z_pts: np.ndarray):
        """Append valid mapped points (cell_id >= 0) from a source chunk."""
        if cell_ids.size == 0:
            return
        z_flat = np.asarray(z_pts, dtype=np.float64).ravel()
        valid = (cell_ids >= 0) & np.isfinite(z_flat)
        if np.any(valid):
            self._cell_id_chunks.append(np.ascontiguousarray(cell_ids[valid], dtype=np.int64))
            self._dx_chunks.append(np.ascontiguousarray(dx_cell[valid], dtype=np.float64))
            self._dy_chunks.append(np.ascontiguousarray(dy_cell[valid], dtype=np.float64))
            self._z_chunks.append(np.ascontiguousarray(z_flat[valid], dtype=np.float64))

    def merge(self, other: "CSRBucketAggregator") -> "CSRBucketAggregator":
        """Merge another CSRBucketAggregator's collected chunks into this one."""
        self._cell_id_chunks.extend(other._cell_id_chunks)
        self._dx_chunks.extend(other._dx_chunks)
        self._dy_chunks.extend(other._dy_chunks)
        self._z_chunks.extend(other._z_chunks)
        return self

    def finalize(self, method: str = "median", compute_res: bool = True) -> np.ndarray:
        """Bucket all collected points into CSR layout and compute per-cell statistics."""
        if len(self._cell_id_chunks) == 0:
            all_ids = np.empty(0, dtype=np.int64)
            all_dx = np.empty(0, dtype=np.float64)
            all_dy = np.empty(0, dtype=np.float64)
            all_z = np.empty(0, dtype=np.float64)
        elif len(self._cell_id_chunks) == 1:
            all_ids = self._cell_id_chunks[0]
            all_dx = self._dx_chunks[0]
            all_dy = self._dy_chunks[0]
            all_z = self._z_chunks[0]
        else:
            all_ids = np.concatenate(self._cell_id_chunks)
            all_dx = np.concatenate(self._dx_chunks)
            all_dy = np.concatenate(self._dy_chunks)
            all_z = np.concatenate(self._z_chunks)

        offsets, dx_csr, dy_csr, z_csr = build_csr_buckets_kernel(
            all_ids, all_dx, all_dy, all_z, self.n_cells
        )
        use_median = method == "median"
        out_flat = compute_csr_cell_stats_kernel(
            offsets,
            dx_csr,
            dy_csr,
            z_csr,
            self.cell_n_singular,
            use_median,
            compute_res,
        )
        return out_flat.reshape((5, self.ny, self.nx))
