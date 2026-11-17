"""Hardware backend abstraction for KDTree indexing and array execution (CPU & GPU)."""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

try:
    from pykdtree.kdtree import KDTree as PyKDTree

    HAS_PYKDTREE = True
except ImportError:
    PyKDTree = None
    HAS_PYKDTREE = False

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from numba import cuda

    HAS_NUMBA_CUDA = cuda.is_available()
except Exception:
    cuda = None
    HAS_NUMBA_CUDA = False


class CornerKDTreeIndex:
    """Wrapper around CPU (pykdtree / scipy) or GPU KDTree for model grid corners.

    Maps query points `(x, y)` to canonical corner indices `0 .. N_corners - 1`,
    transparently resolving periodic ghost corners added near the dateline.
    """

    def __init__(
        self,
        tree_coords: np.ndarray,
        tree_to_corner_id: np.ndarray,
        backend: str = "auto",
        leafsize: int = 16,
    ):
        """
        Parameters
        ----------
        tree_coords : np.ndarray, shape (N_tree, 2)
            2D coordinates (x/lon, y/lat) of corners and any periodic ghost corners.
        tree_to_corner_id : np.ndarray, shape (N_tree,), dtype=int64
            Mapping from KDTree point index to canonical corner index in [0, Ny_c * Nx_c).
        backend : {"auto", "cpu", "scipy", "pykdtree", "gpu"}
            Backend selection.
        leafsize : int
            KDTree leaf size.
        """
        self.tree_coords = np.ascontiguousarray(tree_coords, dtype=np.float64)
        self.tree_to_corner_id = np.ascontiguousarray(tree_to_corner_id, dtype=np.int64)
        self.backend = self._resolve_backend(backend)
        self.leafsize = leafsize

        self._tree = None
        self._gpu_coords = None
        self._gpu_corner_ids = None
        self._gpu_tree = None

        if self.backend == "pykdtree":
            self._tree = PyKDTree(self.tree_coords, leafsize=leafsize)
        elif self.backend in ("cpu", "scipy"):
            self._tree = ScipyKDTree(self.tree_coords, leafsize=leafsize)
        elif self.backend == "gpu":
            self._init_gpu_index()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        backend = backend.lower()
        if backend == "auto":
            if HAS_PYKDTREE:
                return "pykdtree"
            return "scipy"
        if backend == "cpu":
            return "pykdtree" if HAS_PYKDTREE else "scipy"
        if backend == "pykdtree":
            if not HAS_PYKDTREE:
                raise RuntimeError("pykdtree is not installed; use backend='cpu' or 'scipy'")
            return "pykdtree"
        if backend == "scipy":
            return "scipy"
        if backend == "gpu":
            if not (HAS_CUPY or HAS_NUMBA_CUDA):
                raise RuntimeError(
                    "GPU backend requested, but neither CuPy nor an active Numba CUDA device is available."
                )
            return "gpu"
        raise ValueError(f"Unknown backend '{backend}'. Choose from 'auto', 'cpu', 'scipy', 'pykdtree', 'gpu'.")

    def _init_gpu_index(self):
        """Initialize GPU KDTree or device corner coordinate arrays."""
        if HAS_CUPY:
            self._gpu_coords = cp.asarray(self.tree_coords, dtype=cp.float64)
            self._gpu_corner_ids = cp.asarray(self.tree_to_corner_id, dtype=cp.int64)
            try:
                from cupyx.scipy.spatial import KDTree as CupyKDTree

                self._gpu_tree = CupyKDTree(self._gpu_coords, leafsize=self.leafsize)
            except Exception:
                self._gpu_tree = None
        elif HAS_NUMBA_CUDA:
            self._gpu_coords = cuda.to_device(self.tree_coords)
            self._gpu_corner_ids = cuda.to_device(self.tree_to_corner_id)

    def query(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        workers: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest canonical corner index and squared distance for query points.

        Parameters
        ----------
        x_pts : np.ndarray, 1D
            Flattened x/longitude coordinates of source points.
        y_pts : np.ndarray, 1D
            Flattened y/latitude coordinates of source points.
        workers : int
            Number of CPU threads for KDTree query (-1 uses all cores).

        Returns
        -------
        corner_ids : np.ndarray, shape (N_pts,), dtype=int64
            Canonical corner index `J * Nx_c + I` for each query point.
        dist2 : np.ndarray, shape (N_pts,), dtype=float64
            Squared Euclidean distance to the matched corner.
        """
        pts = np.column_stack((x_pts, y_pts))
        if self.backend == "pykdtree":
            dists, idx = self._tree.query(np.ascontiguousarray(pts, dtype=np.float64), k=1)
            return self.tree_to_corner_id[idx], np.square(dists, dtype=np.float64)
        elif self.backend == "scipy":
            dists, idx = self._tree.query(pts, k=1, workers=workers)
            return self.tree_to_corner_id[idx], np.square(dists, dtype=np.float64)
        elif self.backend == "gpu":
            return self._query_gpu(pts)
        raise RuntimeError(f"Invalid backend state: {self.backend}")

    def _query_gpu(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Query nearest corners on GPU using cupyx KDTree or tiled pairwise kernel."""
        if HAS_CUPY and self._gpu_tree is not None:
            pts_gpu = cp.asarray(pts, dtype=cp.float64)
            dists_gpu, idx_gpu = self._gpu_tree.query(pts_gpu, k=1)
            corner_ids = cp.asnumpy(self._gpu_corner_ids[idx_gpu])
            dist2 = cp.asnumpy(cp.square(dists_gpu))
            return corner_ids, dist2

        if HAS_CUPY:
            # Batched matrix distance computation on GPU when cupyx KDTree is unavailable
            pts_gpu = cp.asarray(pts, dtype=cp.float64)
            n_pts = pts_gpu.shape[0]
            corner_ids_out = np.empty(n_pts, dtype=np.int64)
            dist2_out = np.empty(n_pts, dtype=np.float64)
            batch_size = max(1, min(65536, (256 * 1024 * 1024) // max(1, self._gpu_coords.shape[0] * 8)))
            c_norm2 = cp.sum(cp.square(self._gpu_coords), axis=1)[None, :]  # (1, N_tree)
            for start in range(0, n_pts, batch_size):
                end = min(start + batch_size, n_pts)
                sub = pts_gpu[start:end]  # (B, 2)
                p_norm2 = cp.sum(cp.square(sub), axis=1)[:, None]
                d2 = p_norm2 + c_norm2 - 2.0 * (sub @ self._gpu_coords.T)
                idx = cp.argmin(d2, axis=1)
                min_d2 = cp.take_along_axis(d2, idx[:, None], axis=1).ravel()
                corner_ids_out[start:end] = cp.asnumpy(self._gpu_corner_ids[idx])
                dist2_out[start:end] = cp.asnumpy(cp.maximum(min_d2, 0.0))
            return corner_ids_out, dist2_out

        # Fallback if only numba.cuda is present without cupy
        tree = ScipyKDTree(self.tree_coords, leafsize=self.leafsize)
        dists, idx = tree.query(pts, k=1, workers=-1)
        return self.tree_to_corner_id[idx], np.square(dists, dtype=np.float64)
