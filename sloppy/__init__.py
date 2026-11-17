"""sloppy: Ocean model bathymetry and sub-grid scale roughness generator."""

from sloppy.grid import CornerGrid
from sloppy.mapper import map_points_to_cells
from sloppy.runner import compute_bathy, compute_block_corner
from sloppy.stats import CSRBucketAggregator, StreamingCellAccumulator
from sloppy.version import __version__

__all__ = [
    "CornerGrid",
    "CSRBucketAggregator",
    "StreamingCellAccumulator",
    "compute_bathy",
    "compute_block_corner",
    "map_points_to_cells",
    "__version__",
]
