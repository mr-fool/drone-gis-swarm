"""
sensor_heatmap.py

Converts drone swarm sensor readings into georeferenced raster layers
(GeoTIFF) over the Big Cottonwood Canyon simulation area.

Takes drone position + detection confidence data from the swarm simulation
and produces:
  - Detection confidence heatmap (GeoTIFF)
  - Coverage density map (GeoTIFF)
  - Altitude above ground layer (GeoTIFF)

These rasters can be loaded directly into QGIS or ArcGIS.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os
from typing import Optional
from gis.coordinate_transformer import CoordinateTransformer, BCC_BOUNDS


# Output raster resolution in degrees (~100m cells at this latitude)
RASTER_RESOLUTION_DEG = 0.001


class SensorHeatmap:
    """
    Builds georeferenced raster layers from drone sensor readings.

    Each raster cell aggregates readings from all drone passes over
    that geographic area during the simulation.
    """

    def __init__(
        self,
        transformer: CoordinateTransformer,
        resolution_deg: float = RASTER_RESOLUTION_DEG,
        output_dir: str = "results/simulation_outputs/rasters"
    ):
        self.transformer  = transformer
        self.resolution   = resolution_deg
        self.output_dir   = output_dir
        self.bounds       = BCC_BOUNDS

        os.makedirs(self.output_dir, exist_ok=True)

        # Calculate raster dimensions
        self.n_cols = int(
            (self.bounds["east"] - self.bounds["west"]) / self.resolution
        )
        self.n_rows = int(
            (self.bounds["north"] - self.bounds["south"]) / self.resolution
        )

        # Affine transform: maps pixel → geographic coordinates
        self.transform = from_bounds(
            west   = self.bounds["west"],
            south  = self.bounds["south"],
            east   = self.bounds["east"],
            north  = self.bounds["north"],
            width  = self.n_cols,
            height = self.n_rows
        )

        # Initialize accumulator grids
        self._confidence_sum   = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self._confidence_count = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        self._altitude_sum     = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self._visit_count      = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)

        print(f"Heatmap grid: {self.n_cols} cols × {self.n_rows} rows")
        print(f"Cell size:    ~{self.resolution * 111000:.0f}m × "
              f"{self.resolution * 84000:.0f}m")

    def _latlon_to_pixel(
        self, lat: float, lon: float
    ) -> tuple[int, int]:
        """Convert lat/lon to raster pixel (row, col). Returns (-1,-1) if out of bounds."""
        col = int((lon - self.bounds["west"])  / self.resolution)
        row = int((self.bounds["north"] - lat) / self.resolution)
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return row, col
        return -1, -1

    def add_drone_reading(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        confidence: float
    ):
        """
        Record a single drone sensor reading at simulation position (x, y, z).

        Parameters
        ----------
        x_m        : eastward position in meters from SW corner
        y_m        : northward position in meters from SW corner
        z_m        : altitude above ground in meters
        confidence : detection confidence from swarm simulation (0.0 – 1.0)
        """
        lat, lon = self.transformer.sim_to_latlon(x_m, y_m)
        row, col = self._latlon_to_pixel(lat, lon)

        if row == -1:
            return

        self._confidence_sum[row, col]   += confidence
        self._confidence_count[row, col] += 1
        self._altitude_sum[row, col]     += z_m
        self._visit_count[row, col]      += 1

    def add_batch_readings(self, positions: list[dict]):
        """
        Add multiple drone readings at once.

        Each dict should have: x, y, z, confidence
        Convenience wrapper around add_drone_reading for bulk processing.
        """
        for pos in positions:
            self.add_drone_reading(
                x_m        = pos["x"],
                y_m        = pos["y"],
                z_m        = pos.get("z", 0.0),
                confidence = pos.get("confidence", 0.0)
            )

    def get_confidence_raster(self) -> np.ndarray:
        """
        Return mean detection confidence per cell.
        Cells with no readings return 0.0.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(
                self._confidence_count > 0,
                self._confidence_sum / self._confidence_count,
                0.0
            )
        return result.astype(np.float32)

    def get_coverage_raster(self) -> np.ndarray:
        """
        Return visit count per cell (how many drone passes covered each cell).
        Useful for identifying coverage gaps.
        """
        return self._visit_count.astype(np.float32)

    def get_altitude_raster(self) -> np.ndarray:
        """
        Return mean drone altitude above ground per cell.
        Cells with no readings return 0.0.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(
                self._visit_count > 0,
                self._altitude_sum / self._visit_count,
                0.0
            )
        return result.astype(np.float32)

    def _save_raster(
        self,
        data: np.ndarray,
        filename: str,
        nodata: float = 0.0
    ) -> str:
        """Save a numpy array as a georeferenced GeoTIFF."""
        filepath = os.path.join(self.output_dir, filename)
        with rasterio.open(
            filepath,
            mode      = "w",
            driver    = "GTiff",
            height    = self.n_rows,
            width     = self.n_cols,
            count     = 1,
            dtype     = data.dtype,
            crs       = CRS.from_epsg(4326),
            transform = self.transform,
            nodata    = nodata
        ) as dst:
            dst.write(data, 1)
        return filepath

    def save_all_rasters(self, prefix: str = "bcc") -> dict[str, str]:
        """
        Save all raster layers to GeoTIFF files.

        Parameters
        ----------
        prefix : filename prefix (default 'bcc' for Big Cottonwood Canyon)

        Returns
        -------
        dict mapping layer name → filepath
        """
        saved = {}

        confidence = self.get_confidence_raster()
        path = self._save_raster(confidence, f"{prefix}_confidence.tif")
        saved["confidence"] = path
        print(f"Saved confidence raster → {path}")

        coverage = self.get_coverage_raster()
        path = self._save_raster(coverage, f"{prefix}_coverage.tif")
        saved["coverage"] = path
        print(f"Saved coverage raster   → {path}")

        altitude = self.get_altitude_raster()
        path = self._save_raster(altitude, f"{prefix}_altitude.tif")
        saved["altitude"] = path
        print(f"Saved altitude raster   → {path}")

        return saved

    def get_coverage_stats(self) -> dict:
        """
        Return summary statistics about simulation coverage.
        Useful for the paper's results section.
        """
        total_cells    = self.n_rows * self.n_cols
        covered_cells  = int(np.sum(self._visit_count > 0))
        coverage_pct   = (covered_cells / total_cells) * 100

        confidence     = self.get_confidence_raster()
        covered_mask   = self._visit_count > 0

        return {
            "total_cells":        total_cells,
            "covered_cells":      covered_cells,
            "coverage_pct":       round(coverage_pct, 2),
            "mean_confidence":    round(float(confidence[covered_mask].mean()), 4)
                                  if covered_cells > 0 else 0.0,
            "max_confidence":     round(float(confidence.max()), 4),
            "mean_visits":        round(float(self._visit_count[covered_mask].mean()), 2)
                                  if covered_cells > 0 else 0.0,
            "gap_cells":          total_cells - covered_cells,
            "gap_pct":            round(100 - coverage_pct, 2),
        }

    def reset(self):
        """Clear all accumulated readings."""
        self._confidence_sum[:]   = 0
        self._confidence_count[:] = 0
        self._altitude_sum[:]     = 0
        self._visit_count[:]      = 0


if __name__ == "__main__":
    import random

    print("Testing SensorHeatmap with synthetic drone readings...")

    ct = CoordinateTransformer()
    hm = SensorHeatmap(transformer=ct)

    # Simulate 500 random drone readings across the area
    random.seed(42)
    from gis.coordinate_transformer import SIM_WIDTH_M, SIM_HEIGHT_M

    synthetic_readings = [
        {
            "x":          random.uniform(0, SIM_WIDTH_M),
            "y":          random.uniform(0, SIM_HEIGHT_M),
            "z":          random.uniform(50, 300),
            "confidence": random.uniform(0.3, 0.9),
        }
        for _ in range(500)
    ]

    hm.add_batch_readings(synthetic_readings)

    stats = hm.get_coverage_stats()
    print("\nCoverage statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    saved = hm.save_all_rasters(prefix="test")
    print(f"\nRasters saved: {list(saved.keys())}")
    print("Open any .tif in QGIS to verify output.")
