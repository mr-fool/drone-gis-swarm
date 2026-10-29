"""
coverage_analyzer.py

Analyzes drone swarm coverage gaps and terrain mapping completeness
over the Big Cottonwood Canyon simulation area.

Takes output from SensorHeatmap and produces:
  - Coverage gap polygons (GeoJSON) — areas with no drone coverage
  - Coverage statistics report (dict) — for paper results section
  - Terrain coverage breakdown by elevation band — high/mid/low altitude zones
  - GeoJSON flight path output — drone trajectories as LineStrings

These outputs can be loaded directly into QGIS or ArcGIS for visualization.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union
from shapely.affinity import affine_transform
from scipy import ndimage
import json
import os
from typing import Optional

from gis.coordinate_transformer import CoordinateTransformer, BCC_BOUNDS
from gis.sensor_heatmap import SensorHeatmap
from gis.footprint_coverage import FootprintCoverage

# A coarse (0.001-degree) cell counts as covered when at least this fraction
# of its area was inside a camera footprint.
COARSE_COVERED_FRACTION = 0.5


# Elevation bands for terrain coverage analysis (meters ASL)
ELEVATION_BANDS = {
    "valley":    (1375, 1800),   # Canyon floor
    "mid_slope": (1800, 2600),   # Mid elevation
    "alpine":    (2600, 3500),   # High alpine zone
}


class CoverageAnalyzer:
    """
    Analyzes spatial coverage quality from drone swarm simulation.

    Identifies gaps, computes statistics by terrain zone, and exports
    results as GeoJSON for use in GIS platforms and the paper.
    """

    def __init__(
        self,
        transformer: CoordinateTransformer,
        heatmap: SensorHeatmap,
        output_dir: str = "results/simulation_outputs",
        footprint: Optional[FootprintCoverage] = None,
    ):
        """
        footprint : if given, all coverage statistics are area-based (camera
                    footprint along the flight path). If None, the legacy
                    point-sample metric is used (a cell is covered if any
                    position sample falls in it) -- kept only for comparison.
        """
        self.transformer = transformer
        self.heatmap     = heatmap
        self.footprint   = footprint
        self.output_dir  = output_dir
        self.geojson_dir = os.path.join(output_dir, "geojson")
        self.dem         = transformer.get_dem_array()

        os.makedirs(self.geojson_dir, exist_ok=True)

    def count_gap_zones(self, min_gap_cells: int = 4) -> int:
        """Number of gap zones (same definition as get_coverage_gaps,
        without building polygons)."""
        labels, n = ndimage.label(self._coverage_raster() == 0)
        if n == 0:
            return 0
        return int((np.bincount(labels.ravel())[1:] >= min_gap_cells).sum())

    def get_coverage_gaps(
        self, min_gap_cells: int = 4
    ) -> gpd.GeoDataFrame:
        """
        Identify geographic areas with zero drone coverage.

        Parameters
        ----------
        min_gap_cells : minimum contiguous uncovered cells to report
                        (filters out single-cell noise)

        Returns
        -------
        GeoDataFrame of gap polygons in WGS84, one row per gap zone

        Gap zones are the 4-connected components of uncovered grid cells,
        found on the raster itself. (The previous version unioned per-cell
        boxes whose edges were computed in floating point; adjacent boxes
        did not always share exact edges, so an entirely empty map came out
        as 40 horizontal strips instead of one gap.)
        """
        coverage = self._coverage_raster()
        gap_mask = coverage == 0

        resolution = self.heatmap.resolution
        bounds     = self.heatmap.bounds

        labels, n_labels = ndimage.label(gap_mask)   # 4-connectivity
        if n_labels == 0:
            return gpd.GeoDataFrame(
                {"geometry": [], "area_km2": [], "n_cells": []}, crs="EPSG:4326"
            )
        sizes = np.bincount(labels.ravel())

        # Cell -> lon/lat: x = west + col*res, y = north - row*res
        to_lonlat = [resolution, 0.0, 0.0, -resolution,
                     bounds["west"], bounds["north"]]
        cell_km2  = resolution * resolution * 111 * 84

        geoms, areas, n_cells = [], [], []
        for lab in range(1, n_labels + 1):
            if sizes[lab] < min_gap_cells:
                continue
            rows, cols = np.nonzero(labels == lab)
            # Build the polygon on the integer grid, where shared edges are
            # exact, then transform to geographic coordinates.
            grid_poly = unary_union([box(c, r, c + 1, r + 1)
                                     for r, c in zip(rows, cols)])
            geoms.append(affine_transform(grid_poly, to_lonlat))
            areas.append(round(float(sizes[lab]) * cell_km2, 4))
            n_cells.append(int(sizes[lab]))

        gdf = gpd.GeoDataFrame(
            {"geometry": geoms, "area_km2": areas, "n_cells": n_cells},
            crs="EPSG:4326"
        )
        return gdf

    def get_coverage_fraction_raster(self) -> np.ndarray:
        """Fraction of each coarse cell imaged (footprint metric only)."""
        hm = self.heatmap
        return self.footprint.coarse_fraction(hm.n_rows, hm.n_cols, hm.bounds, hm.resolution)

    def _coverage_raster(self) -> np.ndarray:
        """Coarse-grid raster that is > 0 where a cell counts as covered."""
        if self.footprint is None:
            return self.heatmap.get_coverage_raster()
        frac = self.get_coverage_fraction_raster()
        return (frac >= COARSE_COVERED_FRACTION).astype(np.float32)

    def get_elevation_band_coverage(self) -> dict:
        """
        Break down coverage by terrain elevation band.

        Compares drone coverage raster against DEM elevation zones
        to identify which terrain types are well covered vs missed.

        Returns
        -------
        dict with coverage stats per elevation band
        """
        coverage  = self.heatmap.get_coverage_raster()
        dem       = self.dem
        bounds    = self.heatmap.bounds
        h_heat, w_heat = coverage.shape
        h_dem,  w_dem  = dem.shape

        results = {}
        for band_name, (elev_min, elev_max) in ELEVATION_BANDS.items():
            # Scale DEM to heatmap resolution
            row_scale = h_dem / h_heat
            col_scale = w_dem / w_heat

            band_cells    = 0
            covered_cells = 0
            confidence_vals = []

            confidence = self.heatmap.get_confidence_raster()

            for row in range(h_heat):
                for col in range(w_heat):
                    dem_row = min(int(row * row_scale), h_dem - 1)
                    dem_col = min(int(col * col_scale), w_dem - 1)
                    elev    = dem[dem_row, dem_col]

                    if elev_min <= elev < elev_max:
                        band_cells += 1
                        if coverage[row, col] > 0:
                            covered_cells += 1
                            confidence_vals.append(confidence[row, col])

            coverage_pct = (covered_cells / band_cells * 100) \
                           if band_cells > 0 else 0.0
            mean_conf    = float(np.mean(confidence_vals)) \
                           if confidence_vals else 0.0

            results[band_name] = {
                "elevation_range_m": f"{elev_min}–{elev_max}",
                "total_cells":       band_cells,
                "covered_cells":     covered_cells,
                "coverage_pct":      round(coverage_pct, 2),
                "mean_confidence":   round(mean_conf, 4),
            }

        if self.footprint is not None:
            fp = self.footprint.band_coverage(ELEVATION_BANDS)
            for band_name, r in results.items():
                r["legacy_point_sample_coverage_pct"] = r.pop("coverage_pct")
                r["legacy_total_cells"]   = r.pop("total_cells")
                r["legacy_covered_cells"] = r.pop("covered_cells")
                r["total_pixels"]  = fp[band_name]["total_pixels"]
                r["imaged_pixels"] = fp[band_name]["imaged_pixels"]
                r["coverage_pct"]  = fp[band_name]["coverage_pct"]

        return results

    def export_flight_paths(
        self,
        positions: list[dict],
        filename: str = "bcc_flight_paths.geojson"
    ) -> str:
        """
        Export drone flight paths as GeoJSON LineStrings.

        Groups positions by drone_id and creates one LineString
        per drone showing its complete trajectory.

        Parameters
        ----------
        positions : list of dicts with x, y, z, drone_id, timestep
        filename  : output filename

        Returns
        -------
        filepath of saved GeoJSON
        """
        gdf = self.transformer.sim_positions_to_geodataframe(positions)

        # Group by drone_id and build trajectories
        trajectories = []
        drone_ids    = gdf["drone_id"].unique() \
                       if "drone_id" in gdf.columns else [0]

        for drone_id in drone_ids:
            if "drone_id" in gdf.columns:
                drone_df = gdf[gdf["drone_id"] == drone_id].sort_values(
                    "timestep" if "timestep" in gdf.columns else gdf.index.name
                )
            else:
                drone_df = gdf

            if len(drone_df) < 2:
                continue

            coords = list(zip(drone_df["lon"], drone_df["lat"]))
            trajectories.append({
                "drone_id":    int(drone_id),
                "n_positions": len(drone_df),
                "geometry":    LineString(coords),
            })

        if not trajectories:
            print("No trajectories to export.")
            return ""

        traj_gdf  = gpd.GeoDataFrame(trajectories, crs="EPSG:4326")
        filepath  = os.path.join(self.geojson_dir, filename)
        traj_gdf.to_file(filepath, driver="GeoJSON")
        print(f"Saved flight paths → {filepath}")
        return filepath

    def export_coverage_gaps(
        self, filename: str = "bcc_coverage_gaps.geojson"
    ) -> str:
        """Export coverage gap polygons as GeoJSON."""
        gaps     = self.get_coverage_gaps()
        filepath = os.path.join(self.geojson_dir, filename)
        if len(gaps) > 0:
            gaps.to_file(filepath, driver="GeoJSON")
            print(f"Saved coverage gaps → {filepath} ({len(gaps)} gap zones)")
        else:
            print("No coverage gaps found.")
        return filepath

    def generate_report(self) -> dict:
        """
        Generate a complete coverage report for the paper results section.

        Returns
        -------
        dict with all coverage statistics
        """
        basic_stats = self.heatmap.get_coverage_stats()
        if self.footprint is not None:
            legacy = basic_stats
            covered = self._coverage_raster() > 0
            total   = int(covered.size)
            basic_stats = {
                **self.footprint.summary(),
                "total_cells":   total,
                "covered_cells": int(covered.sum()),
                "covered_cell_definition": f">= {COARSE_COVERED_FRACTION:.0%} of 0.001-degree cell imaged",
                "gap_cells":     total - int(covered.sum()),
                "gap_pct":       round(100 - self.footprint.coverage_pct(), 2),
                "mean_confidence": legacy["mean_confidence"],
                "max_confidence":  legacy["max_confidence"],
                "legacy_point_sample_coverage_pct": legacy["coverage_pct"],
                "legacy_point_sample_covered_cells": legacy["covered_cells"],
            }
        elev_stats  = self.get_elevation_band_coverage()
        gaps        = self.get_coverage_gaps()

        report = {
            "simulation_area": {
                "bounds":       BCC_BOUNDS,
                "location":     "Big Cottonwood Canyon, Wasatch Range, Utah",
                "width_m":      self.transformer.sim_width_m,
                "height_m":     self.transformer.sim_height_m,
            },
            "overall_coverage":     basic_stats,
            "elevation_band_coverage": elev_stats,
            "coverage_gaps": {
                "n_gap_zones":  len(gaps),
                "total_gap_km2": round(float(gaps["area_km2"].sum()), 4)
                                 if len(gaps) > 0 else 0.0,
            },
        }
        return report

    def print_report(self):
        """Print a formatted coverage report to console."""
        report = self.generate_report()

        print("\n" + "=" * 55)
        print("Coverage Analysis Report — Big Cottonwood Canyon")
        print("=" * 55)

        oc = report["overall_coverage"]
        print(f"\nOverall Coverage:")
        print(f"  Total cells:      {oc['total_cells']:,}")
        print(f"  Covered cells:    {oc['covered_cells']:,}")
        print(f"  Coverage:         {oc['coverage_pct']}%")
        if "imaged_km2" in oc:
            print(f"  Imaged area:      {oc['imaged_km2']} of {oc['total_km2']} km² "
                  f"(footprint, FOV {oc['fov_deg']}°, {oc['altitude_mode']})")
            print(f"  Legacy point-sample coverage: {oc['legacy_point_sample_coverage_pct']}%")
        print(f"  Mean confidence:  {oc['mean_confidence']}")
        print(f"  Gap cells:        {oc['gap_cells']:,} ({oc['gap_pct']}%)")

        print(f"\nCoverage by Elevation Band:")
        for band, stats in report["elevation_band_coverage"].items():
            print(f"  {band:12} ({stats['elevation_range_m']}m): "
                  f"{stats['coverage_pct']}% covered, "
                  f"confidence={stats['mean_confidence']}")

        gaps = report["coverage_gaps"]
        print(f"\nCoverage Gaps:")
        print(f"  Gap zones:        {gaps['n_gap_zones']:,}")
        print(f"  Total gap area:   {gaps['total_gap_km2']} km²")
        print("=" * 55)


if __name__ == "__main__":
    import random
    from gis.coordinate_transformer import SIM_WIDTH_M, SIM_HEIGHT_M

    print("Testing CoverageAnalyzer with synthetic drone data...")

    ct = CoordinateTransformer()
    hm = SensorHeatmap(transformer=ct)

    # Simulate a more realistic swarm pattern —
    # 10 drones each making a sweep across the canyon
    random.seed(42)
    positions = []
    for drone_id in range(10):
        y_start = random.uniform(0, SIM_HEIGHT_M)
        for step in range(50):
            positions.append({
                "drone_id":  drone_id,
                "timestep":  step,
                "x":         (step / 50) * SIM_WIDTH_M,
                "y":         y_start + random.uniform(-500, 500),
                "z":         random.uniform(100, 400),
                "confidence": random.uniform(0.4, 0.9),
            })

    hm.add_batch_readings(positions)

    analyzer = CoverageAnalyzer(transformer=ct, heatmap=hm)
    analyzer.print_report()
    analyzer.export_coverage_gaps()
    analyzer.export_flight_paths(positions)
    print("\nOpen geojson outputs in QGIS to verify.")
