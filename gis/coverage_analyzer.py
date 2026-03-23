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
import json
import os
from typing import Optional

from gis.coordinate_transformer import CoordinateTransformer, BCC_BOUNDS
from gis.sensor_heatmap import SensorHeatmap


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
        output_dir: str = "results/simulation_outputs"
    ):
        self.transformer = transformer
        self.heatmap     = heatmap
        self.output_dir  = output_dir
        self.geojson_dir = os.path.join(output_dir, "geojson")
        self.dem         = transformer.get_dem_array()

        os.makedirs(self.geojson_dir, exist_ok=True)

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
        GeoDataFrame of gap polygons in WGS84
        """
        coverage = self.heatmap.get_coverage_raster()
        gap_mask = coverage == 0

        resolution = self.heatmap.resolution
        bounds     = self.heatmap.bounds
        n_rows, n_cols = coverage.shape

        gap_polygons = []
        for row in range(n_rows):
            for col in range(n_cols):
                if gap_mask[row, col]:
                    west  = bounds["west"]  + col * resolution
                    east  = west + resolution
                    north = bounds["north"] - row * resolution
                    south = north - resolution
                    gap_polygons.append(box(west, south, east, north))

        if not gap_polygons:
            return gpd.GeoDataFrame(
                {"geometry": [], "area_km2": []}, crs="EPSG:4326"
            )

        # Merge adjacent gap cells into contiguous polygons
        merged = unary_union(gap_polygons)
        geoms  = list(merged.geoms) if merged.geom_type == "MultiPolygon" \
                 else [merged]

        # Filter small gaps
        cell_area = resolution * resolution
        min_area  = min_gap_cells * cell_area
        geoms     = [g for g in geoms if g.area >= min_area]

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geoms,
                "area_km2": [round(g.area * 111 * 84, 4) for g in geoms],
            },
            crs="EPSG:4326"
        )
        return gdf

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
