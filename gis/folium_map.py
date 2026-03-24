"""
folium_map.py

Generates an interactive HTML web map combining all simulation outputs:
  - USGS terrain elevation as a heatmap base layer
  - Drone flight paths as colored polylines
  - Coverage confidence heatmap overlay
  - Coverage gap polygons
  - Clickable drone position markers with metadata

Output is a single self-contained HTML file that opens in any browser.
Can be shared, embedded in a portfolio, or included as supplementary
material in the paper.
"""

import folium
from folium.plugins import HeatMap, MeasureControl, Fullscreen
import geopandas as gpd
import numpy as np
import os
import json
from datetime import datetime

from gis.coordinate_transformer import CoordinateTransformer, BCC_BOUNDS
from gis.sensor_heatmap import SensorHeatmap
from gis.coverage_analyzer import CoverageAnalyzer


# Color palette for drone flight paths
DRONE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]


class FoliumMap:
    """
    Builds an interactive Folium web map from simulation outputs.

    Layers (toggleable in the browser):
      1. OpenStreetMap base
      2. Terrain elevation heatmap
      3. Drone flight paths (one color per drone)
      4. Detection confidence heatmap
      5. Coverage gap polygons
    """

    def __init__(
        self,
        transformer: CoordinateTransformer,
        heatmap: SensorHeatmap,
        analyzer: CoverageAnalyzer,
        output_dir: str = "results/simulation_outputs/maps"
    ):
        self.transformer = transformer
        self.heatmap     = heatmap
        self.analyzer    = analyzer
        self.output_dir  = output_dir
        self.bounds      = BCC_BOUNDS

        os.makedirs(self.output_dir, exist_ok=True)

        # Map center
        self.center_lat = (self.bounds["south"] + self.bounds["north"]) / 2
        self.center_lon = (self.bounds["west"]  + self.bounds["east"])  / 2

    def _build_base_map(self) -> folium.Map:
        """Initialize the Folium map with base layers and controls."""
        m = folium.Map(
            location      = [self.center_lat, self.center_lon],
            zoom_start    = 11,
            tiles         = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr          = "Esri World Imagery",
            control_scale = True,
        )

        # Add controls
        MeasureControl(position="topleft").add_to(m)
        Fullscreen(position="topright").add_to(m)

        # Simulation bounding box outline
        folium.Rectangle(
            bounds = [
                [self.bounds["south"], self.bounds["west"]],
                [self.bounds["north"], self.bounds["east"]],
            ],
            color     = "#e74c3c",
            weight    = 2,
            fill      = False,
            tooltip   = "Simulation area — Big Cottonwood Canyon",
        ).add_to(m)

        return m

    def _add_elevation_heatmap(self, m: folium.Map):
        """Add terrain elevation as a heatmap layer."""
        dem    = self.transformer.get_dem_array()
        bounds = self.bounds
        h, w   = dem.shape

        # Sample DEM at regular intervals for heatmap points
        step = max(1, h // 50)
        points = []
        for row in range(0, h, step):
            for col in range(0, w, step):
                elev = float(dem[row, col])
                if elev < -9000:
                    continue
                lat = bounds["north"] - (row / h) * (
                    bounds["north"] - bounds["south"]
                )
                lon = bounds["west"]  + (col / w) * (
                    bounds["east"]  - bounds["west"]
                )
                # Normalize elevation to 0-1 for heatmap intensity
                norm = (elev - 1375) / (3500 - 1375)
                points.append([lat, lon, float(np.clip(norm, 0, 1))])

        HeatMap(
            points,
            name       = "Terrain elevation",
            min_opacity = 0.2,
            max_zoom   = 18,
            radius     = 15,
            blur       = 10,
            gradient   = {
                "0.0": "#313695",
                "0.3": "#74add1",
                "0.5": "#fee090",
                "0.7": "#f46d43",
                "1.0": "#a50026",
            },
        ).add_to(m)

    def _add_confidence_heatmap(self, m: folium.Map):
        """Add detection confidence as a heatmap layer."""
        confidence = self.heatmap.get_confidence_raster()
        bounds     = self.bounds
        h, w       = confidence.shape

        points = []
        for row in range(h):
            for col in range(w):
                val = float(confidence[row, col])
                if val <= 0:
                    continue
                lat = bounds["north"] - (row / h) * (
                    bounds["north"] - bounds["south"]
                )
                lon = bounds["west"]  + (col / w) * (
                    bounds["east"]  - bounds["west"]
                )
                points.append([lat, lon, val])

        if points:
            HeatMap(
                points,
                name        = "Detection confidence",
                min_opacity = 0.3,
                radius      = 20,
                blur        = 15,
                gradient    = {
                    "0.0": "#ffffcc",
                    "0.5": "#41b6c4",
                    "1.0": "#0c2c84",
                },
            ).add_to(m)

    def _add_flight_paths(
        self, m: folium.Map, positions: list[dict]
    ):
        """Add drone flight paths as colored polylines."""
        if not positions:
            return

        gdf = self.transformer.sim_positions_to_geodataframe(positions)

        drone_ids = sorted(gdf["drone_id"].unique()) \
                    if "drone_id" in gdf.columns else [0]

        path_group = folium.FeatureGroup(name="Drone flight paths")

        for i, drone_id in enumerate(drone_ids):
            color = DRONE_COLORS[i % len(DRONE_COLORS)]

            if "drone_id" in gdf.columns:
                drone_df = gdf[gdf["drone_id"] == drone_id]
                if "timestep" in drone_df.columns:
                    drone_df = drone_df.sort_values("timestep")
            else:
                drone_df = gdf

            if len(drone_df) < 2:
                continue

            coords = [[row["lat"], row["lon"]]
                      for _, row in drone_df.iterrows()]

            folium.PolyLine(
                locations = coords,
                color     = color,
                weight    = 2,
                opacity   = 0.8,
                tooltip   = f"Drone {drone_id} — {len(coords)} waypoints",
            ).add_to(path_group)

            # Mark start and end
            folium.CircleMarker(
                location = coords[0],
                radius   = 5,
                color    = color,
                fill     = True,
                tooltip  = f"Drone {drone_id} start",
            ).add_to(path_group)

        path_group.add_to(m)

    def _add_coverage_gaps(self, m: folium.Map):
        """Add coverage gap polygons as a layer."""
        gaps = self.analyzer.get_coverage_gaps()
        if len(gaps) == 0:
            return

        gap_group = folium.FeatureGroup(name="Coverage gaps")

        for _, row in gaps.iterrows():
            geom = row["geometry"]
            area = row["area_km2"]

            # Convert shapely polygon to folium coordinates
            if geom.geom_type == "Polygon":
                coords = [[y, x] for x, y in geom.exterior.coords]
                folium.Polygon(
                    locations  = coords,
                    color      = "#e74c3c",
                    weight     = 1,
                    fill       = True,
                    fill_color = "#e74c3c",
                    fill_opacity = 0.3,
                    tooltip    = f"Coverage gap: {area} km²",
                ).add_to(gap_group)

        gap_group.add_to(m)

    def _add_summary_panel(self, m: folium.Map):
        """Add a summary statistics panel to the map."""
        stats  = self.heatmap.get_coverage_stats()
        report = self.analyzer.generate_report()
        now    = datetime.now().strftime("%Y-%m-%d %H:%M")

        elev_rows = ""
        for band, s in report["elevation_band_coverage"].items():
            elev_rows += (
                f"<tr><td>{band}</td>"
                f"<td>{s['elevation_range_m']}m</td>"
                f"<td>{s['coverage_pct']}%</td>"
                f"<td>{s['mean_confidence']}</td></tr>"
            )

        html = f"""
        <div style="
            position: fixed;
            bottom: 30px; left: 30px;
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            max-width: 320px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        ">
            <b>Drone Swarm GIS Simulation</b><br>
            Big Cottonwood Canyon, Utah<br>
            <small style="color:#666">{now}</small>
            <hr style="margin:6px 0">
            <b>Overall Coverage</b><br>
            Coverage: {stats['coverage_pct']}% &nbsp;|&nbsp;
            Confidence: {stats['mean_confidence']}<br>
            Gap zones: {report['coverage_gaps']['n_gap_zones']} &nbsp;|&nbsp;
            Gap area: {report['coverage_gaps']['total_gap_km2']} km²
            <hr style="margin:6px 0">
            <b>By Elevation Band</b><br>
            <table style="width:100%;border-collapse:collapse;font-size:11px">
                <tr style="background:#f5f5f5">
                    <th>Zone</th><th>Elevation</th>
                    <th>Coverage</th><th>Confidence</th>
                </tr>
                {elev_rows}
            </table>
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))

    def build(
        self,
        positions: list[dict],
        filename: str = "bcc_simulation_map.html"
    ) -> str:
        """
        Build and save the complete interactive map.

        Parameters
        ----------
        positions : list of drone position dicts
        filename  : output HTML filename

        Returns
        -------
        filepath of saved HTML map
        """
        print("Building interactive map...")

        m = self._build_base_map()

        print("  Adding terrain elevation layer...")
        self._add_elevation_heatmap(m)

        print("  Adding detection confidence layer...")
        self._add_confidence_heatmap(m)

        print("  Adding flight paths...")
        self._add_flight_paths(m, positions)

        print("  Adding coverage gaps...")
        self._add_coverage_gaps(m)

        print("  Adding summary panel...")
        self._add_summary_panel(m)

        # Layer control (toggle layers on/off)
        folium.LayerControl(position="topright", collapsed=False).add_to(m)

        filepath = os.path.join(self.output_dir, filename)
        m.save(filepath)
        print(f"\nMap saved → {filepath}")
        print(f"Open in any browser to view.")
        return filepath


if __name__ == "__main__":
    import random
    from gis.coordinate_transformer import SIM_WIDTH_M, SIM_HEIGHT_M

    print("Building interactive map with synthetic drone data...")

    ct = CoordinateTransformer()
    hm = SensorHeatmap(transformer=ct)
    random.seed(42)

    # Simulate 10 drones sweeping across the canyon
    positions = []
    for drone_id in range(10):
        y_start = random.uniform(0, SIM_HEIGHT_M)
        for step in range(50):
            positions.append({
                "drone_id":   drone_id,
                "timestep":   step,
                "x":          (step / 50) * SIM_WIDTH_M,
                "y":          y_start + random.uniform(-500, 500),
                "z":          random.uniform(100, 400),
                "confidence": random.uniform(0.4, 0.9),
            })

    hm.add_batch_readings(positions)

    analyzer = CoverageAnalyzer(transformer=ct, heatmap=hm)
    fm       = FoliumMap(transformer=ct, heatmap=hm, analyzer=analyzer)
    filepath = fm.build(positions)

    print(f"\nDone. Open this file in your browser:")
    print(f"  {filepath}")
