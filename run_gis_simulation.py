"""
run_gis_simulation.py

Headless pipeline runner — connects the existing drone swarm simulation
(src/) to the GIS layer (gis/) and produces all outputs.

Runs without a GUI. For interactive use, run app.py instead.

Usage:
    python run_gis_simulation.py
    python run_gis_simulation.py --drones 20 --pattern formation_flying --duration 30
"""

import argparse
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.drone_trajectory_generator import (
    DroneSwarmGenerator, SimulationBounds, FlightPattern
)
from gis.coordinate_transformer import CoordinateTransformer, SIM_WIDTH_M, SIM_HEIGHT_M
from gis.sensor_heatmap import SensorHeatmap
from gis.coverage_analyzer import CoverageAnalyzer
from gis.folium_map import FoliumMap


# Keep original 1km scale — coordinate transformer scales to full BCC area
BCC_SIM_BOUNDS = SimulationBounds(
    x_min = 0.0,
    x_max = 1000.0,
    y_min = 0.0,
    y_max = 1000.0,
    z_min = 50.0,
    z_max = 500.0
)

PATTERN_MAP = {
    "coordinated_attack": FlightPattern.COORDINATED_ATTACK,
    "random_dispersal":   FlightPattern.RANDOM_DISPERSAL,
    "formation_flying":   FlightPattern.FORMATION_FLYING,
    "evasive_maneuvers":  FlightPattern.EVASIVE_MANEUVERS,
    "perimeter_sweep":    FlightPattern.PERIMETER_SWEEP,
}


def extract_positions(swarm_result: dict) -> list[dict]:
    """
    Convert swarm simulation output to GIS position format.

    Scales x/y from 1km simulation space to full BCC canyon area
    so drone paths spread across the entire simulation region.
    """
    trajectories = swarm_result["trajectories"]  # [drone, time, 3]
    n_drones, n_steps, _ = trajectories.shape

    # Scale 0-1000m sim space to full BCC area
    x_scale = SIM_WIDTH_M  / 1000.0   # ~19.32x
    y_scale = SIM_HEIGHT_M / 1000.0   # ~13.32x

    confidences = swarm_result.get("detection_confidences", None)

    positions = []
    for drone_id in range(n_drones):
        for step in range(n_steps):
            x, y, z = trajectories[drone_id, step]

            if confidences is not None and drone_id < len(confidences):
                conf = float(confidences[drone_id])
            else:
                conf = 0.65

            positions.append({
                "drone_id":   drone_id,
                "timestep":   step,
                "x":          float(x) * x_scale,
                "y":          float(y) * y_scale,
                "z":          float(z),
                "confidence": conf,
            })

    return positions


def run_simulation(
    n_drones:      int   = 10,
    pattern:       str   = "perimeter_sweep",
    duration:      float = 60.0,
    timestep:      float = 0.5,
    drone_type:    str   = "small",
    output_prefix: str   = "bcc",
) -> dict:
    """
    Run the full drone swarm GIS simulation pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix    = f"{output_prefix}_{n_drones}drones_{pattern}_{timestamp}"

    print("=" * 55)
    print("Drone Swarm GIS Simulation — Big Cottonwood Canyon")
    print("=" * 55)
    print(f"Drones:    {n_drones}")
    print(f"Pattern:   {pattern}")
    print(f"Duration:  {duration}s")
    print(f"Timestep:  {timestep}s")
    print(f"Drone:     {drone_type}")
    print("=" * 55)

    # ── Step 1: Run swarm simulation ──────────────────────────
    print("\n[1/5] Running swarm simulation...")
    generator      = DroneSwarmGenerator(BCC_SIM_BOUNDS)
    flight_pattern = PATTERN_MAP.get(pattern, FlightPattern.PERIMETER_SWEEP)

    swarm_result = generator.generate_swarm_trajectories(
        num_drones  = n_drones,
        pattern     = flight_pattern,
        duration    = duration,
        timestep    = timestep,
        drone_type  = drone_type,
    )

    positions = extract_positions(swarm_result)
    print(f"    Generated {len(positions):,} position readings "
          f"from {n_drones} drones")

    # ── Step 2: Transform coordinates ────────────────────────
    print("\n[2/5] Loading terrain and transforming coordinates...")
    transformer = CoordinateTransformer()

    # ── Step 3: Build sensor heatmap ─────────────────────────
    print("\n[3/5] Building sensor heatmap...")
    heatmap = SensorHeatmap(transformer=transformer)
    heatmap.add_batch_readings(positions)
    raster_paths = heatmap.save_all_rasters(prefix=prefix)

    # ── Step 4: Analyze coverage ──────────────────────────────
    print("\n[4/5] Analyzing coverage...")
    analyzer = CoverageAnalyzer(transformer=transformer, heatmap=heatmap)
    analyzer.print_report()
    analyzer.export_coverage_gaps(filename=f"{prefix}_gaps.geojson")
    analyzer.export_flight_paths(positions, filename=f"{prefix}_paths.geojson")

    # ── Step 5: Build interactive map ────────────────────────
    print("\n[5/5] Building interactive map...")
    fm       = FoliumMap(transformer=transformer, heatmap=heatmap, analyzer=analyzer)
    map_path = fm.build(positions, filename=f"{prefix}_map.html")

    # ── Save coverage report ──────────────────────────────────
    report      = analyzer.generate_report()
    report_path = os.path.join(
        "results/simulation_outputs", f"{prefix}_report.json"
    )
    os.makedirs("results/simulation_outputs", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {report_path}")

    outputs = {
        "map":         map_path,
        "rasters":     raster_paths,
        "report":      report_path,
        "n_positions": len(positions),
    }

    print("\n" + "=" * 55)
    print("Simulation complete.")
    print(f"Open map: {map_path}")
    print("=" * 55)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drone Swarm GIS Simulation — Big Cottonwood Canyon"
    )
    parser.add_argument("--drones",      type=int,   default=10)
    parser.add_argument("--pattern",     type=str,   default="perimeter_sweep",
                        choices=list(PATTERN_MAP.keys()))
    parser.add_argument("--duration",    type=float, default=60.0)
    parser.add_argument("--timestep",    type=float, default=0.5)
    parser.add_argument("--drone-type",  type=str,   default="small",
                        choices=["micro", "small", "medium"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        n_drones   = args.drones,
        pattern    = args.pattern,
        duration   = args.duration,
        timestep   = args.timestep,
        drone_type = args.drone_type,
    )
