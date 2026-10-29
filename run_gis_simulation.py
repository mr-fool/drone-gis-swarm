"""
run_gis_simulation.py

Headless pipeline runner — connects the existing drone swarm simulation
(src/) to the GIS layer (gis/) and produces all outputs.

Runs without a GUI. For interactive use, run app.py instead.

Usage:
    python run_gis_simulation.py
    python run_gis_simulation.py --drones 20 --pattern formation_flying --duration 1800 --seed 42

Simulation runs in REAL meters over the Big Cottonwood Canyon bounding box
(~19.3 km x 13.3 km) at the drones' physical speeds, and coverage is the area
imaged by each drone's nadir camera footprint along its path (see
gis/footprint_coverage.py). Earlier versions simulated a 1 km box and scaled
positions by ~19x afterwards, which made drones fly at 150-330 m/s, and
counted a cell as covered if any position sample fell in it.
"""

import argparse
import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.drone_trajectory_generator import (
    DroneSwarmGenerator, SimulationBounds, FlightPattern
)
from src.sensor_simulation import VirtualSensorArray, SensorType
from gis.coordinate_transformer import CoordinateTransformer, SIM_WIDTH_M, SIM_HEIGHT_M
from gis.sensor_heatmap import SensorHeatmap
from gis.coverage_analyzer import CoverageAnalyzer
from gis.folium_map import FoliumMap
from gis.footprint_coverage import (
    FootprintCoverage, DEFAULT_FOV_DEG, DEFAULT_MAX_MAPPING_AGL
)


# Simulate directly in real meters over the full BCC bounding box.
# z is height above ground (50–500 m) in "agl" mode, or height above the
# highest terrain in the area in "msl" mode (see gis/footprint_coverage.py).
BCC_SIM_BOUNDS = SimulationBounds(
    x_min = 0.0,
    x_max = SIM_WIDTH_M,
    y_min = 0.0,
    y_max = SIM_HEIGHT_M,
    z_min = 50.0,
    z_max = 500.0
)

# Default mission duration: 30 minutes, a typical small-UAS endurance.
DEFAULT_DURATION_S = 1800.0

PATTERN_MAP = {
    "coordinated_attack": FlightPattern.COORDINATED_ATTACK,
    "random_dispersal":   FlightPattern.RANDOM_DISPERSAL,
    "formation_flying":   FlightPattern.FORMATION_FLYING,
    "evasive_maneuvers":  FlightPattern.EVASIVE_MANEUVERS,
    "perimeter_sweep":    FlightPattern.PERIMETER_SWEEP,
}


def compute_sensor_confidences(
    trajectories: np.ndarray,
    bounds: SimulationBounds,
    timestep: float
) -> np.ndarray:
    """
    Run the sensor simulation over swarm trajectories and return
    per-drone per-timestep confidence scores.

    Uses a perimeter camera array matching the original JDMS publication
    setup — 8 visible-spectrum cameras at 600m radius, 100m elevation.

    Parameters
    ----------
    trajectories : np.ndarray shape [n_drones, n_steps, 3]
    bounds       : SimulationBounds
    timestep     : simulation timestep in seconds

    Returns
    -------
    np.ndarray shape [n_drones, n_steps] — confidence scores 0.0–1.0
    """
    print("    Setting up sensor array...")
    sensor_array = VirtualSensorArray(bounds)
    sensor_array.setup_perimeter_array(
        num_cameras = 8,
        sensor_type = SensorType.VISIBLE_SPECTRUM
    )

    n_drones, n_steps, _ = trajectories.shape
    confidences = np.zeros((n_drones, n_steps), dtype=np.float32)

    print(f"    Computing confidence for {n_drones} drones × "
          f"{n_steps} timesteps...")

    # Sample every 5th timestep for speed — interpolate the rest
    sample_interval = 5
    for step in range(0, n_steps, sample_interval):
        drone_positions = trajectories[:, step, :]  # [n_drones, 3]
        timestamp       = step * timestep

        # Serial on purpose: the threaded version draws from the global
        # random generator in thread-scheduling order, so a fixed seed did
        # not give reproducible confidences.
        observations = sensor_array.observe_targets(
            drone_positions, timestamp
        )

        # Build per-drone confidence from all camera observations
        drone_conf = np.zeros(n_drones)
        drone_obs_count = np.zeros(n_drones)

        for obs in observations:
            for idx, det in enumerate(obs.detected_objects):
                drone_id = det["drone_id"]
                if drone_id < n_drones and idx < len(obs.confidence_scores):
                    drone_conf[drone_id]      += obs.confidence_scores[idx]
                    drone_obs_count[drone_id] += 1

        # Mean confidence across all cameras that detected each drone
        for d in range(n_drones):
            if drone_obs_count[d] > 0:
                confidences[d, step] = drone_conf[d] / drone_obs_count[d]
            else:
                # Drone not detected by any camera — low confidence
                confidences[d, step] = 0.1

    # Interpolate sampled steps to fill all timesteps
    for d in range(n_drones):
        sampled_steps = list(range(0, n_steps, sample_interval))
        all_steps     = list(range(n_steps))
        confidences[d] = np.interp(
            all_steps,
            sampled_steps,
            confidences[d, sampled_steps]
        )

    mean_conf = confidences[confidences > 0.1].mean() if (confidences > 0.1).any() else 0.0
    print(f"    Mean confidence: {mean_conf:.4f}")
    return confidences


def extract_positions(
    swarm_result: dict,
    confidences: np.ndarray = None
) -> list[dict]:
    """
    Convert swarm simulation output to GIS position format.

    Trajectories are already in real meters over the BCC area (no scaling).
    Uses real sensor confidence scores if provided.
    """
    trajectories = swarm_result["trajectories"]  # [drone, time, 3]
    n_drones, n_steps, _ = trajectories.shape

    positions = []
    for drone_id in range(n_drones):
        for step in range(n_steps):
            x, y, z = trajectories[drone_id, step]

            if confidences is not None:
                conf = float(confidences[drone_id, step])
            else:
                conf = 0.65

            positions.append({
                "drone_id":   drone_id,
                "timestep":   step,
                "x":          float(x),
                "y":          float(y),
                "z":          float(z),
                "confidence": conf,
            })

    return positions


def run_simulation(
    n_drones:      int   = 10,
    pattern:       str   = "perimeter_sweep",
    duration:      float = DEFAULT_DURATION_S,
    timestep:      float = 0.5,
    drone_type:    str   = "small",
    output_prefix: str   = "bcc",
    use_sensor:    bool  = True,
    seed:          int   = None,
    fov_deg:       float = DEFAULT_FOV_DEG,
    altitude_mode: str   = "agl",
    max_mapping_agl: float = DEFAULT_MAX_MAPPING_AGL,
    build_map:     bool  = True,
) -> dict:
    """
    Run the full drone swarm GIS simulation pipeline.

    Parameters
    ----------
    use_sensor    : if True, compute real sensor confidence scores
                    if False, use flat 0.65 default (faster)
    seed          : seeds NumPy's global RNG before anything random happens;
                    the same seed gives identical reports. None = unseeded.
    fov_deg       : camera field of view used for the coverage footprint
    altitude_mode : "agl" (terrain following) or "msl" (fixed altitude)
    build_map     : skip the Folium map (slow for long missions) if False
    """
    if seed is not None:
        np.random.seed(seed)
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
    print(f"Sensor:    {'real confidence' if use_sensor else 'default 0.65'}")
    print(f"Seed:      {seed}")
    print(f"Coverage:  camera footprint, FOV {fov_deg}°, altitude mode {altitude_mode}")
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

    trajectories = swarm_result["trajectories"]
    n_drones_actual, n_steps, _ = trajectories.shape
    print(f"    Generated {n_drones_actual * n_steps:,} position readings "
          f"from {n_drones_actual} drones")

    # ── Step 1b: Compute sensor confidence ───────────────────
    confidences = None
    if use_sensor:
        print("\n[1b] Computing sensor confidence scores...")
        confidences = compute_sensor_confidences(
            trajectories, BCC_SIM_BOUNDS, timestep
        )

    positions = extract_positions(swarm_result, confidences)

    # ── Step 2: Transform coordinates ────────────────────────
    print("\n[2/5] Loading terrain and transforming coordinates...")
    transformer = CoordinateTransformer()

    # ── Step 3: Build sensor heatmap ─────────────────────────
    print("\n[3/5] Building sensor heatmap...")
    heatmap = SensorHeatmap(transformer=transformer)
    heatmap.add_batch_readings(positions)
    raster_paths = heatmap.save_all_rasters(prefix=prefix)

    # ── Step 3b: Camera-footprint coverage ───────────────────
    print("\n[3b] Computing camera-footprint coverage...")
    footprint = FootprintCoverage(
        transformer, fov_deg=fov_deg, altitude_mode=altitude_mode,
        max_mapping_agl=max_mapping_agl,
    )
    footprint.add_trajectories(trajectories)

    # ── Step 4: Analyze coverage ──────────────────────────────
    print("\n[4/5] Analyzing coverage...")
    analyzer = CoverageAnalyzer(transformer=transformer, heatmap=heatmap,
                                footprint=footprint)
    frac_path = heatmap._save_raster(
        analyzer.get_coverage_fraction_raster(), f"{prefix}_footprint_coverage.tif"
    )
    raster_paths["footprint_coverage"] = frac_path
    print(f"Saved footprint coverage raster → {frac_path}")
    analyzer.print_report()
    analyzer.export_coverage_gaps(filename=f"{prefix}_gaps.geojson")
    analyzer.export_flight_paths(positions, filename=f"{prefix}_paths.geojson")

    # ── Step 5: Build interactive map ────────────────────────
    map_path = None
    if build_map:
        print("\n[5/5] Building interactive map...")
        fm       = FoliumMap(transformer=transformer, heatmap=heatmap, analyzer=analyzer)
        map_path = fm.build(positions, filename=f"{prefix}_map.html")

    # ── Save coverage report ──────────────────────────────────
    report      = analyzer.generate_report()
    report["run_config"] = {
        "n_drones": n_drones, "pattern": pattern, "duration_s": duration,
        "timestep_s": timestep, "drone_type": drone_type, "seed": seed,
        "use_sensor": use_sensor, "fov_deg": fov_deg,
        "altitude_mode": altitude_mode, "max_mapping_agl_m": max_mapping_agl,
        "sim_bounds_m": [BCC_SIM_BOUNDS.x_max, BCC_SIM_BOUNDS.y_max],
    }
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
    parser.add_argument("--duration",    type=float, default=DEFAULT_DURATION_S,
                        help="Mission duration in seconds (default 1800 = 30 min)")
    parser.add_argument("--timestep",    type=float, default=0.5)
    parser.add_argument("--drone-type",  type=str,   default="small",
                        choices=["micro", "small", "medium"])
    parser.add_argument("--no-sensor",   action="store_true",
                        help="Skip sensor confidence (faster, uses default 0.65)")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="Random seed; same seed gives identical results")
    parser.add_argument("--fov",         type=float, default=DEFAULT_FOV_DEG,
                        help="Camera field of view for the coverage footprint (deg)")
    parser.add_argument("--altitude-mode", type=str, default="agl",
                        choices=["agl", "msl"])
    parser.add_argument("--no-map",      action="store_true",
                        help="Skip the interactive Folium map")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        n_drones   = args.drones,
        pattern    = args.pattern,
        duration   = args.duration,
        timestep   = args.timestep,
        drone_type = args.drone_type,
        use_sensor = not args.no_sensor,
        seed       = args.seed,
        fov_deg    = args.fov,
        altitude_mode = args.altitude_mode,
        build_map  = not args.no_map,
    )
