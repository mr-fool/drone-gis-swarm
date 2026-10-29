# drone-gis-swarm

A Python simulation framework that fuses multi-drone swarm data with real-world geospatial intelligence (GIS) pipelines. Drones autonomously collect terrain data over Big Cottonwood Canyon, Utah, which is then processed into georeferenced maps, coverage heatmaps, and interactive visualizations using industry-standard GIS tools.

**Use cases:** environmental monitoring, terrain mapping, search and rescue planning, autonomous surveying, and situational awareness applications.

---

## What This Project Does

Simulates a swarm of drones flying over real terrain (USGS elevation data, Big Cottonwood Canyon, Utah), collecting sensor readings, and fusing that data into a live geospatial map. The output is a set of GIS artifacts — GeoJSON flight paths, GeoTIFF rasters, and an interactive Folium web map — that can be loaded into any GIS platform including QGIS and ArcGIS.

---

## Repository Structure

```
drone-gis-swarm/
├── data/
│   └── raw/
│       └── USGS_13_n41w112_20260113.tif   # Big Cottonwood Canyon DEM (USGS 3DEP, 394MB, Git LFS)
│
├── gis/                                    # GIS pipeline — primary contribution
│   ├── __init__.py
│   ├── coordinate_transformer.py           # Converts simulation x/y meters → real lat/lon
│   │                                       # Clips DEM to Big Cottonwood Canyon bounds
│   ├── sensor_heatmap.py                   # Visit-count and altitude rasters
│   ├── coverage_analyzer.py                # Gap detection and coverage statistics
│   ├── footprint_coverage.py               # Camera-footprint (area) coverage at DEM resolution
│   └── folium_map.py                       # Generates interactive web map output
│
├── src/                                    # Drone swarm simulation core
│   ├── __init__.py
│   ├── drone_trajectory_generator.py       # Flight path generation
│   └── mission_orientation.py              # Start edge / heading of a mission
│
├── paper/
│   ├── draft.md                            # Research paper draft
│   ├── figures/                            # Publication-ready figures (300 DPI PNG)
│   └── references.bib                      # Bibliography
│
├── results/
│   ├── paper/                              # Main results — all 5 flight patterns
│   │   ├── comparison_table_*.json         # Coverage comparison table
│   │   └── comparison_table_*.csv          # CSV version for analysis
│   ├── sensitivity/                        # Sensitivity study results
│   │   └── sensitivity_results_*.json
│   └── simulation_outputs/
│       ├── geojson/                        # Drone path GeoJSON files
│       ├── rasters/                        # Generated GeoTIFF outputs
│       └── maps/                           # Folium HTML map outputs
│
├── tests/test_coverage_metric.py          # Coverage-metric regression tests
├── app.py                                  # Gradio UI entry point
├── run_gis_simulation.py                   # Headless pipeline runner (no GUI)
├── generate_paper_results.py               # Runs all 5 patterns, saves comparison table
├── generate_paper_figures.py               # Generates publication-ready figures from results
├── generate_sensitivity_study.py           # Sensitivity analysis — swarm size, duration, drone type
├── requirements_gis.txt                    # GIS dependencies
├── requirements.txt                        # Swarm simulation dependencies
└── README.md                               # This file
```

---

## Terrain Data

**Dataset:** USGS 3DEP 1/3 Arc-Second Digital Elevation Model
**Tile:** USGS_13_n41w112_20260113
**Published:** 2026-01-13
**Coverage:** Big Cottonwood Canyon, Wasatch Range, Utah
**Resolution:** ~10 meters (1/3 arc-second)
**Format:** GeoTIFF (Cloud Optimized)
**File size:** 394 MB
**Source:** https://apps.nationalmap.gov/downloader/
**License:** Public domain, no use restrictions

> The TIF file is stored via Git LFS.
> Run `git lfs pull` after cloning to download it.

**Coordinate Reference System (CRS):** `EPSG:4269` (NAD83)
> Note: The USGS DEM uses NAD83 (EPSG:4269), not WGS84 (EPSG:4326).
> The simulation works directly in the DEM's coordinates. GeoTIFF, GeoJSON and
> Folium outputs are labeled WGS84 (EPSG:4326), as those formats expect, but no
> datum transformation is applied, so their coordinates are NAD83 values.
> The two datums differ by about 1–2 m in the continental US, far below the
> 10 m DEM resolution and the ~84 m × 111 m analysis grid, so this does not
> affect the results.

**Simulation bounding box (clipped from full tile):**

| Parameter | Value |
|-----------|-------|
| South | 40.55° N |
| North | 40.67° N |
| West | -111.81° W |
| East | -111.58° W |
| Elevation min | 1,375.6m |
| Elevation max | 3,500.3m |
| DEM shape | 2,484 × 1,296 pixels |

---

## Installation

```bash
git clone https://github.com/mr-fool/drone-gis-swarm
cd drone-gis-swarm

# Pull the DEM file via Git LFS
git lfs pull

# Install swarm simulation dependencies
pip install -r requirements.txt

# Install GIS dependencies
pip install -r requirements_gis.txt
```

**GIS dependencies** (`requirements_gis.txt`):

```
rasterio
geopandas
shapely
folium
gradio
matplotlib
numpy
pyproj
pandas
```

---

## Usage

### Run with Gradio UI

```bash
python app.py
```

Opens an interactive interface at `http://localhost:7860` with drone swarm controls, simulation parameters, live coverage report, and downloadable output files.

### Recommended workflow — full paper pipeline

```bash
# Step 1 — run a single simulation and verify outputs
python run_gis_simulation.py --drones 20 --pattern perimeter_sweep --duration 1800 --seed 42

# Step 2 — run all 5 patterns and generate comparison table
python generate_paper_results.py

# Step 3 — generate publication figures from results
python generate_paper_figures.py

# Step 4 — run sensitivity study (swarm size, duration, drone type)
python generate_sensitivity_study.py

# Step 5 — replicated runs with confidence intervals and tests (~30 min on 2 cores)
python generate_replicate_study.py
```

### run_gis_simulation.py options

```bash
# Default: 10 drones, perimeter_sweep, 30 minutes, orientation 0 (west -> east)
python run_gis_simulation.py

# Custom parameters
python run_gis_simulation.py --drones 20 --pattern formation_flying --duration 1800 --seed 42

# Other start edge / heading (0-7), camera and resolution settings
python run_gis_simulation.py --orientation 4 --fov 60 --image-width 1920 --gsd-threshold 15

# Skip the interactive map (faster)
python run_gis_simulation.py --no-map

# Available patterns (internal code names → paper names)
# perimeter_sweep      → Perimeter Sweep
# formation_flying     → Formation Traverse
# random_dispersal     → Stochastic Dispersal
# evasive_maneuvers    → Sinusoidal Dispersal
# coordinated_attack   → Centripetal Convergence

# Available drone types
# micro | small | medium
```

### generate_sensitivity_study.py options

```bash
# Run all three studies
python generate_sensitivity_study.py --seed 42

# Run one study at a time
python generate_sensitivity_study.py --study swarm_size
python generate_sensitivity_study.py --study duration
python generate_sensitivity_study.py --study drone_type

```

---

## Pipeline Overview

```
Drone swarm simulation (src/)
        ↓
Coordinate transformation: x/y meters → lat/lon (gis/coordinate_transformer.py)
        ↓
Camera-footprint coverage and ground resolution (GSD) at DEM resolution (gis/footprint_coverage.py)
        ↓
Visit-count and altitude rasters (gis/sensor_heatmap.py)
        ↓
Coverage analysis: gap detection, elevation band breakdown (gis/coverage_analyzer.py)
        ↓
Geospatial output: GeoJSON + GeoTIFF + Folium web map (gis/folium_map.py)
```

---

## Build Order (GIS Layer)

Modules were built and tested in this order:

1. `gis/coordinate_transformer.py` [ok]
2. `gis/sensor_heatmap.py` [ok]
3. `gis/coverage_analyzer.py` [ok]
4. `gis/folium_map.py` [ok]
5. `app.py` [ok]

---

## GIS Stack

| Tool | Purpose |
|------|---------|
| Rasterio | Read/write GeoTIFF, coordinate reference systems |
| GeoPandas | Vector data, GeoJSON, spatial operations |
| Shapely | Geometry operations |
| Folium | Interactive web map output |
| Gradio | UI for demo and portfolio |
| QGIS | Visualization and validation (external) |

---

## Technical Notes

- **Flight pattern naming:** The `FlightPattern` enum in `src/drone_trajectory_generator.py` uses internal code names inherited from a prior project (`coordinated_attack`, `evasive_maneuvers`, `formation_flying`, `random_dispersal`, `perimeter_sweep`). The accompanying ASPJ paper uses descriptive geometric names: Centripetal Convergence, Sinusoidal Dispersal, Formation Traverse, Stochastic Dispersal, and Perimeter Sweep respectively. The code names and paper names refer to the same patterns; the enum values are not renamed to preserve compatibility with existing result files.
- **CRS:** DEM is NAD83 (EPSG:4269). Outputs (GeoTIFF, GeoJSON, Folium) are labeled WGS84 (EPSG:4326) without a datum transformation; the datums differ by ~1–2 m in the continental US, negligible at this resolution.
- **DEM clipping:** `coordinate_transformer.py` clips the full 394MB tile to the BCC bounding box on load — only ~10MB held in memory during simulation.
- **Elevation validated:** Range 1,375.6m–3,500.3m matches known Big Cottonwood Canyon topography.
- **Coordinate scaling:** Swarm simulation runs in real meters over the full BCC area (19.3km × 13.3km); no post-hoc scaling.
- **Mapping quality:** Ground sample distance (GSD) from camera geometry and height above ground; see the GSD notes below.
- **Git LFS:** The 394MB DEM file is stored via Git LFS. Run `git lfs pull` after cloning.

---

## Coverage metric and simulation scale (changed)

- **Real scale.** Trajectories are simulated directly in real meters over the full
  BCC bounding box (~19.3 km x 13.3 km) at the drones' physical speeds. Previously a
  1 km box was simulated and positions were scaled ~19x afterwards, which made
  drones fly at 150–330 m/s and produced an artificial coverage plateau when they
  were clamped against the far edge. Drones now reflect off the boundary;
  perimeter sweep and formation traverse turn around, and perimeter sweep shifts
  one lane (`lane_spacing_m`, default 200 m) per turn (boustrophedon).
- **Area-based coverage.** Coverage is the fraction of ground area imaged by a
  nadir camera (default 60° FOV, footprint radius = AGL·tan(FOV/2)) along each
  drone's continuous path, computed at DEM resolution (`gis/footprint_coverage.py`).
  The old metric (a 0.001° cell is covered if any position sample falls in it) is
  still reported as `legacy_point_sample_coverage_pct` for comparison only.
- **Altitude modes.** `--altitude-mode agl` (default) treats z as height above
  ground (ideal terrain following); `msl` flies at a fixed altitude above the
  highest terrain, so footprint size depends on the terrain below. Occlusion is
  not modeled.
- **Seeds.** `--seed N` on all scripts gives identical results.
- **Default mission:** 1800 s (30 min). Sensitivity durations: 600–2400 s.
- **Tests:** `pytest tests/` checks timestep invariance, altitude sensitivity,
  no boundary pile-up and seed reproducibility.

- **Gap zones** are the 4-connected components of uncovered 0.001° cells
  (at least 4 cells), found on the raster. An empty map is one gap zone.
- **Altitude hold.** Wind is applied horizontally only; previously its vertical
  component made every drone climb ~90 m per 30 minutes.

- **Mission orientation.** `--orientation 0-7` flies any pattern from any edge
  and heading (`src/mission_orientation.py`; 0 = west -> east, the original).
  Wind stays fixed in the world frame.
- **Replicate study.** `python generate_replicate_study.py` runs every pattern and
  sensitivity condition 32 times (seed varies, orientation cycles through all 8),
  and writes means with 95% confidence intervals, paired pattern comparisons
  (paired t and Wilcoxon, Holm-adjusted), within-pattern elevation-band tests,
  per-orientation means and two figures to `results/replicates/` and
  `paper/figures/`.

- **Mapping quality = ground sample distance (GSD).** Each ground pixel keeps
  the finest GSD (cm per image pixel) of any pass that imaged it:
  GSD = 2·AGL·tan(FOV/2) / image width at nadir, growing by 1 + (d/AGL)² toward
  the image edge. Reports give the mean, median and 90th percentile GSD
  overall and per elevation band, plus `quality_coverage_pct`: the share of
  the area imaged at or below `--gsd-threshold` (default 15 cm/pixel). A GSD
  raster (`*_gsd_cm.tif`) is written per run and Figure 5 maps it.
  This replaces the earlier "detection confidence", which came from a ring of
  ground cameras watching the drones (carried over from counter-drone work)
  and did not measure mapping quality; `src/sensor_simulation.py` has been
  removed.

## Known Limitations

These limitations are acknowledged in the paper's methodology and limitations section.

**1. Terrain occlusion not modeled**
The camera footprint assumes every ground point inside it is visible. Canyon walls and ridgelines that would hide parts of the ground from a drone are not accounted for, which overestimates coverage in steep terrain. GSD is computed for flat ground below the drone and ignores slope. Future work: viewshed (ray-casting) against the DEM.

**2. Terrain following is idealized**
Trajectories are generated in real meters without reference to the DEM. In the default `agl` coverage mode, z is treated as height above ground (perfect terrain following); in `msl` mode drones hold a fixed altitude above the highest terrain and the footprint varies with the ground below. Neither mode plans altitude around ridgelines or avoids terrain.

**3. Imaging model is geometric only**
Coverage and GSD come from camera geometry (field of view, pixel count, height above ground). Image overlap needed for photogrammetry, motion blur, lighting and weather are not modeled.
