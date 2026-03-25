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
│   ├── sensor_heatmap.py                   # Maps sensor readings → raster layer
│   ├── coverage_analyzer.py                # Gap detection and coverage statistics
│   └── folium_map.py                       # Generates interactive web map output
│
├── src/                                    # Drone swarm simulation core
│   ├── __init__.py
│   ├── drone_trajectory_generator.py       # Flight path generation
│   ├── sensor_simulation.py                # Sensor array modeling with real confidence scores
│   └── volumetric_detection.py             # 3D detection pipeline
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
> The difference is less than 1 meter in the continental US and does not
> meaningfully affect simulation results. Acknowledged in paper methodology.
> Folium and GeoJSON outputs use WGS84 (EPSG:4326) as required by those formats.

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
python run_gis_simulation.py --drones 20 --pattern perimeter_sweep --duration 60

# Step 2 — run all 5 patterns and generate comparison table
python generate_paper_results.py

# Step 3 — generate publication figures from results
python generate_paper_figures.py

# Step 4 — run sensitivity study (swarm size, duration, drone type)
python generate_sensitivity_study.py
```

### run_gis_simulation.py options

```bash
# Default: 10 drones, perimeter_sweep, 60 seconds, real sensor confidence
python run_gis_simulation.py

# Custom parameters
python run_gis_simulation.py --drones 20 --pattern formation_flying --duration 90

# Skip sensor confidence for speed
python run_gis_simulation.py --no-sensor

# Available patterns
# perimeter_sweep | formation_flying | random_dispersal | evasive_maneuvers | coordinated_attack

# Available drone types
# micro | small | medium
```

### generate_sensitivity_study.py options

```bash
# Run all three studies (~30 minutes)
python generate_sensitivity_study.py

# Run one study at a time
python generate_sensitivity_study.py --study swarm_size
python generate_sensitivity_study.py --study duration
python generate_sensitivity_study.py --study drone_type

# Skip sensor confidence for speed
python generate_sensitivity_study.py --no-sensor
```

---

## Pipeline Overview

```
Drone swarm simulation (src/)
        ↓
Sensor confidence scoring: VirtualSensorArray — real detection confidence per position
        ↓
Coordinate transformation: x/y meters → lat/lon (gis/coordinate_transformer.py)
        ↓
Sensor heatmap: confidence + visit count → raster layer (gis/sensor_heatmap.py)
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

- **CRS:** DEM is NAD83 (EPSG:4269). Outputs (GeoJSON, Folium) use WGS84 (EPSG:4326). Difference < 1m in continental US.
- **DEM clipping:** `coordinate_transformer.py` clips the full 394MB tile to the BCC bounding box on load — only ~10MB held in memory during simulation.
- **Elevation validated:** Range 1,375.6m–3,500.3m matches known Big Cottonwood Canyon topography.
- **Coordinate scaling:** Swarm simulation uses 1km × 1km space. Positions are scaled to full BCC area (19.3km × 13.3km) before GIS processing.
- **Sensor confidence:** Real detection confidence computed via `VirtualSensorArray` with 8-camera perimeter array. Use `--no-sensor` flag for faster runs with default 0.65 confidence.
- **Git LFS:** The 394MB DEM file is stored via Git LFS. Run `git lfs pull` after cloning.

---

## Known Limitations

These limitations are acknowledged in the paper's methodology and limitations section.

**1. Terrain occlusion not modeled**
The sensor confidence model assumes unobstructed line-of-sight between cameras and drones. Canyon walls and ridgelines that would block detection in reality are not accounted for. This may overestimate detection confidence in areas of steep relief. Future work: incorporate ray-casting against the DEM.

**2. Trajectories are not terrain-following**
Drone flight paths are generated in a normalized 1km × 1km virtual space with simple altitude constraints. The DEM is used to compute altitude above ground level (AGL) after the fact, but drone altitude is not adjusted to maintain constant AGL or avoid terrain obstacles. A drone crossing a 200m ridge at a set altitude of 100m will be 100m AGL on the ridge but 300m AGL in the adjacent valley.

**3. Aspect ratio distortion from coordinate scaling**
The 1km × 1km simulation space is linearly scaled to the 19.3km × 13.3km canyon area, producing a 1:0.69 aspect ratio transformation. Flight pattern geometry is stretched accordingly — a square perimeter sweep in virtual space becomes a rectangle in real space. This is documented but not corrected in the current implementation.

**4. Sensor confidence is single-view not multi-view fused**
Detection confidence is computed per camera observation and averaged. A proper multi-view fusion model would weight confidence by triangulation geometry and camera overlap. The current approach is appropriate for a coverage heatmap but does not reflect optimal sensor fusion.

**5. volumetric_detection.py is not integrated**
`src/volumetric_detection.py` is carried over from the prior JDMS publication on counter-drone detection. It is not part of the GIS pipeline and is not called by `run_gis_simulation.py`. It is retained for reference only.
