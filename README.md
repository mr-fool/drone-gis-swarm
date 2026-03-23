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
│       ├── USGS_13_n41w112_20260113.tif   # Big Cottonwood Canyon DEM (USGS 3DEP, 394MB)
│       └── README_data.md                  # Data provenance and download instructions
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
│   ├── sensor_simulation.py                # Sensor array modeling
│   └── volumetric_detection.py             # 3D detection pipeline
│
├── paper/
│   ├── draft.md                            # Research paper draft
│   ├── figures/                            # Generated map outputs for paper
│   └── references.bib                      # Bibliography
│
├── results/
│   └── simulation_outputs/
│       ├── geojson/                        # Drone path GeoJSON files
│       ├── rasters/                        # Generated GeoTIFF outputs
│       └── maps/                           # Folium HTML map outputs
│
├── tests/
│   ├── __init__.py
│   ├── test_coordinate_transformer.py
│   ├── test_sensor_heatmap.py
│   └── test_coverage_analyzer.py
│
├── app.py                                  # Gradio UI entry point
├── run_gis_simulation.py                   # Headless pipeline runner (no GUI)
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

**Simulation bounding box (clipped from full tile):**

| Parameter | Value |
|-----------|-------|
| South | 40.55° N |
| North | 40.67° N |
| West | -111.81° W |
| East | -111.58° W |
| Elevation range | ~1,443m – ~3,200m |

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
```

---

## Usage

### Run with Gradio UI

```bash
python app.py
```

Opens an interactive interface with drone swarm controls, simulation parameters, and live map output.

### Run headless pipeline

```bash
python run_gis_simulation.py
```

Runs the full pipeline without the GUI. Outputs saved to `results/simulation_outputs/`.

---

## Pipeline Overview

```
Drone swarm simulation (src/)
        ↓
Coordinate transformation: x/y meters → lat/lon (gis/coordinate_transformer.py)
        ↓
Sensor data collection: elevation, coverage, signal readings (gis/sensor_heatmap.py)
        ↓
GIS data fusion: GeoPandas + Rasterio processing (gis/coverage_analyzer.py)
        ↓
Geospatial map output: GeoJSON + GeoTIFF + Folium web map (gis/folium_map.py)
```

---

## Build Order (GIS Layer)

If contributing or extending, build and test modules in this order:

1. `gis/coordinate_transformer.py` — foundation, everything depends on this
2. `gis/sensor_heatmap.py` — requires coordinate transformer
3. `gis/coverage_analyzer.py` — requires heatmap
4. `gis/folium_map.py` — requires all of the above
5. `app.py` — Gradio interface, built last

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

## Author

**ORCID:** https://orcid.org/0009-0002-6160-0993
