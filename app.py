"""
app.py

Gradio UI for the Drone Swarm GIS Simulation.
Provides an interactive interface for running simulations and viewing results.

Usage:
    python app.py
"""

import gradio as gr
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from run_gis_simulation import run_simulation, PATTERN_MAP


def format_report(report: dict) -> str:
    """Format the JSON report as readable markdown for Gradio."""
    oc   = report["overall_coverage"]
    gaps = report["coverage_gaps"]
    area = report["simulation_area"]

    lines = [
        "## Simulation Results",
        f"**Location:** {area['location']}",
        f"**Area:** {area['width_m']:.0f}m × {area['height_m']:.0f}m",
        "",
        "### Overall Coverage",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cells | {oc['total_cells']:,} |",
        f"| Covered cells | {oc['covered_cells']:,} |",
        f"| Coverage | {oc['coverage_pct']}% |",
        f"| Mean confidence | {oc['mean_confidence']} |",
        f"| Gap cells | {oc['gap_cells']:,} ({oc['gap_pct']}%) |",
        "",
        "### Coverage by Elevation Band",
        "| Zone | Elevation | Coverage | Confidence |",
        "|------|-----------|----------|------------|",
    ]

    for band, stats in report["elevation_band_coverage"].items():
        lines.append(
            f"| {band} | {stats['elevation_range_m']}m | "
            f"{stats['coverage_pct']}% | {stats['mean_confidence']} |"
        )

    lines += [
        "",
        "### Coverage Gaps",
        f"| Gap zones | {gaps['n_gap_zones']} |",
        f"|-----------|------|",
        f"| Total gap area | {gaps['total_gap_km2']} km² |",
    ]

    return "\n".join(lines)


def run_simulation_ui(
    n_drones:    int,
    pattern:     str,
    duration:    float,
    drone_type:  str,
    progress=gr.Progress()
) -> tuple:
    """
    Gradio-compatible wrapper around run_simulation().
    Returns (report_markdown, map_filepath, raster_filepaths, json_report)
    """
    progress(0, desc="Starting simulation...")

    try:
        progress(0.1, desc="Running swarm simulation...")
        outputs = run_simulation(
            n_drones   = int(n_drones),
            pattern    = pattern,
            duration   = float(duration),
            timestep   = 0.5,
            drone_type = drone_type,
        )

        progress(0.8, desc="Loading results...")

        # Read report
        with open(outputs["report"]) as f:
            report = json.load(f)

        report_md = format_report(report)

        # Collect output files for download
        files = [outputs["map"], outputs["report"]]
        for path in outputs["rasters"].values():
            if os.path.exists(path):
                files.append(path)

        progress(1.0, desc="Done!")

        return (
            report_md,
            outputs["map"],
            files,
            json.dumps(report, indent=2),
        )

    except Exception as e:
        error_msg = f"## Error\n```\n{str(e)}\n```"
        return error_msg, None, [], "{}"


def build_ui() -> gr.Blocks:
    """Build and return the Gradio UI."""

    with gr.Blocks(
        title = "Drone Swarm GIS Simulation",
        theme = gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # Drone Swarm GIS Simulation
        **Big Cottonwood Canyon, Wasatch Range, Utah**

        Simulates a drone swarm flying over real USGS terrain data and
        produces georeferenced GIS outputs — coverage heatmaps, flight
        path GeoJSON, and an interactive web map.
        """)

        with gr.Row():

            # ── Left panel: controls ──────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Simulation Parameters")

                n_drones = gr.Slider(
                    minimum = 5,
                    maximum = 50,
                    value   = 20,
                    step    = 5,
                    label   = "Number of drones",
                )

                pattern = gr.Dropdown(
                    choices = list(PATTERN_MAP.keys()),
                    value   = "perimeter_sweep",
                    label   = "Flight pattern",
                )

                duration = gr.Slider(
                    minimum = 10,
                    maximum = 120,
                    value   = 60,
                    step    = 10,
                    label   = "Duration (seconds)",
                )

                drone_type = gr.Radio(
                    choices = ["micro", "small", "medium"],
                    value   = "small",
                    label   = "Drone type",
                )

                run_btn = gr.Button(
                    "Run Simulation",
                    variant = "primary",
                    size    = "lg",
                )

                gr.Markdown("""
                ---
                **Flight patterns:**
                - `perimeter_sweep` — systematic area coverage
                - `formation_flying` — coordinated group movement
                - `random_dispersal` — independent trajectories
                - `evasive_maneuvers` — irregular zigzag paths
                - `coordinated_attack` — converging from perimeter
                """)

            # ── Right panel: outputs ──────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Results")

                report_md = gr.Markdown(
                    value = "*Run a simulation to see results.*"
                )

                with gr.Accordion("Raw JSON report", open=False):
                    json_out = gr.Code(
                        language = "json",
                        label    = "report.json",
                    )

                gr.Markdown("### Download Outputs")
                files_out = gr.Files(
                    label = "Generated files (map, rasters, GeoJSON)",
                )

                gr.Markdown("""
                > **Tip:** Download the `.html` map file and open it
                > in your browser for the full interactive experience.
                > Load `.tif` files in QGIS or ArcGIS.
                > Load `.geojson` files at geojson.io.
                """)

        # ── Wire up the run button ────────────────────────────
        run_btn.click(
            fn      = run_simulation_ui,
            inputs  = [n_drones, pattern, duration, drone_type],
            outputs = [report_md, gr.State(), files_out, json_out],
        )

        gr.Markdown("""
        ---
        **Data:** USGS 3DEP 1/3 Arc-Second DEM, tile n41w112 (2026-01-13)  
        **CRS:** EPSG:4269 (NAD83) → outputs in EPSG:4326 (WGS84)  
        **Elevation range:** 1,375m – 3,500m  
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        inbrowser   = True,
    )
