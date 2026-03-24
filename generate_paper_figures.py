"""
generate_paper_figures.py

Generates publication-ready figures for the ASPJ paper.

Usage:
    python generate_paper_figures.py
    python generate_paper_figures.py --results results/paper/comparison_table_TIMESTAMP.json
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import rasterio

sys.path.insert(0, os.path.dirname(__file__))
from gis.coordinate_transformer import CoordinateTransformer

FIGURES_DIR = "paper/figures"
DPI         = 300

COLORS = {
    "perimeter_sweep":    "#2196F3",
    "formation_flying":   "#4CAF50",
    "random_dispersal":   "#FF9800",
    "evasive_maneuvers":  "#9C27B0",
    "coordinated_attack": "#F44336",
}


def load_latest_results(results_path=None):
    if results_path and os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    files = sorted(glob.glob("results/paper/comparison_table_*.json"))
    if not files:
        raise FileNotFoundError("No results found. Run generate_paper_results.py first.")
    with open(files[-1]) as f:
        return json.load(f)


def fig1_coverage_comparison(results):
    fig, ax = plt.subplots(figsize=(11, 6))
    patterns   = [r["pattern"].replace("_", "\n") for r in results]
    coverage   = [r["coverage_pct"] for r in results]
    gaps       = [r["gap_pct"] for r in results]
    colors     = [COLORS[r["pattern"]] for r in results]
    x          = np.arange(len(patterns))
    width      = 0.35

    bars1 = ax.bar(x - width/2, coverage, width, label="Covered (%)",
                   color=colors, alpha=0.85)
    bars2 = ax.bar(x + width/2, gaps, width, label="Gap (%)",
                   color=colors, alpha=0.35, hatch="///")

    ax.set_xlabel("Flight Pattern", fontsize=12)
    ax.set_ylabel("Area (%)", fontsize=12)
    ax.set_title(
        f"Figure 1: Coverage vs Gap by Flight Pattern\n"
        f"Big Cottonwood Canyon — {results[0]['n_drones']} drones, "
        f"{results[0]['duration_s']}s",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_coverage_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 → {path}")
    return path


def fig2_elevation_heatmap(results):
    patterns    = [r["pattern"] for r in results]
    band_labels = ["Valley\n(1375–1800m)", "Mid-slope\n(1800–2600m)",
                   "Alpine\n(2600–3500m)"]
    data = np.array([
        [r["valley_coverage_pct"], r["midslope_coverage_pct"],
         r["alpine_coverage_pct"]]
        for r in results
    ])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=15)
    ax.set_xticks(range(3))
    ax.set_xticklabels(band_labels, fontsize=11)
    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels([p.replace("_", " ").title() for p in patterns], fontsize=11)

    for i in range(len(patterns)):
        for j in range(3):
            ax.text(j, i, f"{data[i, j]:.1f}%",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="black" if data[i, j] < 8 else "white")

    plt.colorbar(im, ax=ax, label="Coverage (%)")
    ax.set_title(
        "Figure 2: Coverage by Elevation Band and Flight Pattern\n"
        "Big Cottonwood Canyon, Wasatch Range, Utah",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_elevation_heatmap.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 → {path}")
    return path


def fig3_coverage_vs_gaps(results):
    """Fixed: clear per-point label offsets to prevent overlap."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Manual offsets to keep labels readable
    label_offsets = {
        "perimeter_sweep":    (10,   5),
        "formation_flying":   (-95, -18),
        "random_dispersal":   (10,   5),
        "evasive_maneuvers":  (10,   5),
        "coordinated_attack": (10,  -18),
    }

    for r in results:
        color  = COLORS[r["pattern"]]
        label  = r["pattern"].replace("_", " ").title()
        offset = label_offsets.get(r["pattern"], (10, 5))

        ax.scatter(r["coverage_pct"], r["total_gap_km2"],
                   s=200, color=color, zorder=5, label=label)
        ax.annotate(
            label,
            (r["coverage_pct"], r["total_gap_km2"]),
            textcoords="offset points", xytext=offset,
            fontsize=9, color=color, fontweight="bold"
        )

    ax.set_xlabel("Overall Coverage (%)", fontsize=12)
    ax.set_ylabel("Total Gap Area (km²)", fontsize=12)
    ax.set_title(
        "Figure 3: Coverage Efficiency by Flight Pattern\n"
        "Higher coverage + lower gap area = bottom right (ideal)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_coverage_vs_gaps.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 → {path}")
    return path


def fig4_terrain_overview():
    """Fixed: clean terrain map with clear elevation band shading."""
    ct  = CoordinateTransformer()
    dem = ct.get_dem_array()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: terrain map — plain DEM coloring, no overlay (cleaner)
    im = ax1.imshow(
        dem, cmap="terrain",
        extent=[ct.bounds["west"], ct.bounds["east"],
                ct.bounds["south"], ct.bounds["north"]],
        aspect="auto", origin="upper"
    )
    cbar = plt.colorbar(im, ax=ax1, label="Elevation (m)", shrink=0.85)
    ax1.set_xlabel("Longitude", fontsize=11)
    ax1.set_ylabel("Latitude", fontsize=11)
    ax1.set_title(
        "Big Cottonwood Canyon DEM\nUSGS 3DEP 1/3 Arc-Second, tile n41w112",
        fontsize=12, fontweight="bold"
    )

    # Add elevation band contour lines instead of overlay
    lats = np.linspace(ct.bounds["south"], ct.bounds["north"], dem.shape[0])
    lons = np.linspace(ct.bounds["west"],  ct.bounds["east"],  dem.shape[1])
    LON, LAT = np.meshgrid(lons, lats)
    ax1.contour(LON, LAT, dem, levels=[1800, 2600],
                colors=["#1565C0", "#E65100"], linewidths=1.5,
                linestyles="--")
    ax1.text(-111.79, 40.565, "Valley", color="#1565C0",
             fontsize=9, fontweight="bold")
    ax1.text(-111.79, 40.595, "Mid-slope", color="#E65100",
             fontsize=9, fontweight="bold")
    ax1.text(-111.79, 40.645, "Alpine", color="#6A1B9A",
             fontsize=9, fontweight="bold")

    # Right: clean elevation histogram with band shading
    valid = dem[dem > -9000].flatten()
    ax2.hist(valid, bins=60, orientation="horizontal",
             color="#5c85d6", alpha=0.85, edgecolor="white", linewidth=0.2)

    # Shade elevation bands
    ax2.axhspan(1375, 1800, alpha=0.15, color="#1565C0", label="Valley")
    ax2.axhspan(1800, 2600, alpha=0.15, color="#E65100", label="Mid-slope")
    ax2.axhspan(2600, 3500, alpha=0.15, color="#6A1B9A", label="Alpine")
    ax2.axhline(1800, color="#1565C0", linewidth=2, linestyle="--")
    ax2.axhline(2600, color="#E65100", linewidth=2, linestyle="--")

    # Band labels
    ax2.text(ax2.get_xlim()[1] * 0.02 if ax2.get_xlim()[1] > 0 else 5000,
             1550, "Valley\n(1375–1800m)", color="#1565C0",
             fontsize=9, fontweight="bold", va="center")
    ax2.text(5000, 2150, "Mid-slope\n(1800–2600m)", color="#E65100",
             fontsize=9, fontweight="bold", va="center")
    ax2.text(5000, 3000, "Alpine\n(2600–3500m)", color="#6A1B9A",
             fontsize=9, fontweight="bold", va="center")

    ax2.set_xlabel("Pixel count", fontsize=11)
    ax2.set_ylabel("Elevation (m)", fontsize=11)
    ax2.set_title("Elevation Distribution\nby Zone", fontsize=12,
                  fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Figure 4: Simulation Terrain — Big Cottonwood Canyon, Utah",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_terrain_overview.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 → {path}")
    return path


def fig5_coverage_rasters(results):
    """Detection confidence rasters showing flight path geometry per pattern."""
    raster_data = []
    for r in results:
        pattern_glob = glob.glob(
            f"results/simulation_outputs/rasters/*{r['pattern']}*confidence.tif"
        )
        if pattern_glob:
            with rasterio.open(sorted(pattern_glob)[-1]) as src:
                data = src.read(1)
                raster_data.append((r["pattern"], data))

    if not raster_data:
        print("No raster files found — skipping Figure 5.")
        return None

    n    = len(raster_data)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, (pattern, data) in enumerate(raster_data):
        row, col = divmod(idx, cols)
        ax       = axes[row][col]

        masked = np.ma.masked_where(data == 0, data)
        im = ax.imshow(masked, cmap="plasma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(pattern.replace("_", "\n").title(),
                     fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Confidence", shrink=0.8)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    plt.suptitle(
        "Figure 5: Detection Confidence Rasters by Flight Pattern\n"
        "Big Cottonwood Canyon — shows spatial coverage geometry per pattern",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_coverage_rasters.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 → {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 55)
    print("Generating paper figures...")
    print("=" * 55)

    results = load_latest_results(args.results)
    print(f"Loaded results for {len(results)} patterns\n")

    paths = [
        fig1_coverage_comparison(results),
        fig2_elevation_heatmap(results),
        fig3_coverage_vs_gaps(results),
        fig4_terrain_overview(),
        fig5_coverage_rasters(results),
    ]

    print("\n" + "=" * 55)
    print("All figures saved to paper/figures/")
    print("=" * 55)
    for p in paths:
        if p:
            print(f"  {p}")
