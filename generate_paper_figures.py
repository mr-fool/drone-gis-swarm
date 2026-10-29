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

PATTERN_NAMES = {
    "perimeter_sweep":    "Perimeter Sweep",
    "formation_flying":   "Formation Traverse",
    "random_dispersal":   "Stochastic Dispersal",
    "evasive_maneuvers":  "Sinusoidal Dispersal",
    "coordinated_attack": "Centripetal Convergence",
}

def pattern_label(key):
    return PATTERN_NAMES.get(key, key.replace("_", " ").title())


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
    patterns   = [pattern_label(r["pattern"]) for r in results]
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
        f"Coverage vs Gap by Flight Pattern\n"
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
    ax.set_yticklabels([pattern_label(p) for p in patterns], fontsize=11)

    for i in range(len(patterns)):
        for j in range(3):
            ax.text(j, i, f"{data[i, j]:.1f}%",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="black" if data[i, j] < 8 else "white")

    plt.colorbar(im, ax=ax, label="Coverage (%)")
    ax.set_title(
        "Coverage by Elevation Band and Flight Pattern\n"
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
    """Coverage vs gap area, with labels placed from the data (no hand-tuned
    offsets, which broke whenever the results changed)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    xs = np.array([r["coverage_pct"] for r in results], dtype=float)
    ys = np.array([r["total_gap_km2"] for r in results], dtype=float)

    for r in results:
        ax.scatter(r["coverage_pct"], r["total_gap_km2"], s=200,
                   color=COLORS[r["pattern"]], zorder=5,
                   label=pattern_label(r["pattern"]))

    # Pad the axes so labels stay inside the plotting area
    xpad = 0.08 * max(np.ptp(xs), 1.0); ypad = 0.10 * max(np.ptp(ys), 1.0)
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    fig.canvas.draw()

    # Work in display (pixel) coordinates to judge crowding
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    ax_bbox = ax.get_window_extent()
    sides = ["right"] * len(results)
    vshift = [0] * len(results)
    for i in range(len(results)):
        # Points in the right third get their label on the left
        if pts[i, 0] > ax_bbox.x0 + 0.66 * ax_bbox.width:
            sides[i] = "left"
        for j in range(i + 1, len(results)):
            dx = pts[j, 0] - pts[i, 0]; dy = pts[j, 1] - pts[i, 1]
            if abs(dx) < 160 and abs(dy) < 40:
                # Crowded pair: split the labels to opposite sides and
                # nudge them apart vertically
                left, right = (i, j) if dx > 0 else (j, i)
                sides[left], sides[right] = "left", "right"
                lower, upper = (i, j) if dy > 0 else (j, i)
                vshift[lower] -= 8; vshift[upper] += 8

    for k, r in enumerate(results):
        left = sides[k] == "left"
        ax.annotate(
            pattern_label(r["pattern"]),
            (xs[k], ys[k]),
            textcoords="offset points",
            xytext=(-12 if left else 12, vshift[k]),
            ha="right" if left else "left", va="center",
            fontsize=9, color=COLORS[r["pattern"]], fontweight="bold",
        )

    ax.set_xlabel("Overall Coverage (%)", fontsize=12)
    ax.set_ylabel("Total Gap Area (km²)", fontsize=12)
    ax.set_title(
        "Coverage Efficiency by Flight Pattern\n"
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

    # Elevation band boundaries as contour lines.
    # DEM row 0 is the NORTH edge (imshow origin="upper"), so latitude must
    # run north -> south down the rows. (It previously ran south -> north,
    # which drew the contours mirrored top-to-bottom.) Pixel centers are used.
    n_rows, n_cols = dem.shape
    dlat = (ct.bounds["north"] - ct.bounds["south"]) / n_rows
    dlon = (ct.bounds["east"]  - ct.bounds["west"])  / n_cols
    lats = ct.bounds["north"] - (np.arange(n_rows) + 0.5) * dlat
    lons = ct.bounds["west"]  + (np.arange(n_cols) + 0.5) * dlon
    LON, LAT = np.meshgrid(lons, lats)
    ax1.contour(LON, LAT, dem, levels=[1800, 2600],
                colors=["#1565C0", "#E65100"], linewidths=1.5,
                linestyles="--")
    # Legend instead of hand-placed text labels (those sat on the wrong
    # terrain: the "Alpine" label was on ~1,900 m ground)
    from matplotlib.lines import Line2D
    ax1.legend(
        handles=[
            Line2D([], [], color="#1565C0", linestyle="--", linewidth=1.5,
                   label="1,800 m (valley / mid-slope)"),
            Line2D([], [], color="#E65100", linestyle="--", linewidth=1.5,
                   label="2,600 m (mid-slope / alpine)"),
        ],
        loc="lower right", fontsize=8, framealpha=0.85,
    )

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
        "Simulation Terrain — Big Cottonwood Canyon, Utah",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_terrain_overview.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 → {path}")
    return path


def fig5_coverage_rasters(results):
    """Ground resolution (GSD) rasters per pattern: where each pattern imaged
    the ground, and how finely. Uses the rasters written by the same runs as
    the results table (matched through each run's report path)."""
    raster_data = []
    for r in results:
        prefix = os.path.basename(r.get("report_path", "")).replace("_report.json", "")
        path = os.path.join("results/simulation_outputs/rasters", f"{prefix}_gsd_cm.tif")
        if prefix and os.path.exists(path):
            with rasterio.open(path) as src:
                raster_data.append((r["pattern"], src.read(1)))

    if not raster_data:
        print("No GSD rasters found for these results — skipping Figure 5. "
              "Re-run generate_paper_results.py first.")
        return None

    vmax = max(float(np.percentile(d[d > 0], 98)) for _, d in raster_data if (d > 0).any())

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
        ax.imshow(np.zeros_like(data), cmap="Greys", vmin=0, vmax=1, aspect="auto")
        im = ax.imshow(masked, cmap="viridis_r", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(pattern_label(pattern), fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="GSD (cm/pixel)", shrink=0.8)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    plt.suptitle(
        "Ground Resolution by Flight Pattern (lower = finer; white = not imaged)\n"
        "Big Cottonwood Canyon — mean finest GSD per 0.001° cell",
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
